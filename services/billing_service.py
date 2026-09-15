"""
services/billing_service.py
─────────────────────────────
Stripe payment-method and invoice-history logic for the billing dashboard.

Plan/checkout/cancel logic lives in services/subscription/subscription_service.py.
This module covers payment methods and invoice history only — both are
intentionally NOT stored in MongoDB. Card details and invoices are always
fetched live from Stripe so the dashboard never shows stale data.
"""

from __future__ import annotations

import logging
from typing import Any

import stripe
from motor.motor_asyncio import AsyncIOMotorDatabase

from config import settings
from services.subscription.subscription_service import PRICE_MAP

logger = logging.getLogger(__name__)

# Reverse lookup so each invoice can show the plan that was actually billed
# at the time, instead of whatever the company's *current* tier happens to
# be — otherwise switching plans would relabel old, already-paid invoices.
_PRICE_ID_TO_TIER: dict[str, str] = {
    price_id: key.split(":")[0] for key, price_id in PRICE_MAP.items()
}


def _tier_from_invoice(invoice: Any) -> str | None:
    try:
        lines = getattr(invoice, "lines", None)
        line_data = getattr(lines, "data", []) if lines else []
        if not line_data:
            return None
        line = line_data[0]

        # Most direct signal when present: change_subscription_plan tags its
        # invoices/subscription items with metadata={"tier": ..., ...}, which
        # Stripe carries onto the resulting invoice line item's own metadata.
        # Note: .get() on a StripeObject (unlike a plain dict) raises
        # AttributeError in this SDK version — use "in"/bracket access instead,
        # same as the existing _read_metadata() helper in subscription_service.
        line_metadata = getattr(line, "metadata", None)
        if line_metadata and "tier" in line_metadata and line_metadata["tier"]:
            return line_metadata["tier"]

        # Stripe's "Basil" API version (2025-03-31+) moved the line item's
        # price reference from a top-level `price` field to
        # `pricing.price_details.price` (the old field is always None/absent
        # now) — check both so this works regardless of API version. Without
        # this, every invoice's tier lookup silently returns None and the
        # frontend mislabels every historical invoice with the company's
        # *current* tier instead of whatever it actually was at the time.
        pricing = getattr(line, "pricing", None)
        price_details = getattr(pricing, "price_details", None) if pricing else None
        price_id = getattr(price_details, "price", None) if price_details else None

        if not price_id:
            price = getattr(line, "price", None)
            price_id = getattr(price, "id", None) if price else None

        return _PRICE_ID_TO_TIER.get(price_id)
    except Exception:
        return None


async def _get_customer_id(db: AsyncIOMotorDatabase, company_id: str) -> str | None:
    sub_doc = await db["subscriptions"].find_one(
        {"company_id": company_id}, {"stripe_customer_id": 1},
    )
    return (sub_doc or {}).get("stripe_customer_id")


def _serialize_payment_method(pm: Any, default_id: str | None) -> dict:
    card = getattr(pm, "card", None)
    return {
        "id":         pm.id,
        "brand":      getattr(card, "brand", "unknown"),
        "last4":      getattr(card, "last4", "0000"),
        "exp_month":  getattr(card, "exp_month", None),
        "exp_year":   getattr(card, "exp_year", None),
        "is_default": pm.id == default_id,
    }


def _serialize_invoice(invoice: Any) -> dict:
    amount_due     = (getattr(invoice, "amount_due", 0) or 0) / 100
    billing_reason = getattr(invoice, "billing_reason", None)
    # A $0 "subscription_create" invoice is the trial-start invoice Stripe
    # auto-pays immediately — it records the card collection, not a charge.
    is_trial = amount_due == 0 and billing_reason == "subscription_create"
    return {
        "id":                  getattr(invoice, "id", None),
        "created":             getattr(invoice, "created", None),
        "amount_due":          amount_due,
        "amount_paid":         (getattr(invoice, "amount_paid", 0) or 0) / 100,
        "currency":            getattr(invoice, "currency", "usd"),
        "status":              "paid" if getattr(invoice, "status", None) == "paid" else "due",
        "invoice_pdf":         getattr(invoice, "invoice_pdf", None),
        "hosted_invoice_url":  getattr(invoice, "hosted_invoice_url", None),
        "tier":                _tier_from_invoice(invoice),
        "billing_reason":      billing_reason,
        "period_start":        getattr(invoice, "period_start", None),
        "period_end":          getattr(invoice, "period_end", None),
        "is_trial":            is_trial,
    }


# ── Payment methods ───────────────────────────────────────────────────────────

async def _ensure_default_payment_method(
    customer_id: str,
    default_id: str | None,
    methods: list[Any],
) -> str | None:
    """
    Best practice: a customer with at least one saved card should always
    have a default set on the Stripe Customer object — otherwise future
    Checkout sessions / off-session charges have nothing to fall back to,
    and the dashboard has no card to highlight as "default". If Stripe
    never had one set (e.g. a card added without explicitly choosing
    default), auto-promote the first saved card.
    """
    if default_id or not methods:
        return default_id
    promoted_id = methods[0].id
    stripe.Customer.modify(customer_id, invoice_settings={"default_payment_method": promoted_id})
    logger.info(
        "billing.payment_methods.auto_default_set customer_id=%s payment_method_id=%s",
        customer_id, promoted_id,
    )
    return promoted_id


def _dedupe_payment_methods(methods: list[Any], default_id: str | None) -> list[Any]:
    """
    Stripe doesn't dedupe payment methods by card — re-submitting the exact
    same physical card (e.g. re-testing card 4242 4242 4242 4242, or a
    customer re-entering their own card instead of picking the saved one)
    attaches a second, third, ... distinct PaymentMethod object that looks
    identical in the UI. Groups by the card's `fingerprint` (Stripe's stable
    identifier for "this is physically the same card") and detaches every
    duplicate but one, keeping the current default if it's among the
    duplicates, else the most recently created one.
    """
    by_fingerprint: dict[str, list[Any]] = {}
    for pm in methods:
        fingerprint = getattr(getattr(pm, "card", None), "fingerprint", None) or pm.id
        by_fingerprint.setdefault(fingerprint, []).append(pm)

    kept: list[Any] = []
    for fingerprint, group in by_fingerprint.items():
        if len(group) == 1:
            kept.append(group[0])
            continue

        group.sort(key=lambda pm: getattr(pm, "created", 0), reverse=True)
        keep = next((pm for pm in group if pm.id == default_id), group[0])
        kept.append(keep)
        for pm in group:
            if pm.id == keep.id:
                continue
            try:
                stripe.PaymentMethod.detach(pm.id)
                logger.info(
                    "billing.payment_methods.duplicate_detached payment_method_id=%s "
                    "fingerprint=%s kept=%s",
                    pm.id, fingerprint, keep.id,
                )
            except stripe.StripeError:
                logger.warning(
                    "billing.payment_methods.duplicate_detach_failed payment_method_id=%s",
                    pm.id, exc_info=True,
                )
    return kept


async def list_payment_methods(db: AsyncIOMotorDatabase, company_id: str) -> dict:
    stripe.api_key = settings.STRIPE_SECRET_KEY
    customer_id = await _get_customer_id(db, company_id)
    if not customer_id:
        logger.warning("billing.payment_methods.no_customer company_id=%s", company_id)
        return {"error": "no_subscription", "detail": "No billing account found."}

    try:
        customer   = stripe.Customer.retrieve(customer_id)
        default_id = getattr(customer.invoice_settings, "default_payment_method", None)
        methods    = stripe.PaymentMethod.list(customer=customer_id, type="card")
        default_id = await _ensure_default_payment_method(customer_id, default_id, methods.data)
        deduped    = _dedupe_payment_methods(methods.data, default_id)
        cards      = [_serialize_payment_method(pm, default_id) for pm in deduped]
        logger.info("billing.payment_methods.listed company_id=%s count=%s", company_id, len(cards))
        return {"cards": cards}
    except stripe.StripeError as e:
        logger.error("billing.payment_methods.stripe_error company_id=%s error=%s", company_id, e)
        return {"error": "stripe_error", "detail": str(e)}


async def set_default_payment_method(
    db: AsyncIOMotorDatabase,
    company_id: str,
    payment_method_id: str,
) -> dict:
    stripe.api_key = settings.STRIPE_SECRET_KEY
    customer_id = await _get_customer_id(db, company_id)
    if not customer_id:
        return {"error": "no_subscription", "detail": "No billing account found."}

    try:
        stripe.Customer.modify(
            customer_id,
            invoice_settings={"default_payment_method": payment_method_id},
        )
        logger.info(
            "billing.payment_methods.default_set company_id=%s payment_method_id=%s",
            company_id, payment_method_id,
        )
        return {"ok": True}
    except stripe.StripeError as e:
        logger.error("billing.payment_methods.default_error company_id=%s error=%s", company_id, e)
        return {"error": "stripe_error", "detail": str(e)}


async def create_setup_intent(db: AsyncIOMotorDatabase, company_id: str) -> dict:
    """
    Returns a SetupIntent client_secret so the frontend can collect a new
    card via Stripe Elements (PaymentElement) and attach it to the existing
    customer — fully custom UI, no redirect to Stripe's hosted pages. The
    actual card number never reaches our server; Stripe Elements handles it
    directly in a secure iframe.
    """
    stripe.api_key = settings.STRIPE_SECRET_KEY
    customer_id = await _get_customer_id(db, company_id)
    if not customer_id:
        return {"error": "no_subscription", "detail": "No billing account found."}

    try:
        intent = stripe.SetupIntent.create(
            customer=customer_id,
            payment_method_types=["card"],
            usage="off_session",
        )
        logger.info("billing.setup_intent.created company_id=%s intent_id=%s", company_id, intent.id)
        return {"client_secret": intent.client_secret}
    except stripe.StripeError as e:
        logger.error("billing.setup_intent.stripe_error company_id=%s error=%s", company_id, e)
        return {"error": "stripe_error", "detail": str(e)}


async def remove_payment_method(
    db: AsyncIOMotorDatabase,
    company_id: str,
    payment_method_id: str,
) -> dict:
    stripe.api_key = settings.STRIPE_SECRET_KEY
    customer_id = await _get_customer_id(db, company_id)
    if not customer_id:
        return {"error": "no_subscription", "detail": "No billing account found."}

    try:
        customer   = stripe.Customer.retrieve(customer_id)
        default_id = getattr(customer.invoice_settings, "default_payment_method", None)
        if default_id == payment_method_id:
            return {
                "error": "default_payment_method",
                "detail": "Set another card as default before removing this one.",
            }
        stripe.PaymentMethod.detach(payment_method_id)
        logger.info(
            "billing.payment_methods.removed company_id=%s payment_method_id=%s",
            company_id, payment_method_id,
        )
        return {"ok": True}
    except stripe.StripeError as e:
        logger.error("billing.payment_methods.remove_error company_id=%s error=%s", company_id, e)
        return {"error": "stripe_error", "detail": str(e)}


# ── Invoice history ───────────────────────────────────────────────────────────

async def list_invoices(db: AsyncIOMotorDatabase, company_id: str, limit: int = 12) -> dict:
    stripe.api_key = settings.STRIPE_SECRET_KEY
    customer_id = await _get_customer_id(db, company_id)
    if not customer_id:
        return {"error": "no_subscription", "detail": "No billing account found."}

    try:
        invoices = stripe.Invoice.list(customer=customer_id, limit=limit)
        result   = [_serialize_invoice(inv) for inv in invoices.data]
        logger.info("billing.invoices.listed company_id=%s count=%s", company_id, len(result))
        return {"invoices": result}
    except stripe.StripeError as e:
        logger.error("billing.invoices.stripe_error company_id=%s error=%s", company_id, e)
        return {"error": "stripe_error", "detail": str(e)}
