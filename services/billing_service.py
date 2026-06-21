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
        price = getattr(line_data[0], "price", None)
        price_id = getattr(price, "id", None)
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
    return {
        "id":                  getattr(invoice, "id", None),
        "created":             getattr(invoice, "created", None),
        "amount_due":          (getattr(invoice, "amount_due", 0) or 0) / 100,
        "currency":            getattr(invoice, "currency", "usd"),
        "status":              "paid" if getattr(invoice, "status", None) == "paid" else "due",
        "invoice_pdf":         getattr(invoice, "invoice_pdf", None),
        "hosted_invoice_url":  getattr(invoice, "hosted_invoice_url", None),
        "tier":                _tier_from_invoice(invoice),
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
        cards      = [_serialize_payment_method(pm, default_id) for pm in methods.data]
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
