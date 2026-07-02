"""
services/subscription/subscription_service.py
───────────────────────────────────────────────
All Stripe + MongoDB subscription business logic.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import stripe
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from config import settings
from model.subscription_model import CONVERSATION_LIMITS
from services import notification_service

logger = logging.getLogger(__name__)

# "enterprise" has no fixed price here — it's custom-priced and negotiated
# directly with sales (see create_checkout_session, which rejects self-serve
# checkout for that tier with a "contact_sales" error).
PRICE_MAP: dict[str, str] = {
    "free:monthly":          "price_1RyxWWFS3P7wS29bZsXvMCOR",
    "free:annual":           "price_1RyxWWFS3P7wS29bZsXvMCOR",
    "professional:monthly":  "price_1RyxVtFS3P7wS29b940JDA7E",
    "professional:annual":   "price_1SRPfGFS3P7wS29b1LEGA6HR",
    "advanced:monthly":      "price_1RyxUsFS3P7wS29bjiaTZag4",
    "advanced:annual":       "price_1SRPh0FS3P7wS29bfAjG9QGZ",
}

# New brand-new subscriptions to these tiers get a free trial before the
# first real charge — Stripe bills the saved card automatically once it
# ends, no separate "free" plan needed. Existing subscribers switching
# plans (change_subscription_plan) never get a trial — full price applies
# immediately, same as today.
TRIAL_DAYS = 14
TRIAL_ELIGIBLE_TIERS: set[str] = {"professional", "advanced"}

PLAN_TRAIN_LIMITS: dict[str, int] = {
    "free":         5,
    "professional": 20,
    "advanced":     100,
    "enterprise":   100,   # custom contract may override manually
}

# "starter" was renamed to "free", and the old fixed-price "enterprise" tier
# ($99/999, price IDs above) was renamed to "advanced" — "enterprise" now
# means the new custom-priced, contact-sales-only tier. Normalize old tier
# values found in existing Stripe metadata / subscription docs onto the
# current names.
_LEGACY_TIER_ALIASES: dict[str, str] = {"starter": "free", "enterprise": "advanced"}


def _normalize_tier(tier: str | None) -> str:
    if not tier:
        return "professional"
    return _LEGACY_TIER_ALIASES.get(tier, tier)


def _mark_free_trial_used(doc: dict, existing_tier: str | None, tier: str) -> None:
    """
    Permanently flag a company's one-time free-tier conversation allotment as
    consumed once it moves off the "free" plan onto a paid tier — so it can
    never be re-granted by downgrading back to "free" later.
    """
    if existing_tier == "free" and tier != "free":
        doc["free_trial_used"] = True


def _apply_free_tier_trial(doc: dict, tier: str) -> None:
    """
    The "free" tier is the company's one-time, one-month free trial. Stamp
    trial_start/trial_end from the plan's billing period and immediately mark
    the allotment as used — it is granted exactly once per company.
    """
    if tier == "free":
        doc["trial_start"]     = doc.get("current_period_start")
        doc["trial_end"]       = doc.get("current_period_end")
        doc["free_trial_used"] = True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stripe_ts(ts: int | None) -> datetime | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _to_object_id(company_id: str):
    try:
        return ObjectId(company_id)
    except Exception:
        return company_id


def _read_metadata(obj: Any) -> dict:
    """Safely read metadata from a Stripe object into a plain dict."""
    raw = getattr(obj, "metadata", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    # StripeObject — convert via to_dict if available, else iterate keys
    try:
        return dict(raw.to_dict()) if hasattr(raw, "to_dict") else {k: raw[k] for k in raw}
    except Exception:
        return {}


def _period_bounds(sub: Any) -> tuple[datetime | None, datetime | None]:
    """
    Stripe moved `current_period_start` / `current_period_end` from the
    Subscription object onto each SubscriptionItem (API versions
    2025-03-31+). Check both locations so this works either way.
    """
    start = getattr(sub, "current_period_start", None)
    end   = getattr(sub, "current_period_end", None)
    if start is None or end is None:
        items_list = getattr(sub, "items", None)
        items_data = getattr(items_list, "data", []) if items_list else []
        if items_data:
            item = items_data[0]
            start = start if start is not None else getattr(item, "current_period_start", None)
            end   = end   if end   is not None else getattr(item, "current_period_end", None)
    return _stripe_ts(start), _stripe_ts(end)


def _sub_doc_from_stripe(
    company_id: str,
    sub: Any,
    tier: str,
    billing_cycle: str,
    payment_amount: float,
    currency: str,
) -> dict:
    now = datetime.now(timezone.utc)
    try:
        price_id = sub["items"].data[0].price.id if sub["items"].data else None
    except Exception:
        price_id = None
    period_start, period_end = _period_bounds(sub)
    return {
        "company_id":             company_id,
        "stripe_customer_id":     getattr(sub, "customer", None),
        "stripe_subscription_id": getattr(sub, "id", None),
        "stripe_price_id":        price_id,
        "subscription_tier":      tier,
        "billing_cycle":          billing_cycle,
        "payment_amount":         payment_amount,
        "currency":               currency,
        "conversation_limit":     CONVERSATION_LIMITS.get(tier),
        "subscription_status":    getattr(sub, "status", "active"),
        "cancel_at_period_end":   getattr(sub, "cancel_at_period_end", False),
        "canceled_at":            _stripe_ts(getattr(sub, "canceled_at", None)),
        "ended_at":               _stripe_ts(getattr(sub, "ended_at", None)),
        "current_period_start":   period_start,
        "current_period_end":     period_end,
        "trial_start":            _stripe_ts(getattr(sub, "trial_start", None)),
        "trial_end":              _stripe_ts(getattr(sub, "trial_end", None)),
        "updated_at":             now,
    }


async def _sync_users_doc(
    db: AsyncIOMotorDatabase,
    company_id: str,
    status: str,
    tier: str,
    period_end: datetime | None,
    period_start: datetime | None = None,
) -> None:
    is_active = status in ("active", "trialing")
    # _sync_users_doc is only called when a real Stripe subscription exists
    # (checkout / subscription / invoice webhooks), so any "active" status
    # here means a paid plan — regardless of which tier it's labeled.
    has_paid  = status == "active"
    now       = datetime.now(timezone.utc)
    fields: dict = {
        "is_subscribed":            is_active,
        "has_paid_subscription":    has_paid,
        "subscription_type":        tier,
        "subscription_end_date":    period_end,
        "train_data.update_limit":  PLAN_TRAIN_LIMITS.get(tier, 10),
        "updated_at":               now,
    }
    if period_start:
        fields["subscription_start_date"] = period_start

    user_oid = _to_object_id(company_id)
    result = await db["users"].update_one({"_id": user_oid}, {"$set": fields})
    logger.info(
        "subscription.sync_user company_id=%s status=%s tier=%s matched=%s",
        company_id, status, tier, result.matched_count,
    )
    if result.matched_count == 0:
        logger.warning(
            "subscription.sync_user.no_match company_id=%s user_oid=%s — "
            "user document not found; check ObjectId conversion",
            company_id, user_oid,
        )


# ── Public API ────────────────────────────────────────────────────────────────

async def create_subscription_intent(
    db: AsyncIOMotorDatabase,
    company_id: str,
    tier: str,
    billing_cycle: str,
) -> dict:
    """
    Fully custom signup flow for a brand-new subscriber with no payment
    method on file yet. Creates the Stripe customer (if needed) and a
    Subscription in 'default_incomplete' state, returning a PaymentIntent
    client_secret that the frontend confirms via Stripe Elements
    (PaymentElement) — no redirect to Stripe Checkout.

    For a free ($0) plan, Stripe finalizes the invoice with nothing to
    charge, so there's no payment_intent — the caller should treat
    requires_payment=False as already-complete and just wait for the
    webhook to sync MongoDB.

    professional/advanced get TRIAL_DAYS free before the first real charge
    (trial_settings cancels the subscription if no card is on file by the
    time the trial ends, so it never silently lapses into unpaid access).
    A trial invoice totals $0, so Stripe doesn't return a PaymentIntent for
    it — instead it creates a pending SetupIntent to collect and save the
    card for when billing actually starts. The caller gets that back as
    intent_kind="setup" instead of "payment" so the frontend can call the
    matching Stripe.js confirm method.

    Existing subscribers switching plans should use
    change_subscription_plan instead, which reuses their saved card.
    """
    stripe.api_key = settings.STRIPE_SECRET_KEY
    tier = _normalize_tier(tier)
    logger.info("subscription.create_intent.start company_id=%s tier=%s cycle=%s", company_id, tier, billing_cycle)

    price_key = f"{tier}:{billing_cycle}"
    price_id  = PRICE_MAP.get(price_key)
    if not price_id:
        logger.error("subscription.create_intent.invalid_plan key=%s", price_key)
        return {"error": "invalid_plan", "detail": f"Unknown plan: {price_key}"}

    existing    = await db["subscriptions"].find_one({"company_id": company_id}, {"stripe_customer_id": 1})
    customer_id = (existing or {}).get("stripe_customer_id")

    try:
        if not customer_id:
            user = await db["users"].find_one({"_id": _to_object_id(company_id)}, {"email": 1})
            customer = stripe.Customer.create(
                email=(user or {}).get("email"),
                metadata={"company_id": company_id},
            )
            customer_id = customer.id
            logger.info(
                "subscription.create_intent.customer_created company_id=%s customer_id=%s",
                company_id, customer_id,
            )

        subscription_kwargs: dict[str, Any] = {
            "customer":         customer_id,
            "items":            [{"price": price_id}],
            "payment_behavior": "default_incomplete",
            "payment_settings": {"save_default_payment_method": "on_subscription"},
            "expand":           ["latest_invoice.confirmation_secret", "pending_setup_intent"],
            "metadata":         {"company_id": company_id, "tier": tier, "billing_cycle": billing_cycle},
        }
        if tier in TRIAL_ELIGIBLE_TIERS:
            subscription_kwargs["trial_period_days"] = TRIAL_DAYS
            subscription_kwargs["trial_settings"] = {
                "end_behavior": {"missing_payment_method": "cancel"},
            }

        subscription = stripe.Subscription.create(**subscription_kwargs)

        # Stripe's "Basil" API version (2025-03-31+) removed `payment_intent`
        # from Invoice objects (to support partial payments) — the
        # replacement is `confirmation_secret`, which gives the client_secret
        # directly without needing a separate PaymentIntent lookup.
        latest_invoice = subscription.latest_invoice
        confirmation_secret = getattr(latest_invoice, "confirmation_secret", None) if latest_invoice else None
        client_secret = getattr(confirmation_secret, "client_secret", None) if confirmation_secret else None
        intent_kind = "payment"

        if not client_secret:
            pending_setup_intent = getattr(subscription, "pending_setup_intent", None)
            setup_secret = getattr(pending_setup_intent, "client_secret", None) if pending_setup_intent else None
            if setup_secret:
                client_secret = setup_secret
                intent_kind = "setup"

        logger.info(
            "subscription.create_intent.ok company_id=%s tier=%s sub_id=%s requires_payment=%s intent_kind=%s",
            company_id, tier, subscription.id, bool(client_secret), intent_kind,
        )
        return {
            "subscription_id":  subscription.id,
            "client_secret":    client_secret,
            "requires_payment": bool(client_secret),
            "intent_kind":      intent_kind,
        }
    except stripe.StripeError as e:
        logger.error("subscription.create_intent.stripe_error company_id=%s error=%s", company_id, e)
        return {"error": "stripe_error", "detail": str(e)}


async def create_checkout_session(
    db: AsyncIOMotorDatabase,
    company_id: str,
    tier: str,
    billing_cycle: str,
    success_url: str,
    cancel_url: str,
) -> dict:
    """
    Note: tier="enterprise" always normalizes to "advanced" here (see
    _LEGACY_TIER_ALIASES) and there is intentionally no PRICE_MAP entry for
    the real "enterprise" tier — it's custom-priced and set manually after a
    sales conversation, never purchasable through self-serve checkout.
    """
    stripe.api_key = settings.STRIPE_SECRET_KEY
    tier = _normalize_tier(tier)
    logger.info("subscription.checkout.start company_id=%s tier=%s cycle=%s", company_id, tier, billing_cycle)

    price_key = f"{tier}:{billing_cycle}"
    price_id  = PRICE_MAP.get(price_key)
    if not price_id:
        logger.error("subscription.checkout.invalid_plan key=%s", price_key)
        return {"error": "invalid_plan", "detail": f"Unknown plan: {price_key}"}

    existing    = await db["subscriptions"].find_one({"company_id": company_id}, {"stripe_customer_id": 1})
    customer_id = (existing or {}).get("stripe_customer_id")
    logger.info("subscription.checkout.existing_customer customer_id=%s", customer_id)

    session_kwargs: dict[str, Any] = {
        "mode":              "subscription",
        "line_items":        [{"price": price_id, "quantity": 1}],
        "success_url":       success_url,
        "cancel_url":        cancel_url,
        "metadata":          {"company_id": company_id, "tier": tier, "billing_cycle": billing_cycle},
        "subscription_data": {"metadata": {"company_id": company_id, "tier": tier, "billing_cycle": billing_cycle}},
    }
    if tier in TRIAL_ELIGIBLE_TIERS:
        session_kwargs["subscription_data"]["trial_period_days"] = TRIAL_DAYS
        session_kwargs["subscription_data"]["trial_settings"] = {
            "end_behavior": {"missing_payment_method": "cancel"},
        }
        # Checkout (unlike Elements) needs this explicitly or it won't
        # collect a card at all during a $0-due trial signup.
        session_kwargs["payment_method_collection"] = "always"
    if customer_id:
        session_kwargs["customer"] = customer_id
    else:
        user = await db["users"].find_one({"_id": _to_object_id(company_id)}, {"email": 1})
        if user and user.get("email"):
            session_kwargs["customer_email"] = user["email"]
            logger.info("subscription.checkout.prefill_email email=%s", user["email"])

    try:
        session = stripe.checkout.Session.create(**session_kwargs)
        logger.info("subscription.checkout.session_created company_id=%s session_id=%s url=%s",
                    company_id, session.id, session.url)
        return {"checkout_url": session.url}
    except stripe.StripeError as e:
        logger.error("subscription.checkout.stripe_error company_id=%s error=%s", company_id, e)
        return {"error": "stripe_error", "detail": str(e)}


async def change_subscription_plan(
    db: AsyncIOMotorDatabase,
    company_id: str,
    tier: str,
    billing_cycle: str,
) -> dict:
    """
    Switch plan for a company that already has a Stripe subscription
    (including a free-tier one) — modifies the existing Subscription's
    price directly via the API instead of creating a new subscription or
    Checkout Session.

    No prorated credit for unused time on the old plan: proration_behavior
    is "none" and billing_cycle_anchor is reset to "now", so the customer
    is billed the FULL new-plan price immediately and the next renewal is
    exactly one cycle from today — switching plans is a fresh payment, not
    a top-up of the difference.

    Uses payment_behavior="default_incomplete" so Stripe tells us
    explicitly what happened instead of us assuming a card is on file:
      - If the customer already has a usable default payment method,
        Stripe charges it immediately and the invoice is already "paid" —
        requires_payment comes back False, nothing more for the frontend
        to do.
      - If there's no payment method yet (e.g. upgrading straight from a
        card-less free trial), Stripe creates an invoice that needs
        confirmation — requires_payment comes back True with a
        client_secret, and the frontend shows the same Stripe Elements
        form used for brand-new signups.

    create_subscription_intent (below) is only for a company with no
    Stripe subscription at all yet.
    """
    stripe.api_key = settings.STRIPE_SECRET_KEY
    tier = _normalize_tier(tier)
    logger.info("subscription.change_plan.start company_id=%s tier=%s cycle=%s", company_id, tier, billing_cycle)

    price_key = f"{tier}:{billing_cycle}"
    price_id  = PRICE_MAP.get(price_key)
    if not price_id:
        logger.error("subscription.change_plan.invalid_plan key=%s", price_key)
        return {"error": "invalid_plan", "detail": f"Unknown plan: {price_key}"}

    existing = await db["subscriptions"].find_one({"company_id": company_id})
    if not existing or not existing.get("stripe_subscription_id"):
        # No Stripe subscription yet — create a fresh one instead of erroring
        logger.info("subscription.change_plan.no_existing_sub company_id=%s tier=%s, creating new", company_id, tier)
        return await create_subscription_intent(db, company_id, tier, billing_cycle)

    sub_id = existing["stripe_subscription_id"]
    try:
        sub = stripe.Subscription.retrieve(sub_id)

        if sub.status in ("incomplete", "incomplete_expired", "canceled"):
            # Can't modify these — start fresh with a new subscription instead.
            logger.warning(
                "subscription.change_plan.non_modifiable_sub_replacing company_id=%s old_sub_id=%s status=%s",
                company_id, sub_id, sub.status,
            )
            if sub.status != "canceled":
                try:
                    stripe.Subscription.cancel(sub_id)
                except stripe.StripeError as cancel_err:
                    logger.warning(
                        "subscription.change_plan.cancel_failed sub_id=%s error=%s",
                        sub_id, cancel_err,
                    )
            return await create_subscription_intent(db, company_id, tier, billing_cycle)

        item_id = sub["items"]["data"][0]["id"]

        # When the sub is still in trial, billing_cycle_anchor="now" is rejected
        # by Stripe because the anchor can't precede the trial end date.
        # End the trial immediately so the new plan takes effect right away.
        modify_kwargs: dict = {
            "items": [{"id": item_id, "price": price_id}],
            "proration_behavior": "none",
            "payment_behavior": "default_incomplete",
            "expand": ["latest_invoice.confirmation_secret"],
            "metadata": {"company_id": company_id, "tier": tier, "billing_cycle": billing_cycle},
        }
        if sub.status == "trialing":
            modify_kwargs["trial_end"] = "now"
        else:
            modify_kwargs["billing_cycle_anchor"] = "now"

        updated = stripe.Subscription.modify(sub_id, **modify_kwargs)

        # Stripe's "Basil" API version (2025-03-31+) removed `payment_intent`
        # from Invoice objects — `confirmation_secret` is the replacement and
        # gives the client_secret directly. If the invoice already paid
        # automatically (valid default payment method on file), its status
        # is "paid" and there's nothing left for the frontend to confirm.
        latest_invoice = updated.latest_invoice
        invoice_status = getattr(latest_invoice, "status", None) if latest_invoice else None
        confirmation_secret = getattr(latest_invoice, "confirmation_secret", None) if latest_invoice else None
        client_secret = getattr(confirmation_secret, "client_secret", None) if confirmation_secret else None
        requires_payment = bool(client_secret) and invoice_status != "paid"

        logger.info(
            "subscription.change_plan.ok company_id=%s tier=%s cycle=%s sub_id=%s requires_payment=%s",
            company_id, tier, billing_cycle, sub_id, requires_payment,
        )
        # The customer.subscription.updated webhook will fire next and sync
        # the new tier/price/period into MongoDB — no manual write needed here.
        return {
            "ok": True,
            "requires_payment": requires_payment,
            "client_secret": client_secret if requires_payment else None,
        }
    except stripe.StripeError as e:
        logger.error("subscription.change_plan.stripe_error company_id=%s error=%s", company_id, e)
        return {"error": "stripe_error", "detail": str(e)}


async def create_portal_session(
    db: AsyncIOMotorDatabase,
    company_id: str,
    return_url: str,
) -> dict:
    stripe.api_key = settings.STRIPE_SECRET_KEY
    sub_doc = await db["subscriptions"].find_one({"company_id": company_id})
    if not sub_doc or not sub_doc.get("stripe_customer_id"):
        logger.warning("subscription.portal.no_subscription company_id=%s", company_id)
        return {"error": "no_subscription", "detail": "No active subscription found."}
    try:
        session = stripe.billing_portal.Session.create(
            customer=sub_doc["stripe_customer_id"],
            return_url=return_url,
        )
        logger.info("subscription.portal.created company_id=%s", company_id)
        return {"portal_url": session.url}
    except stripe.StripeError as e:
        logger.error("subscription.portal.stripe_error company_id=%s error=%s", company_id, e)
        return {"error": "stripe_error", "detail": str(e)}


async def cancel_subscription(
    db: AsyncIOMotorDatabase,
    company_id: str,
    immediately: bool = False,
) -> dict:
    stripe.api_key = settings.STRIPE_SECRET_KEY
    sub_doc = await db["subscriptions"].find_one({"company_id": company_id})
    if not sub_doc or not sub_doc.get("stripe_subscription_id"):
        logger.warning("subscription.cancel.no_subscription company_id=%s", company_id)
        return {"error": "no_subscription", "detail": "No active subscription found."}
    sub_id = sub_doc["stripe_subscription_id"]
    try:
        if immediately:
            stripe.Subscription.cancel(sub_id)
        else:
            stripe.Subscription.modify(sub_id, cancel_at_period_end=True)
        logger.info("subscription.cancel.ok company_id=%s immediately=%s", company_id, immediately)
        return {"ok": True}
    except stripe.StripeError as e:
        logger.error("subscription.cancel.stripe_error company_id=%s error=%s", company_id, e)
        return {"error": "stripe_error", "detail": str(e)}


async def get_subscription(
    db: AsyncIOMotorDatabase,
    company_id: str,
) -> dict | None:
    doc = await db["subscriptions"].find_one({"company_id": company_id}, {"_id": 0})
    logger.info("subscription.get company_id=%s found=%s", company_id, doc is not None)
    return doc


# ── Webhook dispatcher ────────────────────────────────────────────────────────

async def handle_webhook_event(db: AsyncIOMotorDatabase, event: Any) -> None:
    etype = event["type"]
    data  = event["data"]["object"]
    logger.info("webhook.dispatch type=%s event_id=%s", etype, getattr(event, "id", "?"))

    if etype == "checkout.session.completed":
        await _on_checkout_completed(db, data)
    elif etype in ("customer.subscription.created", "customer.subscription.updated"):
        await _on_subscription_upsert(db, data)
    elif etype == "customer.subscription.deleted":
        await _on_subscription_deleted(db, data)
    elif etype == "customer.subscription.trial_will_end":
        await _on_trial_will_end(db, data)
    elif etype == "invoice.payment_succeeded":
        await _on_payment_succeeded(db, data)
    elif etype == "invoice.payment_failed":
        await _on_payment_failed(db, data)
    else:
        logger.debug("webhook.ignored type=%s", etype)


# ── Event handlers ────────────────────────────────────────────────────────────

async def _on_checkout_completed(db: AsyncIOMotorDatabase, session: Any) -> None:
    session_id = getattr(session, "id", "?")
    metadata   = _read_metadata(session)
    logger.info("checkout.completed session_id=%s metadata=%s", session_id, metadata)

    company_id = metadata.get("company_id")
    if not company_id:
        logger.warning("checkout.completed.no_company_id session_id=%s", session_id)
        return

    sub_id = getattr(session, "subscription", None)
    if not sub_id:
        logger.warning("checkout.completed.no_sub_id session_id=%s company_id=%s", session_id, company_id)
        return

    tier          = _normalize_tier(metadata.get("tier", "professional"))
    billing_cycle = metadata.get("billing_cycle", "monthly")
    customer_id   = getattr(session, "customer", None)
    amount_total  = (getattr(session, "amount_total", 0) or 0) / 100
    currency      = getattr(session, "currency", "usd") or "usd"
    now           = datetime.now(timezone.utc)

    existing_doc  = await db["subscriptions"].find_one({"company_id": company_id}, {"subscription_tier": 1})
    existing_tier = _normalize_tier(existing_doc["subscription_tier"]) if existing_doc and existing_doc.get("subscription_tier") else None

    # Fetch the full subscription so period/trial dates are filled in right away
    # instead of waiting on a separate customer.subscription.* webhook.
    stripe.api_key = settings.STRIPE_SECRET_KEY
    try:
        sub = stripe.Subscription.retrieve(sub_id)
    except stripe.StripeError as e:
        logger.error("checkout.completed.subscription_retrieve_error sub_id=%s error=%s", sub_id, e)
        sub = None

    period_start, period_end = _period_bounds(sub) if sub else (None, None)
    trial_start  = _stripe_ts(getattr(sub, "trial_start", None)) if sub else None
    trial_end    = _stripe_ts(getattr(sub, "trial_end", None))   if sub else None
    sub_status   = getattr(sub, "status", "active") if sub else "active"

    logger.info("checkout.completed.writing company_id=%s tier=%s cycle=%s sub_id=%s customer_id=%s status=%s",
                company_id, tier, billing_cycle, sub_id, customer_id, sub_status)

    doc = {
        "company_id":             company_id,
        "stripe_customer_id":     customer_id,
        "stripe_subscription_id": sub_id,
        "subscription_tier":      tier,
        "billing_cycle":          billing_cycle,
        "payment_amount":         amount_total,
        "currency":               currency,
        "conversation_limit":     CONVERSATION_LIMITS.get(tier),
        "subscription_status":    sub_status,
        "cancel_at_period_end":   False,
        "current_period_start":   period_start,
        "current_period_end":     period_end,
        "trial_start":            trial_start,
        "trial_end":              trial_end,
        "updated_at":             now,
    }
    _mark_free_trial_used(doc, existing_tier, tier)
    _apply_free_tier_trial(doc, tier)

    set_on_insert: dict[str, Any] = {"created_at": now, "conversations_used": 0}
    if "free_trial_used" not in doc:
        set_on_insert["free_trial_used"] = False

    result = await db["subscriptions"].update_one(
        {"company_id": company_id},
        {"$set": doc, "$setOnInsert": set_on_insert},
        upsert=True,
    )
    logger.info("checkout.completed.subscriptions_write matched=%s upserted_id=%s",
                result.matched_count, result.upserted_id)

    await _sync_users_doc(db, company_id, sub_status, tier, period_end, period_start)
    logger.info("checkout.completed.done company_id=%s", company_id)


async def _on_subscription_upsert(db: AsyncIOMotorDatabase, sub: Any) -> None:
    sub_id   = getattr(sub, "id", "?")
    metadata = _read_metadata(sub)
    logger.info("subscription.upsert sub_id=%s metadata=%s", sub_id, metadata)

    company_id = metadata.get("company_id")
    if not company_id:
        existing = await db["subscriptions"].find_one({"stripe_subscription_id": sub_id})
        if existing:
            company_id = existing["company_id"]
            logger.info("subscription.upsert.fallback_lookup company_id=%s", company_id)
    if not company_id:
        logger.warning("subscription.upsert.no_company_id sub_id=%s", sub_id)
        return

    existing_doc  = await db["subscriptions"].find_one(
        {"company_id": company_id},
        {"subscription_tier": 1, "billing_cycle": 1, "cancel_at_period_end": 1},
    )
    existing_tier = _normalize_tier(existing_doc["subscription_tier"]) if existing_doc and existing_doc.get("subscription_tier") else None
    tier          = _normalize_tier(metadata.get("tier") or (existing_doc.get("subscription_tier") if existing_doc else None))
    billing_cycle = metadata.get("billing_cycle") or (existing_doc.get("billing_cycle") if existing_doc else None) or "monthly"

    items_list = getattr(sub, "items", None)
    items_data = getattr(items_list, "data", []) if items_list else []
    price_obj  = getattr(items_data[0], "price", None) if items_data else None
    amount     = (getattr(price_obj, "unit_amount", 0) or 0) / 100
    currency   = getattr(price_obj, "currency", "usd") or "usd"

    logger.info("subscription.upsert.values company_id=%s tier=%s cycle=%s amount=%s status=%s",
                company_id, tier, billing_cycle, amount, getattr(sub, "status", "?"))

    now = datetime.now(timezone.utc)
    doc = _sub_doc_from_stripe(company_id, sub, tier, billing_cycle, amount, currency)
    doc["updated_at"] = now
    _mark_free_trial_used(doc, existing_tier, tier)
    _apply_free_tier_trial(doc, tier)

    set_on_insert: dict[str, Any] = {"created_at": now, "conversations_used": 0}
    if "free_trial_used" not in doc:
        set_on_insert["free_trial_used"] = False

    result = await db["subscriptions"].update_one(
        {"company_id": company_id},
        {"$set": doc, "$setOnInsert": set_on_insert},
        upsert=True,
    )
    logger.info("subscription.upsert.db_write matched=%s upserted_id=%s", result.matched_count, result.upserted_id)

    await _sync_users_doc(
        db, company_id, getattr(sub, "status", "active"), tier,
        doc["current_period_end"], doc["current_period_start"],
    )

    # Notify only on the False→True transition so re-saving an already-ending
    # subscription doesn't spam a fresh alert on every webhook delivery.
    was_ending = bool(existing_doc.get("cancel_at_period_end")) if existing_doc else False
    now_ending = bool(doc.get("cancel_at_period_end"))
    if now_ending and not was_ending:
        await notification_service.create_subscription_ending_notification(
            db, company_id, doc.get("current_period_end"),
        )

    logger.info("subscription.upsert.done company_id=%s", company_id)


async def _on_subscription_deleted(db: AsyncIOMotorDatabase, sub: Any) -> None:
    sub_id   = getattr(sub, "id", "?")
    metadata = _read_metadata(sub)
    logger.info("subscription.deleted sub_id=%s metadata=%s", sub_id, metadata)

    company_id = metadata.get("company_id")
    if not company_id:
        existing = await db["subscriptions"].find_one({"stripe_subscription_id": sub_id})
        if existing:
            company_id = existing["company_id"]
    if not company_id:
        logger.warning("subscription.deleted.no_company_id sub_id=%s", sub_id)
        return

    now = datetime.now(timezone.utc)
    await db["subscriptions"].update_one(
        {"company_id": company_id},
        {"$set": {
            "subscription_status": "canceled",
            "canceled_at":         _stripe_ts(getattr(sub, "canceled_at", None)) or now,
            "ended_at":            _stripe_ts(getattr(sub, "ended_at", None))    or now,
            "updated_at":          now,
        }},
    )
    user_oid = _to_object_id(company_id)
    await db["users"].update_one(
        {"_id": user_oid},
        {"$set": {
            "is_subscribed":         False,
            "has_paid_subscription": False,
            "subscription_type":     "free",
            "subscription_end_date": now,
            "updated_at":            now,
        }},
    )
    await notification_service.create_subscription_canceled_notification(db, company_id)
    logger.info("subscription.deleted.done company_id=%s", company_id)


async def _on_payment_succeeded(db: AsyncIOMotorDatabase, invoice: Any) -> None:
    sub_id = getattr(invoice, "subscription", None)
    logger.info("payment.succeeded invoice_id=%s sub_id=%s", getattr(invoice, "id", "?"), sub_id)
    if not sub_id:
        return

    existing = await db["subscriptions"].find_one({"stripe_subscription_id": sub_id})
    if not existing:
        logger.warning("payment.succeeded.no_subscription sub_id=%s — subscription doc missing", sub_id)
        return

    company_id   = existing["company_id"]
    period_start = _stripe_ts(getattr(invoice, "period_start", None))
    period_end   = _stripe_ts(getattr(invoice, "period_end", None))
    now          = datetime.now(timezone.utc)

    await db["subscriptions"].update_one(
        {"company_id": company_id},
        {"$set": {
            "subscription_status":  "active",
            "current_period_start": period_start,
            "current_period_end":   period_end,
            "updated_at":           now,
        }},
    )
    tier = _normalize_tier(existing.get("subscription_tier"))
    await _sync_users_doc(db, company_id, "active", tier, period_end, period_start)
    logger.info("payment.succeeded.done company_id=%s period_end=%s", company_id, period_end)


async def _on_trial_will_end(db: AsyncIOMotorDatabase, sub: Any) -> None:
    """Stripe fires this 3 days before a trial ends. Send a reminder notification."""
    sub_id   = getattr(sub, "id", "?")
    metadata = _read_metadata(sub)
    logger.info("trial_will_end sub_id=%s metadata=%s", sub_id, metadata)

    company_id = metadata.get("company_id")
    if not company_id:
        existing = await db["subscriptions"].find_one({"stripe_subscription_id": sub_id})
        if existing:
            company_id = existing["company_id"]
    if not company_id:
        logger.warning("trial_will_end.no_company_id sub_id=%s", sub_id)
        return

    trial_end = _stripe_ts(getattr(sub, "trial_end", None))
    await notification_service.create_trial_ending_notification(db, company_id, trial_end)
    logger.info("trial_will_end.done company_id=%s trial_end=%s", company_id, trial_end)


async def _on_payment_failed(db: AsyncIOMotorDatabase, invoice: Any) -> None:
    sub_id = getattr(invoice, "subscription", None)
    logger.info("payment.failed invoice_id=%s sub_id=%s", getattr(invoice, "id", "?"), sub_id)
    if not sub_id:
        return

    existing = await db["subscriptions"].find_one({"stripe_subscription_id": sub_id})
    if not existing:
        logger.warning("payment.failed.no_subscription sub_id=%s", sub_id)
        return

    company_id = existing["company_id"]
    now        = datetime.now(timezone.utc)
    await db["subscriptions"].update_one(
        {"company_id": company_id},
        {"$set": {"subscription_status": "past_due", "updated_at": now}},
    )
    tier = _normalize_tier(existing.get("subscription_tier"))
    await _sync_users_doc(db, company_id, "past_due", tier, existing.get("current_period_end"))
    await notification_service.create_payment_failed_notification(db, company_id)
    logger.info("payment.failed.done company_id=%s", company_id)
