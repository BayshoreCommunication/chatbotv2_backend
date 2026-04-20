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

logger = logging.getLogger(__name__)

PRICE_MAP: dict[str, str] = {
    "starter:monthly":       "price_1RyxWWFS3P7wS29bZsXvMCOR",
    "starter:annual":        "price_1RyxWWFS3P7wS29bZsXvMCOR",
    "professional:monthly":  "price_1RyxVtFS3P7wS29b940JDA7E",
    "professional:annual":   "price_1SRPfGFS3P7wS29b1LEGA6HR",
    "enterprise:monthly":    "price_1RyxUsFS3P7wS29bjiaTZag4",
    "enterprise:annual":     "price_1SRPh0FS3P7wS29bfAjG9QGZ",
}

PLAN_TRAIN_LIMITS: dict[str, int] = {
    "starter":      5,
    "professional": 20,
    "enterprise":   100,
}


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
    return {
        "company_id":             company_id,
        "stripe_customer_id":     getattr(sub, "customer", None),
        "stripe_subscription_id": getattr(sub, "id", None),
        "stripe_price_id":        price_id,
        "subscription_tier":      tier,
        "billing_cycle":          billing_cycle,
        "payment_amount":         payment_amount,
        "currency":               currency,
        "subscription_status":    getattr(sub, "status", "active"),
        "cancel_at_period_end":   getattr(sub, "cancel_at_period_end", False),
        "canceled_at":            _stripe_ts(getattr(sub, "canceled_at", None)),
        "ended_at":               _stripe_ts(getattr(sub, "ended_at", None)),
        "current_period_start":   _stripe_ts(getattr(sub, "current_period_start", None)),
        "current_period_end":     _stripe_ts(getattr(sub, "current_period_end", None)),
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
    has_paid  = status == "active" and tier != "free"
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

async def create_checkout_session(
    db: AsyncIOMotorDatabase,
    company_id: str,
    tier: str,
    billing_cycle: str,
    success_url: str,
    cancel_url: str,
) -> dict:
    stripe.api_key = settings.STRIPE_SECRET_KEY
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

    tier          = metadata.get("tier", "professional")
    billing_cycle = metadata.get("billing_cycle", "monthly")
    customer_id   = getattr(session, "customer", None)
    amount_total  = (getattr(session, "amount_total", 0) or 0) / 100
    currency      = getattr(session, "currency", "usd") or "usd"
    now           = datetime.now(timezone.utc)

    logger.info("checkout.completed.writing company_id=%s tier=%s cycle=%s sub_id=%s customer_id=%s",
                company_id, tier, billing_cycle, sub_id, customer_id)

    doc = {
        "company_id":             company_id,
        "stripe_customer_id":     customer_id,
        "stripe_subscription_id": sub_id,
        "subscription_tier":      tier,
        "billing_cycle":          billing_cycle,
        "payment_amount":         amount_total,
        "currency":               currency,
        "subscription_status":    "active",
        "cancel_at_period_end":   False,
        "updated_at":             now,
    }
    result = await db["subscriptions"].update_one(
        {"company_id": company_id},
        {"$set": doc, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    logger.info("checkout.completed.subscriptions_write matched=%s upserted_id=%s",
                result.matched_count, result.upserted_id)

    await _sync_users_doc(db, company_id, "active", tier, None)
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

    existing_doc  = await db["subscriptions"].find_one({"company_id": company_id}, {"subscription_tier": 1, "billing_cycle": 1})
    tier          = metadata.get("tier") or (existing_doc.get("subscription_tier") if existing_doc else None) or "professional"
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

    result = await db["subscriptions"].update_one(
        {"company_id": company_id},
        {"$set": doc, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    logger.info("subscription.upsert.db_write matched=%s upserted_id=%s", result.matched_count, result.upserted_id)

    await _sync_users_doc(
        db, company_id, getattr(sub, "status", "active"), tier,
        _stripe_ts(getattr(sub, "current_period_end", None)),
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
    tier = existing.get("subscription_tier", "professional")
    await _sync_users_doc(db, company_id, "active", tier, period_end, period_start)
    logger.info("payment.succeeded.done company_id=%s period_end=%s", company_id, period_end)


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
    tier = existing.get("subscription_tier", "professional")
    await _sync_users_doc(db, company_id, "past_due", tier, existing.get("current_period_end"))
    logger.info("payment.failed.done company_id=%s", company_id)
