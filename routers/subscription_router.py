"""
routers/subscription_router.py
────────────────────────────────
Subscription endpoints.

  POST /subscription/checkout/{company_id}   — create Stripe Checkout session
  POST /subscription/portal/{company_id}     — create Stripe billing portal session
  POST /subscription/cancel/{company_id}     — cancel subscription
  GET  /subscription/{company_id}            — get current subscription status
  POST /subscription/webhook                 — Stripe webhook receiver (raw body)
"""

from __future__ import annotations

import logging

import stripe
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel

from config import settings
from database import get_database
from model.subscription_model import SubscriptionResponse
from services.subscription.subscription_service import (
    cancel_subscription,
    create_checkout_session,
    create_portal_session,
    get_subscription,
    handle_webhook_event,
)

router = APIRouter(prefix="/subscription", tags=["Subscription"])
logger = logging.getLogger(__name__)


# ── Request / Response schemas ────────────────────────────────────────────────

class CheckoutRequest(BaseModel):
    tier:         str = "professional"   # starter | professional | enterprise
    billing_cycle: str = "monthly"       # monthly | annual
    success_url:  str
    cancel_url:   str


class PortalRequest(BaseModel):
    return_url: str


class CancelRequest(BaseModel):
    immediately: bool = False   # False = cancel at period end (keeps access)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _raise_service_error(result: dict) -> None:
    """Convert service-layer error dicts into HTTP exceptions."""
    err = result.get("error")
    if not err:
        return
    detail = result.get("detail", err)
    if err == "no_subscription":
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail)
    if err == "invalid_plan":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail)
    raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail)


# ── POST /subscription/checkout/{company_id} ─────────────────────────────────

@router.post(
    "/checkout/{company_id}",
    summary="Create a Stripe Checkout session for the given plan",
)
async def checkout(
    company_id: str,
    payload: CheckoutRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    result = await create_checkout_session(
        db=db,
        company_id=company_id,
        tier=payload.tier,
        billing_cycle=payload.billing_cycle,
        success_url=payload.success_url,
        cancel_url=payload.cancel_url,
    )
    _raise_service_error(result)
    return result   # {"checkout_url": "https://checkout.stripe.com/..."}


# ── POST /subscription/portal/{company_id} ───────────────────────────────────

@router.post(
    "/portal/{company_id}",
    summary="Create a Stripe Customer Portal session (manage plan / payment)",
)
async def portal(
    company_id: str,
    payload: PortalRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    result = await create_portal_session(
        db=db,
        company_id=company_id,
        return_url=payload.return_url,
    )
    _raise_service_error(result)
    return result   # {"portal_url": "https://billing.stripe.com/..."}


# ── POST /subscription/cancel/{company_id} ───────────────────────────────────

@router.post(
    "/cancel/{company_id}",
    summary="Cancel the company's subscription (default: at period end)",
)
async def cancel(
    company_id: str,
    payload: CancelRequest = CancelRequest(),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    result = await cancel_subscription(
        db=db,
        company_id=company_id,
        immediately=payload.immediately,
    )
    _raise_service_error(result)
    return {"ok": True, "message": "Subscription cancellation scheduled."}


# ── GET /subscription/{company_id} ───────────────────────────────────────────

@router.get(
    "/{company_id}",
    summary="Get the current subscription status for a company",
)
async def get_status(
    company_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    doc = await get_subscription(db, company_id)
    if not doc:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No subscription found for this company.",
        )
    # Build safe response (strip Stripe IDs)
    is_active = doc.get("subscription_status") in ("active", "trialing")
    return SubscriptionResponse(
        company_id=doc["company_id"],
        subscription_tier=doc.get("subscription_tier", "professional"),
        subscription_status=doc.get("subscription_status", "active"),
        billing_cycle=doc.get("billing_cycle", "monthly"),
        payment_amount=doc.get("payment_amount", 0.0),
        currency=doc.get("currency", "usd"),
        cancel_at_period_end=doc.get("cancel_at_period_end", False),
        current_period_start=doc.get("current_period_start"),
        current_period_end=doc.get("current_period_end"),
        trial_end=doc.get("trial_end"),
        is_active=is_active,
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


# ── POST /subscription/webhook ────────────────────────────────────────────────
# IMPORTANT: This endpoint reads the RAW request body for Stripe signature
# verification. Do NOT add any body-parsing middleware before this route.

@router.post(
    "/webhook",
    summary="Stripe webhook receiver — verifies signature and dispatches events",
    include_in_schema=False,   # hide from public docs
)
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(..., alias="stripe-signature"),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    raw_body = await request.body()

    # Verify the webhook signature to ensure it came from Stripe
    try:
        event = stripe.Webhook.construct_event(
            payload=raw_body,
            sig_header=stripe_signature,
            secret=settings.STRIPE_WEBHOOK_SECRET,
        )
    except stripe.SignatureVerificationError:
        logger.warning("subscription.webhook.invalid_signature")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid Stripe signature.")
    except Exception as e:
        logger.error("subscription.webhook.parse_error error=%s", e)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Webhook parse error: {e}")

    # Dispatch to service layer — never raise from here (Stripe retries on non-2xx)
    try:
        await handle_webhook_event(db, event)
    except Exception as e:
        logger.exception(
            "subscription.webhook.handler_error event=%s error=%s",
            getattr(event, "type", "unknown"), e,
        )
        # Return 200 anyway so Stripe does not keep retrying a broken event
        return {"received": True, "warning": "Handler error — check server logs."}

    return {"received": True}
