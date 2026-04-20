"""
model/subscription.py
──────────────────────
Pydantic models for company subscription documents.

MongoDB collection: `subscriptions`
─────────────────────────────────────
One document per active/historical subscription per company.
Mirrors Stripe subscription state so the app never needs to call
Stripe for routine access checks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ── Enum-style Literals ───────────────────────────────────────────────────────

SubscriptionTier = Literal[
    "starter",
    "professional",
    "enterprise",
]

SubscriptionStatus = Literal[
    "active",               # paid and running
    "trialing",             # within free trial window
    "past_due",             # payment failed, retrying
    "unpaid",               # payment retries exhausted, access suspended
    "canceled",             # explicitly canceled
    "incomplete",           # initial payment pending
    "incomplete_expired",   # initial payment never completed
    "paused",               # billing paused via Stripe
]

BillingCycle = Literal["monthly", "annual"]


# ── Main Document Model ───────────────────────────────────────────────────────

class SubscriptionModel(BaseModel):
    """
    One document per subscription in the `subscriptions` MongoDB collection.

    Created on Stripe `customer.subscription.created` webhook.
    Updated on every subsequent Stripe subscription webhook event.
    """

    # ── References ────────────────────────────────────────────────────────────
    company_id: str = Field(..., description="References users._id (as string).")

    # ── Stripe IDs ────────────────────────────────────────────────────────────
    stripe_customer_id:     str  = Field(..., description="Stripe customer ID (cus_...).")
    stripe_subscription_id: str  = Field(..., description="Stripe subscription ID (sub_...).")
    stripe_price_id:        Optional[str] = Field(
        None, description="Stripe price ID (price_...) — the exact plan variant."
    )

    # ── Plan ──────────────────────────────────────────────────────────────────
    subscription_tier:  SubscriptionTier = "professional"
    billing_cycle:      BillingCycle     = "monthly"
    payment_amount:     float = Field(
        0.0, ge=0,
        description="Amount charged per billing cycle in major currency units (e.g. 49 = $49).",
    )
    currency: str = Field(
        default="usd",
        min_length=3,
        max_length=3,
        description="ISO 4217 currency code, lowercase (e.g. 'usd', 'gbp').",
    )

    # ── Status ────────────────────────────────────────────────────────────────
    subscription_status:  SubscriptionStatus = "active"
    cancel_at_period_end: bool = Field(
        False,
        description="True when the user has requested cancellation but the period has not ended yet.",
    )
    canceled_at: Optional[datetime] = Field(
        None, description="When the subscription was canceled (set by Stripe webhook)."
    )
    ended_at: Optional[datetime] = Field(
        None, description="When the subscription actually ended after cancel_at_period_end."
    )

    # ── Billing Period ────────────────────────────────────────────────────────
    current_period_start: Optional[datetime] = Field(
        None, description="Start of the current billing period."
    )
    current_period_end: Optional[datetime] = Field(
        None, description="End of the current billing period — access valid until this date."
    )

    # ── Trial ─────────────────────────────────────────────────────────────────
    trial_start: Optional[datetime] = None
    trial_end:   Optional[datetime] = None

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Free-form extras ──────────────────────────────────────────────────────
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Stripe metadata, promo codes, internal notes, etc.",
    )

    # ── Computed helpers ──────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """True when the subscription grants access right now."""
        return self.subscription_status in ("active", "trialing")

    @property
    def is_in_trial(self) -> bool:
        now = datetime.now(timezone.utc)
        return (
            self.subscription_status == "trialing"
            and self.trial_end is not None
            and self.trial_end > now
        )

    @property
    def access_expires_at(self) -> Optional[datetime]:
        """The date after which access should be revoked."""
        if self.subscription_status == "trialing" and self.trial_end:
            return self.trial_end
        return self.current_period_end


# ── Lightweight response schema (safe for frontend) ───────────────────────────

class SubscriptionResponse(BaseModel):
    """Read-only view returned to the frontend — no Stripe IDs exposed."""

    company_id:             str
    subscription_tier:      SubscriptionTier
    subscription_status:    SubscriptionStatus
    billing_cycle:          BillingCycle
    payment_amount:         float
    currency:               str
    cancel_at_period_end:   bool
    current_period_start:   Optional[datetime]
    current_period_end:     Optional[datetime]
    trial_end:              Optional[datetime]
    is_active:              bool
    created_at:             datetime
    updated_at:             datetime
