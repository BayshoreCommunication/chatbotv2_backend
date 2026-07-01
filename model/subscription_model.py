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
    "free",
    "professional",
    "advanced",
    # "enterprise" is custom-priced (no fixed Stripe price) — negotiated
    # directly with sales and set manually, not purchasable via checkout.
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


# ── Conversation limits per plan ──────────────────────────────────────────────
# `None` means unlimited. The "free" tier's allotment is a one-time grant per
# company (tracked via `free_trial_used`) — it is not renewed automatically.

CONVERSATION_LIMITS: dict[SubscriptionTier, Optional[int]] = {
    "free":         1000,
    "professional": 5000,
    "advanced":     None,   # unlimited
    "enterprise":   None,   # unlimited (custom contract may override manually)
}


# ── Enterprise custom contract details ────────────────────────────────────────
# Structured, typed record for the manually-negotiated "enterprise" tier —
# kept separate from the free-form `metadata` dict (which stays for truly
# arbitrary stuff like promo codes) so the deal terms stay queryable.

class EnterpriseContractDetails(BaseModel):
    contract_reference: Optional[str] = Field(
        None, description="Internal deal/contract ID, e.g. from a CRM or signed agreement."
    )
    notes: Optional[str] = Field(
        None, description="Free-text notes about the negotiated terms."
    )
    set_by: Optional[str] = Field(
        None, description="Email or user ID of whoever configured this custom plan."
    )
    set_at: Optional[datetime] = Field(
        None, description="When this custom plan was configured."
    )


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

    # ── Conversation usage ───────────────────────────────────────────────────
    conversation_limit: Optional[int] = Field(
        default=CONVERSATION_LIMITS["free"],
        description="Conversation cap for the current plan. None = unlimited (enterprise).",
    )
    conversations_used: int = Field(
        0, ge=0,
        description="Conversations consumed so far (free tier: one-time lifetime total; paid tiers: current period).",
    )
    free_trial_used: bool = Field(
        False,
        description="True once this company has consumed its one-time free-tier conversation allotment.",
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
    collection_method: Literal["charge_automatically", "send_invoice"] = Field(
        "charge_automatically",
        description=(
            "Mirrors Stripe Subscription.collection_method. 'charge_automatically' "
            "bills the default payment method each period; 'send_invoice' requires "
            "the customer to pay each invoice manually. Backs the dashboard's "
            "'Automatic Payments' toggle via stripe.Subscription.modify(...)."
        ),
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

    # ── Enterprise custom contract ───────────────────────────────────────────
    enterprise_contract: Optional[EnterpriseContractDetails] = Field(
        None,
        description=(
            "Populated only when subscription_tier == 'enterprise'. "
            "Structured record of the custom-priced contract, set manually "
            "by an admin after a sales negotiation."
        ),
    )

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
    collection_method:      Literal["charge_automatically", "send_invoice"] = "charge_automatically"
    current_period_start:   Optional[datetime]
    current_period_end:     Optional[datetime]
    trial_end:              Optional[datetime]
    is_active:              bool
    is_in_trial:            bool = False
    conversation_limit:     Optional[int] = None
    conversations_used:     int = 0
    free_trial_used:        bool = False
    created_at:             datetime
    updated_at:             datetime


