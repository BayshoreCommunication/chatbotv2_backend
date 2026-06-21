"""
model/notification_model.py
─────────────────────────────
Pydantic model for notification documents.

MongoDB collection: `notifications`
─────────────────────────────────────
One document per real-time alert raised for a company (new chat session
started, new lead captured, etc.). Read by the dashboard topbar bell.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

NotificationType = Literal[
    "chat_started",
    "lead_captured",
    "subscription_ending",
    "subscription_canceled",
    "payment_failed",
]


class NotificationModel(BaseModel):
    company_id: str
    type: NotificationType
    title: str
    message: str
    lead_id: Optional[str] = None
    session_id: Optional[str] = None
    is_read: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        from_attributes = True
        populate_by_name = True
