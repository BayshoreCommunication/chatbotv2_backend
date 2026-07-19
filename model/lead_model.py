from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class LeadModel(BaseModel):
    """Captured lead from chatbot conversation."""
    company_id: str                             # which company's chatbot captured this
    session_id: str                             # chat session
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    message: Optional[str] = None             # detected service/issue (e.g. "car accident")
    is_contacted: bool = False                # follow-up status
    # Set only once Calendly confirms a real booking via webhook (see
    # services/appointments/service.py:record_appointment_from_webhook) —
    # never from the chatbot merely offering/sharing a slot.
    appointment_time: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
