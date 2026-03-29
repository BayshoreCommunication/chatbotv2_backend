from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class LeadModel(BaseModel):
    """Captured lead from chatbot conversation."""
    company_id: str                         # which company's chatbot captured this
    session_id: str                         # chat session
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    message: Optional[str] = None          # the message that triggered lead capture
    is_contacted: bool = False             # follow-up status
    created_at: datetime = Field(default_factory=datetime.utcnow)
