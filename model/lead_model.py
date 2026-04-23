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
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
