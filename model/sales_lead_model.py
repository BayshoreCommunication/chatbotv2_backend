from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class SalesLeadModel(BaseModel):
    """Enterprise plan inquiry submitted from the pricing page's Contact Sales modal."""
    full_name: str
    email: EmailStr
    company_name: str
    company_size: str                         # e.g. "1-10" ... "500+" — see COMPANY_SIZE_OPTIONS on the frontend
    phone: Optional[str] = None
    message: Optional[str] = None
    is_connected: bool = False                # sales team has reached out to this lead
    is_confirmed: bool = False                # lead confirmed moving forward with the Enterprise plan
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
