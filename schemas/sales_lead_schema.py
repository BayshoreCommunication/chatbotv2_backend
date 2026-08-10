from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class SalesLeadCreate(BaseModel):
    """Payload submitted from the pricing page's Enterprise 'Contact Sales' modal."""
    full_name: str = Field(..., min_length=2, max_length=150)
    email: EmailStr
    company_name: str = Field(..., min_length=1, max_length=200)
    company_size: str
    phone: Optional[str] = None
    message: Optional[str] = Field(None, max_length=2000)


class SalesLeadStatusUpdate(BaseModel):
    """Admin toggles — connect/unconnect and confirm/unconfirm, independently."""
    is_connected: Optional[bool] = None
    is_confirmed: Optional[bool] = None


class SalesLeadResponse(BaseModel):
    id: str
    full_name: str
    email: EmailStr
    company_name: str
    company_size: str
    phone: Optional[str] = None
    message: Optional[str] = None
    is_connected: bool = False
    is_confirmed: bool = False
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        populate_by_name = True


class SalesLeadsResponse(BaseModel):
    leads: list[SalesLeadResponse]
    total: int
    limit: int
    offset: int
