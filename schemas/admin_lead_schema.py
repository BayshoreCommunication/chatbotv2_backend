from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr


class AdminLeadResponse(BaseModel):
    id: str
    company_id: str
    company_name: str
    company_email: str
    session_id: str
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    message: Optional[str] = None
    is_contacted: bool = False
    appointment_time: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class AdminLeadsResponse(BaseModel):
    leads: list[AdminLeadResponse]
    total: int
    limit: int
    offset: int


class AdminLeadContactedUpdate(BaseModel):
    is_contacted: bool
