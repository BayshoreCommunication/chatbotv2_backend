from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class LeadResponse(BaseModel):
    id: str
    company_id: str
    session_id: str
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    message: Optional[str] = None
    is_contacted: bool = False
    created_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True
