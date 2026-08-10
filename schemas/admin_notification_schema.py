from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class AdminNotificationResponse(BaseModel):
    id: str
    company_id: str
    company_name: str
    company_email: str
    type: str
    title: str
    message: str
    lead_id: Optional[str] = None
    session_id: Optional[str] = None
    is_read: bool = False
    created_at: datetime


class AdminNotificationsResponse(BaseModel):
    notifications: list[AdminNotificationResponse]
    total: int
    limit: int
    offset: int
