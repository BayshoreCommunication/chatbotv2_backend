from datetime import datetime
from typing import Optional
from pydantic import BaseModel

from model.notification_model import NotificationType


class NotificationResponse(BaseModel):
    id: str
    company_id: str
    type: NotificationType
    title: str
    message: str
    lead_id: Optional[str] = None
    session_id: Optional[str] = None
    is_read: bool = False
    created_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True
