from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field


TeamAccessStatus = Literal["pending", "active", "inactive", "revoked"]


class TeamAccessModel(BaseModel):
    """
    Team access record in the `team_access` collection.
    Owner adds a member by name+email → pending → member verifies email → active.
    No separate user account is created; member signs in via OTP and gets
    the owner's JWT so they see the owner's full dashboard.
    """

    owner_id: str = Field(..., description="_id of the user who granted access")

    name:  str      = Field(..., min_length=1, max_length=150)
    email: EmailStr

    status: TeamAccessStatus = "pending"

    # Verification token sent in the email link
    verify_token: Optional[str] = None

    # OTP stored here (team members have no users doc)
    otp_code:       Optional[str]      = None
    otp_expires_at: Optional[datetime] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TeamAccessResponse(BaseModel):
    """Safe outbound representation — no tokens or OTP fields."""

    id:         str
    owner_id:   str
    name:       str
    email:      str
    status:     TeamAccessStatus
    created_at: datetime
    updated_at: datetime


class AddMemberRequest(BaseModel):
    name:  str      = Field(..., min_length=1, max_length=150)
    email: EmailStr


class VerifyTokenRequest(BaseModel):
    token: str


class ToggleStatusRequest(BaseModel):
    status: Literal["active", "inactive"]
