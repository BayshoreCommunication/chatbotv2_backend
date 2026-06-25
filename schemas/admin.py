from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field

AdminRole = Literal["super_admin", "admin", "manager"]


# ── Auth ──────────────────────────────────────────────────────────────────────

class AdminSigninRequest(BaseModel):
    email: EmailStr
    password: str


class AdminTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    admin_id: str
    name: str
    email: str
    role: AdminRole


# ── Admin CRUD (super admin only) ─────────────────────────────────────────────

class AdminCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=150)
    email: EmailStr
    password: str = Field(..., min_length=6)
    role: Literal["admin", "manager"] = "admin"
    # "super_admin" is intentionally not a valid choice here — the single
    # super_admin is fixed and seeded at startup, never created via the API.


class AdminUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=150)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=6)
    role: Optional[Literal["admin", "manager"]] = None
    is_active: Optional[bool] = None


class AdminResponse(BaseModel):
    id: str
    name: str
    email: str
    role: AdminRole
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AdminListResponse(BaseModel):
    admins: list[AdminResponse]
    total: int
    page: int
    page_size: int


# ── Platform stats (cross-company, for the admin dashboard) ──────────────────

class SignupMonth(BaseModel):
    month: str   # e.g. "Jan 2026"
    count: int


class AdminPlatformStats(BaseModel):
    total_companies: int
    active_subscriptions: int
    monthly_revenue: float
    active_widgets: int
    verified_companies: int
    trained_companies: int
    avg_kb_score: float
    plan_distribution: dict[str, int]
    signups_by_month: list[SignupMonth]
