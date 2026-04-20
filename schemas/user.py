from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, HttpUrl

SubscriptionType = Literal["free", "starter", "professional", "enterprise"]
CompanyType = Literal[
    "tech-company",
    "law-firm",
    "healthcare-company",
    "realestate-company",
    "consultancy-company",
    "agency-company",
    "other",
]


# ── Nested ────────────────────────────────────────────────────────────────────

class TrainDataSchema(BaseModel):
    history: list[dict] = Field(default_factory=list)
    score: float = 0.0
    last_updated: Optional[datetime] = None
    update_count: int = 0
    update_limit: int = 10
    is_trained: bool = False


# ── Auth Schemas ──────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    company_name: str = Field(..., min_length=2, max_length=150)
    company_type: CompanyType = "other"
    company_website: str = Field(..., description="Company website URL")
    email: EmailStr
    password: str = Field(..., min_length=6)


class OTPVerifyRequest(BaseModel):
    email: EmailStr
    otp_code: str = Field(..., min_length=6, max_length=6)


class SigninRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    company_name: str
    role: str


# ── User CRUD Schemas ─────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    company_name: str = Field(..., min_length=2, max_length=150)
    company_type: CompanyType = "other"
    company_website: Optional[str] = None
    email: EmailStr
    password: str = Field(..., min_length=6)
    role: str = "organization"


class UserUpdate(BaseModel):
    company_name: Optional[str] = Field(None, min_length=2, max_length=150)
    company_type: Optional[CompanyType] = None
    company_website: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=6)
    role: Optional[str] = None
    is_active: Optional[bool] = None
    is_subscribed: Optional[bool] = None
    has_paid_subscription: Optional[bool] = None
    subscription_type: Optional[SubscriptionType] = None
    subscription_start_date: Optional[datetime] = None
    subscription_end_date: Optional[datetime] = None
    vector_store_id: Optional[str] = None
    train_data: Optional[TrainDataSchema] = None


# ── Response Schemas ──────────────────────────────────────────────────────────

class UserResponse(BaseModel):
    id: str
    company_name: str
    company_type: str
    company_website: Optional[str]
    email: str
    role: str
    is_active: bool
    is_verified: bool
    is_subscribed: bool
    has_paid_subscription: bool = False
    subscription_type: str
    subscription_start_date: Optional[datetime]
    subscription_end_date: Optional[datetime]
    vector_store_id: Optional[str]
    train_data: TrainDataSchema
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UserListResponse(BaseModel):
    users: list[UserResponse]
    total: int
    page: int
    page_size: int
