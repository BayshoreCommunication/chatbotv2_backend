from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, EmailStr, Field


# ── Type Aliases ─────────────────────────────────────────────────────────────

SubscriptionType = Literal["free", "pro", "enterprise"]

CompanyType = Literal[
    "tech-company",
    "law-firm",
    "healthcare-company",
    "realestate-company",
    "consultancy-company",
    "agency-company",
    "other",
]


# ── Nested Models ─────────────────────────────────────────────────────────────

class TrainData(BaseModel):
    """Tracks the AI training/vector data status for a company.
    
    This must stay in sync with the $set fields written by knowledge_router.py.
    All fields here map directly to `train_data.*` keys in the MongoDB document.
    """
    is_trained:     bool  = False
    score:          float = Field(default=0.0, ge=0.0, le=100.0)
    last_updated:   Optional[datetime] = None
    update_count:   int   = 0
    update_limit:   int   = 10              # max training runs (plan-based)
    entries_stored: int   = 0              # LLM-approved facts in Pinecone
    pages_crawled:  int   = 0
    categories:     list[str] = Field(default_factory=list)
    namespace:      Optional[str] = None   # Pinecone namespace (= company_id)
    history:        list[dict] = Field(default_factory=list)  # TrainRunHistory entries


# ── Main Document Model ───────────────────────────────────────────────────────

class UserModel(BaseModel):
    """Company account document stored in MongoDB."""

    # ── Company Info ──────────────────────────────────────────────────────────
    company_name: str = Field(..., min_length=2, max_length=150)
    company_type: CompanyType = "other"
    company_website: Optional[str] = None

    # ── Auth ──────────────────────────────────────────────────────────────────
    email: EmailStr
    hashed_password: str
    role: str = "organization"
    is_active: bool = True
    is_verified: bool = False          # True after OTP confirmed

    # ── OTP (cleared after verification) ──────────────────────────────────────
    otp_code: Optional[str] = None
    otp_expires_at: Optional[datetime] = None

    # ── Subscription ──────────────────────────────────────────────────────────
    is_subscribed: bool = False
    subscription_type: SubscriptionType = "free"
    subscription_start_date: Optional[datetime] = None
    subscription_end_date: Optional[datetime] = None

    # ── AI / Vector Store ─────────────────────────────────────────────────────
    vector_store_id: Optional[str] = None
    train_data: TrainData = Field(default_factory=TrainData)

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
