from datetime import datetime
from typing import Literal
from pydantic import BaseModel, EmailStr, Field

AdminRole = Literal["super_admin", "admin", "manager"]


class AdminModel(BaseModel):
    """Admin/staff account document stored in MongoDB (separate from company `users`).

    There is exactly one `super_admin` — the fixed account seeded at startup
    (see services/admin/admin_auth.py:seed_super_admin). Every other document
    has role="admin" or role="manager" — "manager" is just a distinct title,
    it carries the same permissions as "admin".
    """

    name: str = Field(..., min_length=2, max_length=150)
    email: EmailStr
    hashed_password: str
    role: AdminRole = "admin"
    is_active: bool = True

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
