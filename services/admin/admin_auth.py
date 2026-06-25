import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from fastapi import Depends, Header, HTTPException, status
from jose import JWTError, jwt
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from config import settings
from database import get_database
from model.admin_model import AdminModel
from schemas.admin import AdminSigninRequest

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(data: dict) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({**data, "exp": expire}, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


# ── Startup: ensure the fixed super admin account always exists ──────────────

async def seed_super_admin(db: AsyncIOMotorDatabase) -> None:
    """Idempotently ensures the fixed super admin account exists.

    Only creates the document if missing — never overwrites an existing
    password, so a password changed later (e.g. via the update endpoint)
    survives app restarts.
    """
    existing = await db["admins"].find_one({"email": settings.SUPER_ADMIN_EMAIL})
    if existing:
        return

    now = datetime.now(timezone.utc)
    admin_doc = AdminModel(
        name="Super Admin",
        email=settings.SUPER_ADMIN_EMAIL,
        hashed_password=hash_password(settings.SUPER_ADMIN_PASSWORD),
        role="super_admin",
        created_at=now,
        updated_at=now,
    ).model_dump()

    await db["admins"].insert_one(admin_doc)
    logger.info("Seeded fixed super admin account: %s", settings.SUPER_ADMIN_EMAIL)


# ── Sign in ───────────────────────────────────────────────────────────────────

async def signin(db: AsyncIOMotorDatabase, payload: AdminSigninRequest) -> dict:
    """Authenticate an admin/super admin with email + password. Returns a JWT."""

    admin = await db["admins"].find_one({"email": payload.email})

    if not admin or not verify_password(payload.password, admin["hashed_password"]):
        return {"error": "invalid_credentials"}

    if not admin.get("is_active", True):
        return {"error": "account_disabled"}

    token = create_access_token({
        "sub": str(admin["_id"]),
        "email": admin["email"],
        "role": admin["role"],
    })

    return {
        "access_token": token,
        "token_type": "bearer",
        "admin_id": str(admin["_id"]),
        "name": admin["name"],
        "email": admin["email"],
        "role": admin["role"],
    }


# ── Auth dependencies for protected admin routes ──────────────────────────────

async def get_current_admin(
    db: AsyncIOMotorDatabase = Depends(get_database),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> dict[str, Any]:
    if not authorization:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    token = authorization.strip()
    if token.lower().startswith("bearer "):
        token = token.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    admin_id = payload.get("sub")
    if not isinstance(admin_id, str) or not ObjectId.is_valid(admin_id):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject")

    admin = await db["admins"].find_one({"_id": ObjectId(admin_id), "is_active": True})
    if not admin:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Admin not found or inactive")

    return {
        "id": str(admin["_id"]),
        "name": admin.get("name", ""),
        "email": admin.get("email", ""),
        "role": admin.get("role", "admin"),
    }


async def require_super_admin(
    current_admin: dict[str, Any] = Depends(get_current_admin),
) -> dict[str, Any]:
    """Gate for endpoints only the super admin may use (e.g. managing other admins)."""
    if current_admin["role"] != "super_admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Super admin privileges required.")
    return current_admin
