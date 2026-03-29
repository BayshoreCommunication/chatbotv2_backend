import random
import string
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import jwt
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId

from config import settings
from services.user_service import hash_password, verify_password, serialize_user
from model.user_model import UserModel
from schemas.user import SignupRequest, OTPVerifyRequest, SigninRequest
from utils.email import send_otp_email


def _generate_otp() -> str:
    return "".join(random.choices(string.digits, k=6))


def create_access_token(data: dict) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({**data, "exp": expire}, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


# ── Step 1: Signup — store temp data, send OTP email ─────────────────────────

async def signup(db: AsyncIOMotorDatabase, payload: SignupRequest) -> dict:
    """
    Store signup data temporarily and send OTP to email.
    User is NOT created yet — only created after OTP is confirmed.
    """
    # Check if email already registered in users
    if await db["users"].find_one({"email": payload.email}):
        return {"error": "email_taken"}

    otp = _generate_otp()
    otp_expires = datetime.now(timezone.utc) + timedelta(minutes=10)

    # Upsert into temp_signups (replace if same email tries again)
    await db["temp_signups"].replace_one(
        {"email": payload.email},
        {
            "company_name": payload.company_name,
            "company_type": payload.company_type,
            "company_website": payload.company_website,
            "email": payload.email,
            "hashed_password": hash_password(payload.password),
            "otp_code": otp,
            "otp_expires_at": otp_expires,
            "created_at": datetime.now(timezone.utc),
        },
        upsert=True,
    )

    # Send OTP email
    try:
        await send_otp_email(
            to_email=payload.email,
            company_name=payload.company_name,
            otp=otp,
        )
    except Exception as e:
        return {"error": "email_send_failed", "detail": str(e)}

    return {
        "message": f"OTP sent to {payload.email}. Please verify to complete registration.",
    }


# ── Step 2: Verify OTP — confirm email, then create the user ─────────────────

async def verify_otp(db: AsyncIOMotorDatabase, payload: OTPVerifyRequest) -> dict:
    """
    Validate OTP from temp_signups.
    On success: create the real user document, remove temp data, return JWT.
    """
    temp = await db["temp_signups"].find_one({"email": payload.email})

    if not temp:
        return {"error": "otp_not_found"}

    if temp.get("otp_code") != payload.otp_code:
        return {"error": "invalid_otp"}

    if datetime.now(timezone.utc) > temp.get("otp_expires_at", datetime.now(timezone.utc)):
        return {"error": "otp_expired"}

    # ✅ OTP valid — create the real user now
    now = datetime.now(timezone.utc)
    user_doc = UserModel(
        company_name=temp["company_name"],
        company_type=temp["company_type"],
        company_website=temp.get("company_website"),
        email=temp["email"],
        hashed_password=temp["hashed_password"],
        is_verified=True,
        created_at=now,
        updated_at=now,
    ).model_dump()

    result = await db["users"].insert_one(user_doc)
    user_id = str(result.inserted_id)

    # Clean up temp record
    await db["temp_signups"].delete_one({"email": payload.email})

    # Auto sign-in: return JWT so user is immediately authenticated
    token = create_access_token({
        "sub": user_id,
        "email": temp["email"],
        "role": user_doc["role"],
    })

    return {
        "message": "Email verified. Your account is ready.",
        "access_token": token,
        "token_type": "bearer",
        "user_id": user_id,
        "company_name": temp["company_name"],
        "role": user_doc["role"],
    }


# ── Step 3: Signin — email + password → JWT ───────────────────────────────────

async def signin(db: AsyncIOMotorDatabase, payload: SigninRequest) -> dict:
    """Authenticate with email + password. Returns a JWT bearer token."""

    user = await db["users"].find_one({"email": payload.email})

    if not user or not verify_password(payload.password, user["hashed_password"]):
        return {"error": "invalid_credentials"}

    if not user.get("is_verified"):
        return {"error": "email_not_verified"}

    if not user.get("is_active", True):
        return {"error": "account_disabled"}

    token = create_access_token({
        "sub": str(user["_id"]),
        "email": user["email"],
        "role": user["role"],
    })

    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": str(user["_id"]),
        "company_name": user["company_name"],
        "role": user["role"],
    }
