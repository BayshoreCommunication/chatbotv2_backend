import logging
import random
import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import jwt
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId

from config import settings
from services.user_service import hash_password, verify_password, serialize_user
from model.user_model import UserModel
from schemas.user import SignupRequest, OTPVerifyRequest, SigninRequest, LoginOTPRequest, LoginOTPVerifyRequest
from utils.email import send_otp_email, send_login_otp_email

logger = logging.getLogger(__name__)


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

    # Password is optional from the client — generate one if not supplied so
    # the account still has credentials (e.g. for the auto sign-in after
    # OTP verification below).
    password = payload.password or secrets.token_urlsafe(16)

    # Upsert into temp_signups (replace if same email tries again)
    await db["temp_signups"].replace_one(
        {"email": payload.email},
        {
            "company_name": payload.company_name,
            "company_type": payload.company_type,
            "company_website": payload.company_website,
            "phone_number": payload.phone_number,
            "email": payload.email,
            "hashed_password": hash_password(password),
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
        logger.error("Failed to send OTP email to %s: %s", payload.email, e, exc_info=True)
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

    otp_expires_at = temp.get("otp_expires_at", datetime.now(timezone.utc))
    if otp_expires_at.tzinfo is None:
        otp_expires_at = otp_expires_at.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > otp_expires_at:
        return {"error": "otp_expired"}

    # ✅ OTP valid — create the real user now
    now = datetime.now(timezone.utc)
    user_doc = UserModel(
        company_name=temp["company_name"],
        company_type=temp["company_type"],
        company_website=temp.get("company_website"),
        phone_number=temp.get("phone_number"),
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
    """Authenticate with email + password. Returns a JWT bearer token.
    Falls back to team_access for members who have no user account."""

    user = await db["users"].find_one({"email": payload.email})

    if not user:
        # Check team_access — active members skip the password requirement
        # because they already verified their email via the invite link.
        from services.team_access_service import signin_team_member
        team_result = await signin_team_member(db, payload.email)
        if team_result.get("error"):
            err = team_result["error"]
            # Surface specific team errors; fall back to generic for unknown email
            if err in ("team_member_not_verified", "account_disabled", "owner_not_found"):
                return {"error": err}
            return {"error": "invalid_credentials"}

        owner = team_result["owner"]
        token = create_access_token({
            "sub":            str(owner["_id"]),
            "email":          owner["email"],
            "role":           owner["role"],
            "team_member":    True,
            "team_email":     team_result["member_email"],
            "team_access_id": team_result["team_access_id"],
        })
        return {
            "access_token":          token,
            "token_type":            "bearer",
            "user_id":               str(owner["_id"]),
            "company_name":          owner["company_name"],
            "role":                  owner["role"],
            "has_paid_subscription": bool(owner.get("has_paid_subscription", False)),
            "subscription_type":     owner.get("subscription_type", "free"),
            "is_team_member":        True,
            "team_member_email":     team_result["member_email"],
            "team_member_name":      team_result["member_name"],
        }

    if not verify_password(payload.password, user["hashed_password"]):
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
        "access_token":       token,
        "token_type":         "bearer",
        "user_id":            str(user["_id"]),
        "company_name":       user["company_name"],
        "role":               user["role"],
        "has_paid_subscription": bool(user.get("has_paid_subscription", False)),
        "subscription_type":  user.get("subscription_type", "free"),
    }


# ── Passwordless sign-in: request OTP ─────────────────────────────────────────

async def request_login_otp(db: AsyncIOMotorDatabase, payload: LoginOTPRequest) -> dict:
    """Email a 6-digit sign-in code to an existing, verified, active account.
    Falls back to team_access collection if email is not a registered user."""

    user = await db["users"].find_one({"email": payload.email})
    if not user:
        # Check if this email belongs to an active team member
        from services.team_access_service import request_team_otp
        team_result = await request_team_otp(db, payload.email)
        if team_result.get("ok"):
            return {"message": f"A sign-in code has been sent to {payload.email}."}
        if team_result.get("error") == "email_send_failed":
            return {"error": "email_send_failed"}
        return {"error": "account_not_found"}
    if not user.get("is_verified"):
        return {"error": "email_not_verified"}
    if not user.get("is_active", True):
        return {"error": "account_disabled"}

    otp = _generate_otp()
    otp_expires = datetime.now(timezone.utc) + timedelta(minutes=10)

    await db["users"].update_one(
        {"_id": user["_id"]},
        {"$set": {"otp_code": otp, "otp_expires_at": otp_expires}},
    )

    try:
        await send_login_otp_email(
            to_email=payload.email,
            company_name=user["company_name"],
            otp=otp,
        )
    except Exception as e:
        logger.error("Failed to send login OTP email to %s: %s", payload.email, e, exc_info=True)
        return {"error": "email_send_failed", "detail": str(e)}

    return {"message": f"A sign-in code has been sent to {payload.email}."}


# ── Passwordless sign-in: verify OTP → JWT ────────────────────────────────────

async def verify_login_otp(db: AsyncIOMotorDatabase, payload: LoginOTPVerifyRequest) -> dict:
    """Validate the sign-in code and return a JWT bearer token, same shape as signin().
    Falls back to team_access collection — team members get the owner's session."""

    user = await db["users"].find_one({"email": payload.email})
    if not user:
        # Try team member OTP verification
        from services.team_access_service import verify_team_otp
        team_result = await verify_team_otp(db, payload.email, payload.otp_code)
        if team_result.get("error"):
            err = team_result["error"]
            if err == "not_team_member":
                return {"error": "account_not_found"}
            return {"error": err}

        owner = team_result["owner"]
        token = create_access_token({
            "sub":            str(owner["_id"]),
            "email":          owner["email"],
            "role":           owner["role"],
            "team_member":    True,
            "team_email":     team_result["member_email"],
            "team_access_id": team_result["team_access_id"],
        })
        return {
            "access_token":          token,
            "token_type":            "bearer",
            "user_id":               str(owner["_id"]),
            "company_name":          owner["company_name"],
            "role":                  owner["role"],
            "has_paid_subscription": bool(owner.get("has_paid_subscription", False)),
            "subscription_type":     owner.get("subscription_type", "free"),
            "is_team_member":        True,
            "team_member_email":     team_result["member_email"],
            "team_member_name":      team_result["member_name"],
        }

    if not user.get("otp_code"):
        return {"error": "otp_not_requested"}
    if user.get("otp_code") != payload.otp_code:
        return {"error": "invalid_otp"}

    otp_expires_at = user.get("otp_expires_at", datetime.now(timezone.utc))
    if otp_expires_at.tzinfo is None:
        otp_expires_at = otp_expires_at.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > otp_expires_at:
        return {"error": "otp_expired"}

    if not user.get("is_active", True):
        return {"error": "account_disabled"}

    # One-time use — clear it so the same code can't be replayed.
    await db["users"].update_one(
        {"_id": user["_id"]},
        {"$set": {"otp_code": None, "otp_expires_at": None}},
    )

    token = create_access_token({
        "sub": str(user["_id"]),
        "email": user["email"],
        "role": user["role"],
    })

    return {
        "access_token":       token,
        "token_type":         "bearer",
        "user_id":            str(user["_id"]),
        "company_name":       user["company_name"],
        "role":               user["role"],
        "has_paid_subscription": bool(user.get("has_paid_subscription", False)),
        "subscription_type":  user.get("subscription_type", "free"),
    }
