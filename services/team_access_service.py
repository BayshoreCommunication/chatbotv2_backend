from __future__ import annotations

import logging
import random
import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from config import settings
from utils.email import send_team_access_email, send_team_access_otp_email

logger = logging.getLogger(__name__)

MAX_MEMBERS = 5
OTP_EXPIRE_MINUTES = 10


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id":         str(doc["_id"]),
        "owner_id":   str(doc["owner_id"]),
        "name":       doc.get("name", ""),
        "email":      doc.get("email", ""),
        "status":     doc.get("status", "pending"),
        "created_at": doc.get("created_at", _now()),
        "updated_at": doc.get("updated_at", _now()),
    }


# ── Owner: add a team member ──────────────────────────────────────────────────

async def add_team_member(
    db: AsyncIOMotorDatabase,
    owner_id: ObjectId,
    name: str,
    email: str,
) -> dict:
    # Enforce member limit (exclude revoked)
    active_count = await db["team_access"].count_documents(
        {"owner_id": owner_id, "status": {"$ne": "revoked"}}
    )
    if active_count >= MAX_MEMBERS:
        return {"error": "limit_reached", "detail": f"Maximum of {MAX_MEMBERS} team members reached."}

    # Block adding an email that belongs to a registered main user
    main_user = await db["users"].find_one({"email": email}, {"_id": 1})
    if main_user:
        return {"error": "is_registered_user", "detail": "This email belongs to a registered account and cannot be added as a team member."}

    # No duplicate active/pending invite to same email under this owner
    existing = await db["team_access"].find_one(
        {"owner_id": owner_id, "email": email, "status": {"$nin": ["revoked"]}}
    )
    if existing:
        status = existing.get("status", "pending")
        if status == "active":
            return {"error": "already_active", "detail": "This member already has active access."}
        return {"error": "already_exists", "detail": "A pending invite already exists for this email."}

    token = secrets.token_urlsafe(32)
    now   = _now()
    doc   = {
        "owner_id":     owner_id,
        "name":         name,
        "email":        email,
        "status":       "pending",
        "verify_token": token,
        "otp_code":     None,
        "otp_expires_at": None,
        "created_at":   now,
        "updated_at":   now,
    }
    result = await db["team_access"].insert_one(doc)
    doc["_id"] = result.inserted_id

    owner = await db["users"].find_one({"_id": owner_id}, {"company_name": 1})
    owner_name   = (owner or {}).get("company_name", "Your team")
    verify_link  = f"{settings.FRONTEND_URL}/verify-team-access?token={token}"

    try:
        await send_team_access_email(
            to_email=email,
            member_name=name,
            owner_name=owner_name,
            verify_link=verify_link,
        )
    except Exception as exc:
        logger.warning("team_access.add.email_failed email=%s error=%s", email, exc)

    return {"ok": True, "member": _serialize(doc)}


# ── Public: verify token from email link ──────────────────────────────────────

async def verify_team_member(db: AsyncIOMotorDatabase, token: str) -> dict:
    member = await db["team_access"].find_one({"verify_token": token})
    if not member:
        return {"error": "invalid_token", "detail": "Invalid or expired verification link."}
    if member["status"] == "revoked":
        return {"error": "revoked", "detail": "This access has been revoked."}
    if member["status"] == "active":
        return {"ok": True, "already_verified": True, "email": member["email"]}

    await db["team_access"].update_one(
        {"_id": member["_id"]},
        {"$set": {"status": "active", "verify_token": None, "updated_at": _now()}},
    )
    return {"ok": True, "already_verified": False, "email": member["email"]}


# ── Owner: list team members ──────────────────────────────────────────────────

async def list_team_members(db: AsyncIOMotorDatabase, owner_id: ObjectId) -> list[dict]:
    cursor  = db["team_access"].find({"owner_id": owner_id}).sort("created_at", -1)
    members = [_serialize(m) async for m in cursor]
    return members


# ── Owner: toggle active ↔ inactive (block/unblock) ──────────────────────────

async def toggle_member_status(
    db: AsyncIOMotorDatabase,
    owner_id: ObjectId,
    member_id: str,
    new_status: str,
) -> dict:
    if not ObjectId.is_valid(member_id):
        return {"error": "not_found", "detail": "Member not found."}

    result = await db["team_access"].update_one(
        {"_id": ObjectId(member_id), "owner_id": owner_id, "status": {"$nin": ["revoked"]}},
        {"$set": {"status": new_status, "updated_at": _now()}},
    )
    if result.matched_count == 0:
        return {"error": "not_found", "detail": "Member not found or already revoked."}
    return {"ok": True}


# ── Owner: revoke access ──────────────────────────────────────────────────────

async def revoke_member(
    db: AsyncIOMotorDatabase,
    owner_id: ObjectId,
    member_id: str,
) -> dict:
    if not ObjectId.is_valid(member_id):
        return {"error": "not_found", "detail": "Member not found."}

    result = await db["team_access"].update_one(
        {"_id": ObjectId(member_id), "owner_id": owner_id},
        {"$set": {"status": "revoked", "otp_code": None, "updated_at": _now()}},
    )
    if result.matched_count == 0:
        return {"error": "not_found", "detail": "Member not found."}
    return {"ok": True}


# ── Auth: direct sign-in for a verified team member (no password) ────────────

async def signin_team_member(db: AsyncIOMotorDatabase, email: str) -> dict:
    """
    Called from auth_service.signin when the email is not in users.
    If the email belongs to an ACTIVE team member (email already verified),
    return the owner's full user document so the caller can issue a JWT.
    """
    member = await db["team_access"].find_one({"email": email, "status": "active"})
    if not member:
        # pending/inactive/revoked — not allowed
        member_any = await db["team_access"].find_one({"email": email})
        if member_any:
            status = member_any.get("status", "unknown")
            if status == "pending":
                return {"error": "team_member_not_verified"}
            if status in ("inactive", "revoked"):
                return {"error": "account_disabled"}
        return {"error": "not_team_member"}

    owner = await db["users"].find_one({"_id": ObjectId(str(member["owner_id"]))})
    if not owner:
        return {"error": "owner_not_found"}

    return {
        "ok":             True,
        "owner":          owner,
        "member_email":   email,
        "member_name":    member.get("name", ""),
        "team_access_id": str(member["_id"]),
    }


# ── Auth: request OTP for a team member ──────────────────────────────────────

async def request_team_otp(db: AsyncIOMotorDatabase, email: str) -> dict:
    """Called from auth_service when a sign-in email is not in the users collection."""
    member = await db["team_access"].find_one({"email": email, "status": "active"})
    if not member:
        return {"error": "not_team_member"}

    otp     = "".join(random.choices(string.digits, k=6))
    expires = _now() + timedelta(minutes=OTP_EXPIRE_MINUTES)

    await db["team_access"].update_one(
        {"_id": member["_id"]},
        {"$set": {"otp_code": otp, "otp_expires_at": expires, "updated_at": _now()}},
    )

    try:
        await send_team_access_otp_email(
            to_email=email,
            member_name=member.get("name", email),
            otp=otp,
        )
    except Exception as exc:
        logger.error("team_access.otp.email_failed email=%s error=%s", email, exc)
        return {"error": "email_send_failed"}

    return {"ok": True}


# ── Auth: verify OTP → return owner's JWT payload ────────────────────────────

async def verify_team_otp(db: AsyncIOMotorDatabase, email: str, otp_code: str) -> dict:
    """Returns the OWNER's user data so the team member gets the owner's session."""
    member = await db["team_access"].find_one({"email": email, "status": "active"})
    if not member:
        return {"error": "not_team_member"}

    if not member.get("otp_code"):
        return {"error": "otp_not_requested"}
    if member["otp_code"] != otp_code:
        return {"error": "invalid_otp"}

    expires = member.get("otp_expires_at")
    if expires:
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if _now() > expires:
            return {"error": "otp_expired"}

    # Clear OTP — one-time use
    await db["team_access"].update_one(
        {"_id": member["_id"]},
        {"$set": {"otp_code": None, "otp_expires_at": None, "updated_at": _now()}},
    )

    # Fetch the owner's full user document
    owner = await db["users"].find_one({"_id": ObjectId(str(member["owner_id"]))})
    if not owner:
        return {"error": "owner_not_found"}

    return {
        "ok":             True,
        "owner":          owner,
        "member_email":   email,
        "member_name":    member.get("name", ""),
        "team_access_id": str(member["_id"]),
    }
