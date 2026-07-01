from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from jose import JWTError, jwt
from fastapi import APIRouter, Depends, Header, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, EmailStr

from config import settings
from database import get_database
from services.user_service import hash_password
from services.auth_service import create_access_token
from utils.email import send_invite_email

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/invites", tags=["Invites"])

MAX_MEMBERS = 5


# ── Schemas ───────────────────────────────────────────────────────────────────

class InviteRequest(BaseModel):
    name: str
    email: EmailStr


class AcceptRequest(BaseModel):
    token: str


# ── Auth helper (same pattern as user_profile_router) ─────────────────────────

def _extract_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    if authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return authorization.strip()


async def _get_current_user(
    db: AsyncIOMotorDatabase, authorization: str | None
) -> dict[str, Any]:
    token = _extract_token(authorization)
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not ObjectId.is_valid(user_id):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject")

    user = await db["users"].find_one({"_id": ObjectId(user_id), "is_active": True})
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")
    return user


# ── Serializer ────────────────────────────────────────────────────────────────

def _serialize(invite: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(invite["_id"]),
        "owner_id": str(invite["owner_id"]),
        "invitee_name": invite.get("invitee_name", ""),
        "invitee_email": invite.get("invitee_email", ""),
        "token": invite.get("token", ""),
        "status": invite.get("status", "pending"),
        "created_at": invite.get("created_at", datetime.now(timezone.utc)).isoformat(),
    }


# ── POST /api/v1/invites ──────────────────────────────────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED)
async def send_invite(
    payload: InviteRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    owner = await _get_current_user(db, authorization)
    owner_id = owner["_id"]

    # Enforce member limit (count non-revoked invites for this owner)
    active_count = await db["invites"].count_documents(
        {"owner_id": owner_id, "status": {"$ne": "revoked"}}
    )
    if active_count >= MAX_MEMBERS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Member limit of {MAX_MEMBERS} reached. Revoke an existing member first.",
        )

    # Prevent duplicate pending/active invite to same email
    existing = await db["invites"].find_one(
        {"owner_id": owner_id, "invitee_email": payload.email, "status": {"$ne": "revoked"}}
    )
    if existing:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="An active or pending invite already exists for this email.",
        )

    token = secrets.token_urlsafe(32)
    invite_doc = {
        "owner_id": owner_id,
        "invitee_name": payload.name,
        "invitee_email": payload.email,
        "token": token,
        "status": "pending",
        "created_at": datetime.now(timezone.utc),
    }
    result = await db["invites"].insert_one(invite_doc)
    invite_doc["_id"] = result.inserted_id

    # Send invite email (non-blocking on failure — log but don't crash)
    invite_link = f"{settings.FRONTEND_URL}/accept-invite?token={token}"
    owner_name = owner.get("company_name", "A company")
    try:
        await send_invite_email(
            to_email=payload.email,
            invitee_name=payload.name,
            owner_name=owner_name,
            invite_link=invite_link,
        )
    except Exception as exc:
        logger.warning("Invite email failed: %s", exc)

    return _serialize(invite_doc)


# ── GET /api/v1/invites ───────────────────────────────────────────────────────

@router.get("")
async def list_invites(
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    owner = await _get_current_user(db, authorization)
    cursor = db["invites"].find({"owner_id": owner["_id"]}).sort("created_at", -1)
    invites = [_serialize(inv) async for inv in cursor]
    return invites


# ── DELETE /api/v1/invites/{invite_id} ───────────────────────────────────────

@router.delete("/{invite_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_invite(
    invite_id: str,
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    owner = await _get_current_user(db, authorization)

    if not ObjectId.is_valid(invite_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Invite not found.")

    result = await db["invites"].update_one(
        {"_id": ObjectId(invite_id), "owner_id": owner["_id"], "status": {"$ne": "revoked"}},
        {"$set": {"status": "revoked"}},
    )
    if result.matched_count == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Invite not found or already revoked.")


# ── POST /api/v1/invites/accept ───────────────────────────────────────────────

@router.post("/accept")
async def accept_invite(
    payload: AcceptRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    invite = await db["invites"].find_one({"token": payload.token})
    if not invite:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Invite link is invalid.")
    if invite["status"] == "revoked":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="This invite has been revoked by the owner.")

    invitee_email: str = invite["invitee_email"]
    invitee_name: str = invite.get("invitee_name", "")
    owner_id = invite["owner_id"]

    # Find or create the invitee's user account
    existing_user = await db["users"].find_one({"email": invitee_email})

    if existing_user:
        user_id = str(existing_user["_id"])
        company_name = existing_user.get("company_name", invitee_name)
        role = existing_user.get("role", "organization")
        # Link to owner if not already linked
        if not existing_user.get("owner_id"):
            await db["users"].update_one(
                {"_id": existing_user["_id"]},
                {"$set": {"owner_id": str(owner_id), "updated_at": datetime.now(timezone.utc)}},
            )
    else:
        # Create a new account for the invitee
        now = datetime.now(timezone.utc)
        new_user: dict[str, Any] = {
            "company_name": invitee_name or invitee_email.split("@")[0],
            "company_type": "other",
            "company_website": None,
            "phone_number": None,
            "email": invitee_email,
            "hashed_password": hash_password(secrets.token_urlsafe(16)),
            "role": "organization",
            "owner_id": str(owner_id),
            "is_active": True,
            "is_verified": True,
            "is_subscribed": False,
            "has_paid_subscription": False,
            "subscription_type": "free",
            "subscription_start_date": None,
            "subscription_end_date": None,
            "vector_store_id": None,
            "train_data": {
                "is_trained": False,
                "score": 0.0,
                "last_updated": None,
                "update_count": 0,
                "update_limit": 10,
                "entries_stored": 0,
                "pages_crawled": 0,
                "categories": [],
                "namespace": None,
                "history": [],
            },
            "created_at": now,
            "updated_at": now,
        }
        result = await db["users"].insert_one(new_user)
        user_id = str(result.inserted_id)
        company_name = new_user["company_name"]
        role = new_user["role"]

    # Mark the invite as active
    await db["invites"].update_one(
        {"_id": invite["_id"]}, {"$set": {"status": "active"}}
    )

    # Issue a JWT so the invitee is immediately signed in
    access_token = create_access_token({
        "sub": user_id,
        "email": invitee_email,
        "role": role,
    })

    return {
        "success": True,
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user_id,
        "email": invitee_email,
        "company_name": company_name,
        "role": role,
    }
