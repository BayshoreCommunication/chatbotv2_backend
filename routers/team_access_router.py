from __future__ import annotations

from typing import Any

from bson import ObjectId
from jose import JWTError, jwt
from fastapi import APIRouter, Depends, Header, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from config import settings
from database import get_database
from model.team_access_model import (
    AddMemberRequest,
    TeamAccessResponse,
    ToggleStatusRequest,
    VerifyTokenRequest,
)
from services import team_access_service

router = APIRouter(prefix="/team-access", tags=["Team Access"])


# ── Auth helper ───────────────────────────────────────────────────────────────

def _extract_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    return authorization.split(" ", 1)[-1].strip()


async def _get_owner(
    db: AsyncIOMotorDatabase,
    authorization: str | None,
) -> dict[str, Any]:
    token = _extract_token(authorization)
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")

    user_id = payload.get("sub")
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")

    user = await db["users"].find_one({"_id": ObjectId(user_id), "is_active": True})
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    return user


def _raise(result: dict) -> None:
    err = result.get("error")
    if not err:
        return
    detail = result.get("detail", err)
    if err in ("not_found", "invalid_token"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail)
    if err in ("already_exists", "limit_reached"):
        raise HTTPException(status.HTTP_409_CONFLICT, detail)
    raise HTTPException(status.HTTP_400_BAD_REQUEST, detail)


# ── POST /api/v1/team-access  — owner adds a member ──────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED, response_model=TeamAccessResponse)
async def add_member(
    payload: AddMemberRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    owner  = await _get_owner(db, authorization)
    result = await team_access_service.add_team_member(
        db, owner["_id"], payload.name, payload.email
    )
    _raise(result)
    return result["member"]


# ── GET /api/v1/team-access  — owner lists members ───────────────────────────

@router.get("", response_model=list[TeamAccessResponse])
async def list_members(
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    owner   = await _get_owner(db, authorization)
    members = await team_access_service.list_team_members(db, owner["_id"])
    return members


# ── POST /api/v1/team-access/verify  — member clicks email link (no auth) ────

@router.post("/verify")
async def verify_member(
    payload: VerifyTokenRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    result = await team_access_service.verify_team_member(db, payload.token)
    _raise(result)
    return result


# ── PATCH /api/v1/team-access/{member_id}/toggle  — block / unblock ──────────

@router.patch("/{member_id}/toggle")
async def toggle_member(
    member_id: str,
    payload: ToggleStatusRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    owner  = await _get_owner(db, authorization)
    result = await team_access_service.toggle_member_status(
        db, owner["_id"], member_id, payload.status
    )
    _raise(result)
    return {"ok": True}


# ── DELETE /api/v1/team-access/{member_id}  — revoke access ──────────────────

@router.delete("/{member_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_member(
    member_id: str,
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    owner  = await _get_owner(db, authorization)
    result = await team_access_service.revoke_member(db, owner["_id"], member_id)
    _raise(result)
