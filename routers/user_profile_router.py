from __future__ import annotations

from typing import Any

from bson import ObjectId
from jose import JWTError, jwt
from fastapi import APIRouter, Depends, Header, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel

from config import settings
from database import get_database


router = APIRouter(prefix="/user", tags=["User Profile"])


class UserProfileUpdateRequest(BaseModel):
    companyName: str | None = None
    companyType: str | None = None
    website: str | None = None
    avatar: str | None = None


def _extract_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    if authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return authorization.strip()


def _serialize_profile(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(user["_id"]),
        "name": user.get("company_name", ""),
        "companyName": user.get("company_name", ""),
        "companyType": user.get("company_type", "other"),
        "website": user.get("company_website"),
        "avatar": user.get("avatar"),
        "email": user.get("email", ""),
        "role": user.get("role", "organization"),
    }


async def _get_current_user(
    db: AsyncIOMotorDatabase,
    authorization: str | None,
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


@router.get("")
async def get_current_user_profile(
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    user = await _get_current_user(db, authorization)
    return {"payload": {"user": _serialize_profile(user)}}


@router.put("")
async def update_current_user_profile(
    payload: UserProfileUpdateRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    user = await _get_current_user(db, authorization)

    update_fields: dict[str, Any] = {}
    if payload.companyName is not None:
        update_fields["company_name"] = payload.companyName
    if payload.companyType is not None:
        update_fields["company_type"] = payload.companyType
    if payload.website is not None:
        update_fields["company_website"] = payload.website
    if payload.avatar is not None:
        update_fields["avatar"] = payload.avatar

    if update_fields:
        await db["users"].update_one({"_id": user["_id"]}, {"$set": update_fields})
        user = await db["users"].find_one({"_id": user["_id"]}) or user

    return {"payload": {"user": _serialize_profile(user)}}

