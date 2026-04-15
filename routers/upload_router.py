from __future__ import annotations

from typing import Any

from bson import ObjectId
from jose import JWTError, jwt
from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile, File, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from config import settings
from database import get_database
from services import upload_service

router = APIRouter(prefix="/upload", tags=["Upload"])


def _extract_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    if authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return authorization.strip()


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


@router.post("/widget/image")
async def upload_widget_image(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    await _get_current_user(db, authorization)
    try:
        url = await upload_service.upload_image(file)
        return {"url": url}
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/widget/video")
async def upload_widget_video(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    await _get_current_user(db, authorization)
    try:
        url = await upload_service.upload_video(file)
        return {"url": url}
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))
