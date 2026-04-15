from __future__ import annotations

from typing import Any

from bson import ObjectId
from jose import JWTError, jwt
from fastapi import APIRouter, Depends, Header, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from config import settings
from database import get_database
from services import widget_settings as widget_service
from model.widget_settings import WidgetSettingsModel, WidgetSettingsResponse

router = APIRouter(prefix="/widget-settings", tags=["Widget Settings"])


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


@router.get("", response_model=WidgetSettingsResponse)
async def get_widget_settings(
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    user = await _get_current_user(db, authorization)
    doc = await widget_service.get_settings(db, str(user["_id"]))
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Widget settings not found")
    return doc


@router.put("", response_model=WidgetSettingsResponse)
async def update_widget_settings(
    body: WidgetSettingsModel,
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    user = await _get_current_user(db, authorization)
    return await widget_service.upsert_settings(db, str(user["_id"]), body)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_widget_settings(
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    user = await _get_current_user(db, authorization)
    await widget_service.delete_settings(db, str(user["_id"]))


@router.get("/public/{api_key}", response_model=WidgetSettingsResponse)
async def get_public_widget_settings(
    api_key: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Public endpoint used by the embedded widget script.
    Accepts api_key in the format 'org-<company_id>' and returns that company's widget settings.
    """
    if api_key.startswith("org-"):
        company_id = api_key[4:]
    else:
        company_id = api_key

    if not ObjectId.is_valid(company_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid API key format")

    doc = await widget_service.get_settings(db, company_id)
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Widget settings not found")
    return doc
