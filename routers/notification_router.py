from typing import List, Any
from fastapi import APIRouter, Depends, HTTPException, Header, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from jose import JWTError, jwt
from bson import ObjectId

from database import get_database
from config import settings
from services import notification_service
from schemas.notification_schema import NotificationResponse

router = APIRouter(prefix="/notifications", tags=["Notifications"])


async def get_current_user(
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

    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not ObjectId.is_valid(user_id):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject")

    user = await db["users"].find_one({"_id": ObjectId(user_id), "is_active": True})
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    return {"id": str(user["_id"])}


@router.get("/", response_model=List[NotificationResponse])
async def list_notifications(
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Recent notifications for the current user's company, newest first."""
    return await notification_service.get_notifications(db, current_user["id"])


@router.get("/unread-count")
async def unread_count(
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Lightweight endpoint for the topbar bell badge — poll this frequently."""
    count = await notification_service.get_unread_count(db, current_user["id"])
    return {"count": count}


@router.post("/read-all")
async def read_all(
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Mark every notification as read — call this when the bell dropdown is opened."""
    modified = await notification_service.mark_all_read(db, current_user["id"])
    return {"modified": modified}


@router.post("/{notification_id}/read", response_model=NotificationResponse)
async def read_one(
    notification_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    updated = await notification_service.mark_one_read(db, current_user["id"], notification_id)
    if not updated:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Notification not found")
    return updated
