from __future__ import annotations

from typing import Any

from bson import ObjectId
from jose import JWTError, jwt
from fastapi import APIRouter, Depends, HTTPException, Query, status, Header
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field

from config import settings
from database import get_database
from model.appointments import (
    CalendlyAvailabilityResponse,
    CalendlyEventsResponse,
    CalendlySettings,
    CalendlySettingsResponse,
    CalendlyStatsResponse,
    CalendlyTestConnectionRequest,
    CalendlyTestConnectionResponse,
)
from services.appointments import (
    delete_user_calendly_settings,
    get_calendly_availability,
    get_calendly_events,
    get_calendly_stats,
    get_user_calendly_settings,
    save_user_calendly_settings,
    test_calendly_connection,
)
from services.appointments.service import CalendlyAPIError


router = APIRouter(prefix="/appointments", tags=["Appointments"])


class CalendlyConnectRequest(BaseModel):
    access_token: str = Field(..., min_length=1)


class CalendlyEventTypeRequest(BaseModel):
    event_type_uri: str = ""


def _empty_stats() -> dict[str, int]:
    return {"total_events": 0, "active_events": 0, "upcoming_bookings": 0}


async def _build_snapshot(
    db: AsyncIOMotorDatabase,
    user_id: str,
) -> dict[str, Any]:
    settings_doc = await get_user_calendly_settings(db, user_id)
    token = settings_doc.calendly_access_token.strip()

    if not token:
        return {
            "settings": settings_doc.model_dump(),
            "token_configured": False,
            "connected": False,
            "events": [],
            "stats": _empty_stats(),
            "slots": [],
            "error": None,
        }

    try:
        valid = await test_calendly_connection(token)
        if not valid:
            return {
                "settings": settings_doc.model_dump(),
                "token_configured": True,
                "connected": False,
                "events": [],
                "stats": _empty_stats(),
                "slots": [],
                "error": "Saved token is invalid. Please update your token.",
            }

        events = await get_calendly_events(token)
        stats_data = await get_calendly_stats(token)
        slots = []
        if settings_doc.event_type_uri:
            slots = await get_calendly_availability(token, settings_doc.event_type_uri)

        return {
            "settings": settings_doc.model_dump(),
            "token_configured": True,
            "connected": True,
            "events": [event.model_dump() for event in events],
            "stats": stats_data.model_dump(),
            "slots": [slot.model_dump() for slot in slots],
            "error": None,
        }
    except CalendlyAPIError as exc:
        return {
            "settings": settings_doc.model_dump(),
            "token_configured": True,
            "connected": False,
            "events": [],
            "stats": _empty_stats(),
            "slots": [],
            "error": str(exc),
        }


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

    return {
        "id": str(user["_id"]),
        "email": user.get("email", ""),
        "company_name": user.get("company_name", ""),
    }


@router.get("/calendly/snapshot")
async def get_snapshot(
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    return await _build_snapshot(db, current_user["id"])


@router.post("/calendly/connect")
async def connect_token(
    payload: CalendlyConnectRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    valid = await test_calendly_connection(payload.access_token)
    if not valid:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid Calendly access token")

    current = await get_user_calendly_settings(db, current_user["id"])
    next_settings = CalendlySettings(
        calendly_url=current.calendly_url,
        calendly_access_token=payload.access_token,
        event_type_uri=current.event_type_uri,
        auto_embed=current.auto_embed,
    )
    await save_user_calendly_settings(db, current_user["id"], next_settings)
    return await _build_snapshot(db, current_user["id"])


@router.delete("/calendly/token")
async def delete_token(
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    await delete_user_calendly_settings(db, current_user["id"])
    return await _build_snapshot(db, current_user["id"])


@router.patch("/calendly/event-type")
async def update_event_type(
    payload: CalendlyEventTypeRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    current = await get_user_calendly_settings(db, current_user["id"])
    if not current.calendly_access_token:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Calendly token is not configured")

    next_settings = CalendlySettings(
        calendly_url=current.calendly_url,
        calendly_access_token=current.calendly_access_token,
        event_type_uri=payload.event_type_uri,
        auto_embed=current.auto_embed,
    )
    await save_user_calendly_settings(db, current_user["id"], next_settings)
    return await _build_snapshot(db, current_user["id"])


@router.get("/calendly/settings", response_model=CalendlySettingsResponse)
async def get_settings(
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    settings_doc = await get_user_calendly_settings(db, current_user["id"])
    return CalendlySettingsResponse(settings=settings_doc)


@router.post("/calendly/settings")
async def save_settings(
    payload: CalendlySettings,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    if not payload.calendly_access_token.strip():
        await delete_user_calendly_settings(db, current_user["id"])
        return {"ok": True}
    await save_user_calendly_settings(db, current_user["id"], payload)
    return {"ok": True}


@router.post("/calendly/test-connection", response_model=CalendlyTestConnectionResponse)
async def test_connection(
    payload: CalendlyTestConnectionRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    _ = current_user
    valid = await test_calendly_connection(payload.access_token)
    return CalendlyTestConnectionResponse(valid=valid)


@router.get("/calendly/events", response_model=CalendlyEventsResponse)
async def list_events(
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    settings_doc = await get_user_calendly_settings(db, current_user["id"])
    if not settings_doc.calendly_access_token:
        return CalendlyEventsResponse(events=[])

    try:
        events = await get_calendly_events(settings_doc.calendly_access_token)
    except CalendlyAPIError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return CalendlyEventsResponse(events=events)


@router.get("/calendly/stats", response_model=CalendlyStatsResponse)
async def stats(
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    settings_doc = await get_user_calendly_settings(db, current_user["id"])
    if not settings_doc.calendly_access_token:
        return CalendlyStatsResponse(stats={"total_events": 0, "active_events": 0, "upcoming_bookings": 0})

    try:
        stats_data = await get_calendly_stats(settings_doc.calendly_access_token)
    except CalendlyAPIError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return CalendlyStatsResponse(stats=stats_data)


@router.get("/calendly/availability", response_model=CalendlyAvailabilityResponse)
async def availability(
    event_type_uri: str = Query(default=""),
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    settings_doc = await get_user_calendly_settings(db, current_user["id"])
    if not settings_doc.calendly_access_token:
        return CalendlyAvailabilityResponse(slots=[])

    if not event_type_uri:
        event_type_uri = settings_doc.event_type_uri
    if not event_type_uri:
        return CalendlyAvailabilityResponse(slots=[])

    try:
        slots = await get_calendly_availability(settings_doc.calendly_access_token, event_type_uri)
    except CalendlyAPIError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return CalendlyAvailabilityResponse(slots=slots)
