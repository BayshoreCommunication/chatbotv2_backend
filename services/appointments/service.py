from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

from model.appointments import (
    AppointmentSettingsDoc,
    CalendlyEvent,
    CalendlySettings,
    CalendlySlot,
    CalendlyStats,
)


CALENDLY_BASE_URL = "https://api.calendly.com"
SETTINGS_COLLECTION = "appointment_settings"


class CalendlyAPIError(Exception):
    pass


def _serialize_settings(doc: dict[str, Any] | None) -> CalendlySettings:
    if not doc:
        return CalendlySettings()
    return CalendlySettings(
        calendly_url=doc.get("calendly_url", ""),
        calendly_access_token=doc.get("calendly_access_token", ""),
        event_type_uri=doc.get("event_type_uri", ""),
        auto_embed=bool(doc.get("auto_embed", True)),
    )


async def get_user_calendly_settings(db: AsyncIOMotorDatabase, user_id: str) -> CalendlySettings:
    doc = await db[SETTINGS_COLLECTION].find_one({"user_id": user_id})
    return _serialize_settings(doc)


async def save_user_calendly_settings(
    db: AsyncIOMotorDatabase, user_id: str, settings: CalendlySettings
) -> CalendlySettings:
    now = datetime.utcnow()
    doc = AppointmentSettingsDoc(
        user_id=user_id,
        calendly_url=settings.calendly_url,
        calendly_access_token=settings.calendly_access_token,
        event_type_uri=settings.event_type_uri,
        auto_embed=settings.auto_embed,
        updated_at=now,
    ).model_dump()

    await db[SETTINGS_COLLECTION].update_one(
        {"user_id": user_id},
        {
            "$set": {
                "calendly_url": doc["calendly_url"],
                "calendly_access_token": doc["calendly_access_token"],
                "event_type_uri": doc["event_type_uri"],
                "auto_embed": doc["auto_embed"],
                "updated_at": now,
            },
            "$setOnInsert": {
                "user_id": user_id,
                "created_at": now,
            },
        },
        upsert=True,
    )
    return settings


async def delete_user_calendly_settings(db: AsyncIOMotorDatabase, user_id: str) -> bool:
    result = await db[SETTINGS_COLLECTION].delete_one({"user_id": user_id})
    return result.deleted_count > 0


async def _calendly_get(access_token: str, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(f"{CALENDLY_BASE_URL}{path}", headers=headers, params=params)

    if resp.status_code >= 400:
        try:
            body = resp.json()
            detail = body.get("message") or body.get("title") or str(body)
        except Exception:
            detail = resp.text or "(no body)"
        logger.error(
            "calendly.api_error status=%d path=%s detail=%s",
            resp.status_code, path, detail,
        )
        raise CalendlyAPIError(f"Calendly API error: {resp.status_code} — {detail}")
    return resp.json()


async def _get_current_calendly_user(access_token: str) -> dict[str, Any]:
    data = await _calendly_get(access_token, "/users/me")
    resource = data.get("resource")
    if not isinstance(resource, dict):
        raise CalendlyAPIError("Invalid Calendly user response")
    return resource


async def test_calendly_connection(access_token: str) -> bool:
    try:
        user = await _get_current_calendly_user(access_token)
    except CalendlyAPIError:
        return False
    return bool(user.get("uri"))


async def get_calendly_events(access_token: str) -> list[CalendlyEvent]:
    user = await _get_current_calendly_user(access_token)
    user_uri = user.get("uri")
    if not user_uri:
        return []

    data = await _calendly_get(
        access_token,
        "/event_types",
        params={"user": user_uri, "count": 100, "sort": "name:asc"},
    )
    collection = data.get("collection", [])
    events: list[CalendlyEvent] = []
    for item in collection:
        if not isinstance(item, dict):
            continue
        events.append(
            CalendlyEvent(
                uri=str(item.get("uri", "")),
                name=str(item.get("name", "")),
                duration=int(item.get("duration", 0) or 0),
                status="active" if bool(item.get("active", True)) else "inactive",
                booking_url=str(item.get("scheduling_url", "") or ""),
            )
        )
    return events


async def get_calendly_stats(access_token: str) -> CalendlyStats:
    events = await get_calendly_events(access_token)
    active_events = sum(1 for event in events if event.status == "active")

    upcoming_bookings = 0
    try:
        start_time = datetime.now(timezone.utc).isoformat()
        scheduled = await _calendly_get(
            access_token,
            "/scheduled_events",
            params={"status": "active", "min_start_time": start_time, "count": 100},
        )
        collection = scheduled.get("collection", [])
        if isinstance(collection, list):
            upcoming_bookings = len(collection)
    except CalendlyAPIError:
        upcoming_bookings = 0

    return CalendlyStats(
        total_events=len(events),
        active_events=active_events,
        upcoming_bookings=upcoming_bookings,
    )


async def get_calendly_availability(access_token: str, event_type_uri: str) -> list[CalendlySlot]:
    if not event_type_uri:
        return []

    now = datetime.now(timezone.utc)
    # Start 1 hour ahead — Calendly rejects start_time within the event's
    # minimum scheduling notice window, causing a 400 when passing exact now.
    range_start = now + timedelta(hours=1)
    start_time = range_start.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    end_time = (now + timedelta(days=7)).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    data = await _calendly_get(
        access_token,
        "/event_type_available_times",
        params={
            "event_type": event_type_uri,
            "start_time": start_time,
            "end_time": end_time,
        },
    )

    collection = data.get("collection", [])
    slots: list[CalendlySlot] = []
    for item in collection:
        if not isinstance(item, dict):
            continue
        start = str(item.get("start_time", "") or "")
        scheduling_url = str(item.get("scheduling_url", "") or "")
        if not start:
            continue
        slots.append(CalendlySlot(start_time=start, scheduling_url=scheduling_url))
    return slots
