from __future__ import annotations

import hashlib
import hmac
import logging
import time
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
# Imported lazily inside record_appointment_from_webhook, not at module load —
# services.lead_service pulls in services.chatbot (via .llm), which imports
# services.chatbot.tools, which imports THIS module: a circular import at
# top-level. Deferring to call-time (long after both modules have finished
# loading) breaks the cycle.


CALENDLY_BASE_URL = "https://api.calendly.com"
SETTINGS_COLLECTION = "appointment_settings"
# Tolerance for webhook signature timestamp — rejects replayed payloads older
# than this, per Calendly's documented verification recipe.
WEBHOOK_SIGNATURE_TOLERANCE_SECONDS = 5 * 60


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


async def get_webhook_signing_key(db: AsyncIOMotorDatabase, user_id: str) -> str | None:
    """Internal lookup for the webhook route — never exposed to the frontend."""
    doc = await db[SETTINGS_COLLECTION].find_one(
        {"user_id": user_id}, {"calendly_webhook_signing_key": 1}
    )
    key = (doc or {}).get("calendly_webhook_signing_key", "")
    return key or None


async def save_user_calendly_settings(
    db: AsyncIOMotorDatabase, user_id: str, settings: CalendlySettings
) -> CalendlySettings:
    now = datetime.now(timezone.utc)
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


# ── Webhook subscription (real bookings, not just offered slots) ──────────────

async def _calendly_post(access_token: str, path: str, json_body: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(f"{CALENDLY_BASE_URL}{path}", headers=headers, json=json_body)

    if resp.status_code >= 400:
        try:
            body = resp.json()
            detail = body.get("message") or body.get("title") or str(body)
        except Exception:
            detail = resp.text or "(no body)"
        logger.error("calendly.api_error status=%d path=%s detail=%s", resp.status_code, path, detail)
        raise CalendlyAPIError(f"Calendly API error: {resp.status_code} — {detail}")
    return resp.json()


async def ensure_calendly_webhook(
    db: AsyncIOMotorDatabase,
    user_id: str,
    access_token: str,
    callback_url: str,
) -> None:
    """
    Register a Calendly webhook subscription (invitee.created/canceled) for
    this company's account, pointing at our callback_url, and store the
    subscription's signing key so incoming payloads can be verified.

    Idempotent-ish: skips creation if a subscription for this exact callback
    URL already exists. Never raises — a company with scheduling otherwise
    working shouldn't be broken by a webhook registration hiccup; failures are
    logged and appointment auto-recording just won't kick in until retried
    (e.g. next time settings are saved).
    """
    if not callback_url:
        logger.warning(
            "calendly.webhook.skip user_id=%s reason=no_callback_url "
            "(set BACKEND_PUBLIC_URL to enable real-booking tracking)",
            user_id,
        )
        return

    try:
        me = await _get_current_calendly_user(access_token)
        org_uri = me.get("current_organization")
        user_uri = me.get("uri")
        if not org_uri or not user_uri:
            logger.warning("calendly.webhook.skip user_id=%s reason=missing_org_or_user_uri", user_id)
            return

        existing = await _calendly_get(
            access_token,
            "/webhook_subscriptions",
            params={"organization": org_uri, "user": user_uri, "scope": "user", "count": 100},
        )
        for item in existing.get("collection", []):
            if isinstance(item, dict) and item.get("callback_url") == callback_url:
                # Already registered for this exact URL — nothing to do. The
                # signing key from creation time is still stored from before.
                logger.info("calendly.webhook.already_registered user_id=%s uri=%s", user_id, item.get("uri"))
                return

        created = await _calendly_post(
            access_token,
            "/webhook_subscriptions",
            {
                "url": callback_url,
                "events": ["invitee.created", "invitee.canceled"],
                "organization": org_uri,
                "user": user_uri,
                "scope": "user",
            },
        )
        resource = created.get("resource", {})
        webhook_uri = str(resource.get("uri", ""))
        signing_key = str(resource.get("signing_key", ""))
        if not webhook_uri or not signing_key:
            logger.error("calendly.webhook.create_bad_response user_id=%s response=%s", user_id, created)
            return

        await db[SETTINGS_COLLECTION].update_one(
            {"user_id": user_id},
            {"$set": {
                "calendly_webhook_uri": webhook_uri,
                "calendly_webhook_signing_key": signing_key,
                "updated_at": datetime.now(timezone.utc),
            }},
        )
        logger.info("calendly.webhook.created user_id=%s uri=%s", user_id, webhook_uri)
    except CalendlyAPIError as exc:
        logger.error("calendly.webhook.setup_failed user_id=%s error=%s", user_id, exc)
    except Exception as exc:
        logger.exception("calendly.webhook.setup_unexpected_error user_id=%s error=%s", user_id, exc)


def verify_calendly_webhook_signature(
    signing_key: str,
    raw_body: bytes,
    signature_header: str | None,
) -> bool:
    """
    Verify Calendly's `Calendly-Webhook-Signature` header: `t=<unix ts>,v1=<hex hmac>`
    where the signed message is `f"{t}.{raw_body}"` HMAC-SHA256'd with the
    subscription's signing key. Rejects stale timestamps to block replay.
    """
    if not signing_key or not signature_header:
        return False

    parts: dict[str, str] = {}
    for chunk in signature_header.split(","):
        if "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        parts[key.strip()] = value.strip()

    ts = parts.get("t")
    v1 = parts.get("v1")
    if not ts or not v1:
        return False

    try:
        if abs(time.time() - int(ts)) > WEBHOOK_SIGNATURE_TOLERANCE_SECONDS:
            logger.warning("calendly.webhook.stale_signature ts=%s", ts)
            return False
    except ValueError:
        return False

    signed_payload = f"{ts}.".encode() + raw_body
    expected = hmac.new(signing_key.encode(), signed_payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, v1)


async def _get_scheduled_event_start_time(access_token: str, event_uri: str) -> datetime | None:
    """event_uri is the invitee payload's `event` field — a full Calendly API URL."""
    if not event_uri:
        return None
    path = event_uri.replace(CALENDLY_BASE_URL, "")
    try:
        data = await _calendly_get(access_token, path)
    except CalendlyAPIError:
        return None
    start = (data.get("resource") or {}).get("start_time")
    if not start:
        return None
    try:
        return datetime.fromisoformat(str(start).replace("Z", "+00:00"))
    except ValueError:
        return None


async def record_appointment_from_webhook(
    db: AsyncIOMotorDatabase,
    company_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """
    Handle a verified `invitee.created`/`invitee.canceled` webhook payload:
    match it to the lead that was offered this booking link and set (or clear)
    `appointment_time` on that lead — the source of truth the dashboard reads.
    """
    from services import lead_service  # deferred — see import note above

    tracking = payload.get("tracking") or {}
    session_id = str(tracking.get("utm_content") or "").strip() or None
    email = str(payload.get("email") or "").strip() or None

    if event_type == "invitee.canceled":
        await lead_service.set_lead_appointment_time(
            db, company_id, None, session_id=session_id, email=email,
        )
        return

    if event_type != "invitee.created":
        return

    settings_doc = await get_user_calendly_settings(db, company_id)
    start_time = await _get_scheduled_event_start_time(
        settings_doc.calendly_access_token, str(payload.get("event") or "")
    )
    if not start_time:
        logger.warning(
            "calendly.webhook.no_start_time company_id=%s session_id=%s", company_id, session_id,
        )
        return

    await lead_service.set_lead_appointment_time(
        db, company_id, start_time, session_id=session_id, email=email,
    )
