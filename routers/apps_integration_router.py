"""
routers/apps_integration_router.py
─────────────────────────────────────
Dashboard "Apps & Integrations" page — WhatsApp and Messenger so far,
Instagram to follow as a sibling route group in this same router (see
backend/docs/social-channels-integration-plan.md).

Endpoints:
  GET    /apps-integration/whatsapp/snapshot         — current settings + connection status
  POST   /apps-integration/whatsapp/connect          — manual: save + verify a company's WhatsApp credentials
  POST   /apps-integration/whatsapp/connect-embedded — "Continue with Facebook": exchange an
                                                        Embedded Signup code for a token, then save
  DELETE /apps-integration/whatsapp/settings         — disconnect
  GET    /apps-integration/whatsapp/webhook          — Meta's verification handshake
  POST   /apps-integration/whatsapp/webhook          — incoming WhatsApp messages

  GET    /apps-integration/messenger/snapshot         — current settings + connection status
  POST   /apps-integration/messenger/connect          — manual: save + verify a company's Page token
  POST   /apps-integration/messenger/connect-embedded — "Continue with Facebook": exchange a
                                                         Login for Business code for a Page token, then save
  DELETE /apps-integration/messenger/settings         — disconnect
  GET    /apps-integration/messenger/webhook          — Meta's verification handshake
  POST   /apps-integration/messenger/webhook          — incoming Messenger messages
"""

from __future__ import annotations

import logging
from typing import Any

from bson import ObjectId
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorDatabase

from config import settings
from database import get_database
from model.apps_integration import (
    MessengerConnectRequest,
    MessengerEmbeddedConnectRequest,
    MessengerSettingsResponse,
    WhatsAppConnectRequest,
    WhatsAppEmbeddedConnectRequest,
    WhatsAppSettingsResponse,
)
from services.apps_integration import (
    connect_messenger_embedded,
    connect_whatsapp_embedded,
    delete_company_messenger_settings,
    delete_company_whatsapp_settings,
    find_company_id_by_page_id,
    find_company_id_by_phone_number_id,
    get_company_messenger_settings,
    get_company_whatsapp_settings,
    handle_incoming_messenger_message,
    handle_incoming_whatsapp_message,
    save_company_messenger_settings,
    save_company_whatsapp_settings,
    test_messenger_connection,
    test_whatsapp_connection,
    verify_meta_webhook_signature,
    MessengerEmbeddedSignupError,
    WhatsAppEmbeddedSignupError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/apps-integration", tags=["Apps Integration"])


def _verify_meta_challenge(hub_mode: str, hub_verify_token: str, hub_challenge: str) -> int | str:
    """Shared GET-webhook handshake — Messenger, Instagram, and WhatsApp all
    verify the same way: echo hub.challenge back once hub.verify_token
    matches our stored token."""
    if hub_mode == "subscribe" and hub_verify_token == settings.META_WEBHOOK_VERIFY_TOKEN:
        return int(hub_challenge) if hub_challenge.isdigit() else hub_challenge
    raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Verification failed")


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


# ── WhatsApp: dashboard-facing settings ───────────────────────────────────────

@router.get("/whatsapp/snapshot", response_model=WhatsAppSettingsResponse)
async def get_whatsapp_snapshot(
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    company_settings = await get_company_whatsapp_settings(db, current_user["id"])
    connected = bool(company_settings.access_token and company_settings.phone_number_id)
    return WhatsAppSettingsResponse(settings=company_settings, connected=connected)


@router.post("/whatsapp/connect", response_model=WhatsAppSettingsResponse)
async def connect_whatsapp(
    payload: WhatsAppConnectRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    valid, verified_name, display_phone_number = await test_whatsapp_connection(
        payload.phone_number_id, payload.access_token,
    )
    if not valid:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Could not verify this Phone Number ID / access token with Meta.",
        )

    company_settings = await save_company_whatsapp_settings(
        db,
        current_user["id"],
        phone_number_id=payload.phone_number_id,
        access_token=payload.access_token,
        business_account_id=payload.business_account_id,
        verified_name=verified_name,
        display_phone_number=display_phone_number,
    )
    return WhatsAppSettingsResponse(settings=company_settings, connected=True)


@router.post("/whatsapp/connect-embedded", response_model=WhatsAppSettingsResponse)
async def connect_whatsapp_via_embedded_signup(
    payload: WhatsAppEmbeddedConnectRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    try:
        company_settings = await connect_whatsapp_embedded(
            db,
            current_user["id"],
            code=payload.code,
            phone_number_id=payload.phone_number_id,
            business_account_id=payload.business_account_id,
        )
    except WhatsAppEmbeddedSignupError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return WhatsAppSettingsResponse(settings=company_settings, connected=True)


@router.delete("/whatsapp/settings")
async def disconnect_whatsapp(
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    await delete_company_whatsapp_settings(db, current_user["id"])
    return {"ok": True}


# ── WhatsApp: Meta webhook (public — no dashboard auth) ──────────────────────
# Registered once at the Meta App level and shared across every company's
# connected number — Meta has no concept of "per customer" webhook URLs, so
# incoming messages are routed to a company by looking up the payload's
# phone_number_id (see find_company_id_by_phone_number_id).

@router.get("/whatsapp/webhook", include_in_schema=False)
async def verify_whatsapp_webhook(
    hub_mode: str = Query(default="", alias="hub.mode"),
    hub_verify_token: str = Query(default="", alias="hub.verify_token"),
    hub_challenge: str = Query(default="", alias="hub.challenge"),
):
    try:
        return _verify_meta_challenge(hub_mode, hub_verify_token, hub_challenge)
    except HTTPException:
        logger.warning("whatsapp.webhook.verify_failed hub_mode=%s", hub_mode)
        raise


@router.post("/whatsapp/webhook", include_in_schema=False)
async def receive_whatsapp_webhook(
    request: Request,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not verify_meta_webhook_signature(settings.META_APP_SECRET, raw_body, signature):
        logger.warning("whatsapp.webhook.invalid_signature")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid signature")

    body = await request.json()
    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            phone_number_id = str(
                (value.get("metadata") or {}).get("phone_number_id") or ""
            )
            for message in value.get("messages", []):
                # Only plain text handled for now — media/voice/stickers are
                # an explicit v2 scope, not an oversight (see doc's Phase 4).
                if message.get("type") != "text":
                    continue
                from_number = str(message.get("from") or "")
                message_id = str(message.get("id") or "")
                message_text = str((message.get("text") or {}).get("body") or "")
                if not (phone_number_id and from_number and message_id and message_text):
                    continue

                await handle_incoming_whatsapp_message(
                    db, phone_number_id, from_number, message_id, message_text,
                )

    # Always 200 — Meta retries aggressively on non-2xx, and a bug on our
    # side shouldn't turn into an infinite retry storm.
    return {"received": True}


# ── Messenger: dashboard-facing settings ──────────────────────────────────────

@router.get("/messenger/snapshot", response_model=MessengerSettingsResponse)
async def get_messenger_snapshot(
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    company_settings = await get_company_messenger_settings(db, current_user["id"])
    connected = bool(company_settings.access_token and company_settings.page_id)
    return MessengerSettingsResponse(settings=company_settings, connected=connected)


@router.post("/messenger/connect", response_model=MessengerSettingsResponse)
async def connect_messenger(
    payload: MessengerConnectRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    valid, page_name = await test_messenger_connection(payload.page_id, payload.access_token)
    if not valid:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Could not verify this Page ID / access token with Meta.",
        )

    company_settings = await save_company_messenger_settings(
        db,
        current_user["id"],
        page_id=payload.page_id,
        access_token=payload.access_token,
        page_name=page_name,
    )
    return MessengerSettingsResponse(settings=company_settings, connected=True)


@router.post("/messenger/connect-embedded", response_model=MessengerSettingsResponse)
async def connect_messenger_via_login_for_business(
    payload: MessengerEmbeddedConnectRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    try:
        company_settings = await connect_messenger_embedded(
            db,
            current_user["id"],
            code=payload.code,
            page_id=payload.page_id,
        )
    except MessengerEmbeddedSignupError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return MessengerSettingsResponse(settings=company_settings, connected=True)


@router.delete("/messenger/settings")
async def disconnect_messenger(
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    await delete_company_messenger_settings(db, current_user["id"])
    return {"ok": True}


# ── Messenger: Meta webhook (public — no dashboard auth) ─────────────────────
# Same shared-webhook shape as WhatsApp: one callback URL per App, incoming
# messages routed to a company by looking up the payload's Page id (see
# find_company_id_by_page_id).

@router.get("/messenger/webhook", include_in_schema=False)
async def verify_messenger_webhook(
    hub_mode: str = Query(default="", alias="hub.mode"),
    hub_verify_token: str = Query(default="", alias="hub.verify_token"),
    hub_challenge: str = Query(default="", alias="hub.challenge"),
):
    try:
        return _verify_meta_challenge(hub_mode, hub_verify_token, hub_challenge)
    except HTTPException:
        logger.warning("messenger.webhook.verify_failed hub_mode=%s", hub_mode)
        raise


@router.post("/messenger/webhook", include_in_schema=False)
async def receive_messenger_webhook(
    request: Request,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not verify_meta_webhook_signature(settings.META_APP_SECRET, raw_body, signature):
        logger.warning("messenger.webhook.invalid_signature")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid signature")

    body = await request.json()
    for entry in body.get("entry", []):
        page_id = str(entry.get("id") or "")
        for event in entry.get("messaging", []):
            message = event.get("message") or {}
            # Skip echoes of messages we sent ourselves and non-text events
            # (postbacks, reactions, read receipts) — same "text only for
            # now" scope as WhatsApp (see doc's Phase 3).
            if message.get("is_echo") or "text" not in message:
                continue
            sender_id = str((event.get("sender") or {}).get("id") or "")
            message_id = str(message.get("mid") or "")
            message_text = str(message.get("text") or "")
            if not (page_id and sender_id and message_id and message_text):
                continue

            await handle_incoming_messenger_message(
                db, page_id, sender_id, message_id, message_text,
            )

    # Always 200 — Meta retries aggressively on non-2xx, and a bug on our
    # side shouldn't turn into an infinite retry storm.
    return {"received": True}
