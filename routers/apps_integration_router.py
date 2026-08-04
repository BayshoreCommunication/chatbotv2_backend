"""
routers/apps_integration_router.py
─────────────────────────────────────
Dashboard "Apps & Integrations" page. Two coexisting Messenger connect flows
today, plus the multi-connection OAuth flow that also covers Instagram:

  1. Legacy single-connection Messenger flow (`/messenger/*`) — JS-SDK popup
     or manual Page ID + token, one Page per company, stored in
     `messenger_settings`. Still live; not being removed by this change.
  2. New multi-connection flow (`/oauth/*`) — server-redirect OAuth, lets a
     company connect more than one Page (and any linked Instagram account),
     stored in `channel_connections`. Its webhook is
     routers/meta_webhook_router.py (POST /api/webhooks/meta), not the
     `/messenger/webhook` below.

Endpoints:
  GET    /apps-integration/messenger/snapshot         — current settings + connection status
  POST   /apps-integration/messenger/connect          — manual: save + verify a company's Page token
  POST   /apps-integration/messenger/connect-embedded — "Continue with Facebook": exchange a
                                                         Login for Business code for a Page token, then save
  DELETE /apps-integration/messenger/settings         — disconnect
  GET    /apps-integration/messenger/webhook          — Meta's verification handshake
  POST   /apps-integration/messenger/webhook          — incoming Messenger messages

  GET    /apps-integration/oauth/initiate                    — returns the Facebook OAuth dialog URL
  GET    /apps-integration/oauth/callback                    — Meta redirects here (public, no auth header)
  GET    /apps-integration/oauth/pending/{selection_id}       — candidate Pages to show in a picker
  POST   /apps-integration/oauth/confirm                     — save + subscribe the chosen Page(s)
  GET    /apps-integration/oauth/connections                 — list this company's channel_connections
  DELETE /apps-integration/oauth/connections/{channel}/{external_id} — disconnect one

  POST   /apps-integration/whatsapp/connect-embedded         — WhatsApp Embedded Signup: exchange the
                                                                 code + phone_number_id + waba_id Meta hands
                                                                 back, subscribe, save. Listing/disconnect
                                                                 reuse the /oauth/connections endpoints above.
"""

from __future__ import annotations

import logging
from typing import Any

from bson import ObjectId
from config import settings
from database import get_database
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorDatabase
from services.apps_integration import (
    ChannelOAuthError,
    MessengerEmbeddedSignupError,
    build_authorize_url,
    confirm_channel_connections,
    connect_messenger_embedded,
    connect_whatsapp_embedded,
    decode_oauth_state,
    delete_company_messenger_settings,
    disconnect_channel_connection,
    exchange_code_and_list_pages,
    get_company_messenger_settings,
    get_pending_selection,
    handle_incoming_messenger_message,
    list_company_channel_connections,
    save_company_messenger_settings,
    test_messenger_connection,
    verify_meta_webhook_signature,
)

from model.apps_integration_model import (
    ChannelConnectionsListResponse,
    ChannelConnectionSummary,
    ChannelType,
    MessengerConnectRequest,
    MessengerEmbeddedConnectRequest,
    MessengerSettingsResponse,
    OAuthConfirmRequest,
    OAuthInitiateResponse,
    OAuthPendingSelectionResponse,
    WhatsAppEmbeddedConnectRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/apps-integration", tags=["Apps Integration"])


def _verify_meta_challenge(
    hub_mode: str, hub_verify_token: str, hub_challenge: str
) -> int | str:
    """Shared GET-webhook handshake — Messenger, Instagram, and any future
    channel all verify the same way: echo hub.challenge back once
    hub.verify_token matches our stored token."""
    if (
        hub_mode == "subscribe"
        and hub_verify_token == settings.META_WEBHOOK_VERIFY_TOKEN
    ):
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
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
    except JWTError:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )

    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not ObjectId.is_valid(user_id):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject"
        )

    user = await db["users"].find_one({"_id": ObjectId(user_id), "is_active": True})
    if not user:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive"
        )

    return {"id": str(user["_id"])}


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
    valid, page_name = await test_messenger_connection(
        payload.page_id, payload.access_token
    )
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
# Registered once at the Meta App level and shared across every company's
# connected Page — Meta has no concept of "per customer" webhook URLs, so
# incoming messages are routed to a company by looking up the payload's
# Page id (see find_company_id_by_page_id).


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
            # (postbacks, reactions, read receipts) — text-only for now.
            if message.get("is_echo") or "text" not in message:
                continue
            sender_id = str((event.get("sender") or {}).get("id") or "")
            message_id = str(message.get("mid") or "")
            message_text = str(message.get("text") or "")
            if not (page_id and sender_id and message_id and message_text):
                continue

            await handle_incoming_messenger_message(
                db,
                page_id,
                sender_id,
                message_id,
                message_text,
            )

    # Always 200 — Meta retries aggressively on non-2xx, and a bug on our
    # side shouldn't turn into an infinite retry storm.
    return {"received": True}


# ── OAuth: multi-connection Messenger + Instagram (channel_connections) ────
# Server-redirect flow, distinct from the JS-SDK popup Messenger uses above.
# initiate (authenticated) -> Facebook's OAuth dialog -> callback (public,
# Meta redirects the raw browser here) -> pending selection stashed
# server-side -> confirm (authenticated) picks which Page(s) to connect.

@router.get("/oauth/initiate", response_model=OAuthInitiateResponse)
async def initiate_channel_oauth(
    current_user: dict[str, Any] = Depends(get_current_user),
):
    return OAuthInitiateResponse(authorize_url=build_authorize_url(current_user["id"]))


@router.get("/oauth/callback", include_in_schema=False)
async def channel_oauth_callback(
    code: str = Query(default=""),
    state: str = Query(default=""),
    error: str = Query(default=""),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Meta redirects the browser here directly — no Authorization header,
    hence identifying the company via the signed `state` param instead of
    get_current_user. Always redirects onward to the dashboard; never
    returns raw JSON, since this is a full-page browser navigation."""
    dashboard_url = f"{settings.FRONTEND_URL}/apps-integration"

    if error or not code or not state:
        logger.warning("channel_oauth.callback_denied error=%s", error or "missing code/state")
        return RedirectResponse(f"{dashboard_url}?oauth_error=denied")

    try:
        company_id = decode_oauth_state(state)
        selection_id = await exchange_code_and_list_pages(db, company_id, code)
    except ChannelOAuthError as exc:
        logger.warning("channel_oauth.callback_failed error=%s", exc)
        return RedirectResponse(f"{dashboard_url}?oauth_error=failed")

    return RedirectResponse(f"{dashboard_url}?selection_id={selection_id}")


@router.get("/oauth/pending/{selection_id}", response_model=OAuthPendingSelectionResponse)
async def get_channel_oauth_pending_selection(
    selection_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    try:
        pages = await get_pending_selection(db, current_user["id"], selection_id)
    except ChannelOAuthError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return OAuthPendingSelectionResponse(selection_id=selection_id, pages=pages)


@router.post("/oauth/confirm", response_model=ChannelConnectionsListResponse)
async def confirm_channel_oauth_selection(
    payload: OAuthConfirmRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    try:
        saved = await confirm_channel_connections(
            db, current_user["id"], payload.selection_id, payload.external_ids,
        )
    except ChannelOAuthError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return ChannelConnectionsListResponse(
        connections=[
            ChannelConnectionSummary(
                id=c.id or "", channel=c.channel, external_id=c.external_id,
                page_name=c.page_name, status=c.status, connected_at=c.connected_at,
            )
            for c in saved
        ]
    )


@router.get("/oauth/connections", response_model=ChannelConnectionsListResponse)
async def list_channel_connections(
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    connections = await list_company_channel_connections(db, current_user["id"])
    return ChannelConnectionsListResponse(
        connections=[
            ChannelConnectionSummary(
                id=c.id or "", channel=c.channel, external_id=c.external_id,
                page_name=c.page_name, status=c.status, connected_at=c.connected_at,
            )
            for c in connections
        ]
    )


@router.delete("/oauth/connections/{channel}/{external_id}")
async def disconnect_channel_oauth_connection(
    channel: ChannelType,
    external_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    found = await disconnect_channel_connection(db, current_user["id"], channel, external_id)
    if not found:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Connection not found")
    return {"ok": True}


# ── WhatsApp: Embedded Signup ────────────────────────────────────────────────
# One-click connect — Meta's own signup wizard handles WABA/phone number
# creation and verification, the frontend just relays the resulting code +
# phone_number_id + waba_id here. Listing/disconnect reuse the generic
# /oauth/connections endpoints above (ChannelType.whatsapp included for free).


@router.post("/whatsapp/connect-embedded", response_model=ChannelConnectionSummary)
async def connect_whatsapp_via_embedded_signup(
    payload: WhatsAppEmbeddedConnectRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    try:
        connection = await connect_whatsapp_embedded(
            db,
            current_user["id"],
            code=payload.code,
            phone_number_id=payload.phone_number_id,
            waba_id=payload.waba_id,
        )
    except ChannelOAuthError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return ChannelConnectionSummary(
        id=connection.id or "", channel=connection.channel, external_id=connection.external_id,
        page_name=connection.page_name, status=connection.status, connected_at=connection.connected_at,
    )
