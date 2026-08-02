"""
routers/meta_webhook_router.py
─────────────────────────────────
Single shared Meta webhook endpoint — POST /api/webhooks/meta — covering
every channel connected through the multi-connection `channel_connections`
model (Messenger + Instagram today, see services/apps_integration/
channel_connections_service.py). Distinct from the legacy per-channel
webhook at /apps-integration/messenger/webhook (apps_integration_router.py),
which still serves the older single-connection Messenger flow — the two
aren't both registered against the same Meta App subscription at once in
practice, but both exist in code during the migration.

Endpoints:
  GET  /api/webhooks/meta — Meta's verification handshake
  POST /api/webhooks/meta — incoming Messenger/Instagram messages
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from fastapi import Depends

from config import settings
from database import get_database
from model.apps_integration_model import ChannelType
from services.apps_integration import (
    handle_incoming_channel_message,
    verify_meta_webhook_signature,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["Meta Webhooks"])


@router.get("/meta", include_in_schema=False)
async def verify_meta_webhook(
    hub_mode: str = Query(default="", alias="hub.mode"),
    hub_verify_token: str = Query(default="", alias="hub.verify_token"),
    hub_challenge: str = Query(default="", alias="hub.challenge"),
):
    if hub_mode == "subscribe" and hub_verify_token == settings.META_WEBHOOK_VERIFY_TOKEN:
        return int(hub_challenge) if hub_challenge.isdigit() else hub_challenge
    logger.warning("meta_webhook.verify_failed hub_mode=%s", hub_mode)
    raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Verification failed")


@router.post("/meta", include_in_schema=False)
async def receive_meta_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not verify_meta_webhook_signature(settings.META_APP_SECRET, raw_body, signature):
        logger.warning("meta_webhook.invalid_signature")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid signature")

    body = await request.json()
    object_type = body.get("object")
    if object_type == "page":
        channel = ChannelType.messenger
    elif object_type == "instagram":
        channel = ChannelType.instagram
    else:
        logger.info("meta_webhook.unhandled_object object=%s", object_type)
        return {"received": True}

    for entry in body.get("entry", []):
        external_id = str(entry.get("id") or "")
        for event in entry.get("messaging", []):
            message = event.get("message") or {}
            # Skip echoes of messages we sent ourselves and non-text events
            # (postbacks, reactions, read receipts) — text-only for now.
            if message.get("is_echo") or "text" not in message:
                continue
            sender_id = str((event.get("sender") or {}).get("id") or "")
            message_id = str(message.get("mid") or "")
            message_text = str(message.get("text") or "")
            if not (external_id and sender_id and message_id and message_text):
                continue

            # Queued to run after this handler returns — Meta expects a fast
            # 200, and get_ai_reply (which this eventually calls) can be
            # slow. No external queue in this codebase yet, so this uses
            # FastAPI/Starlette's in-process BackgroundTasks: good enough to
            # not block the webhook response, but it won't survive a
            # process restart mid-flight — a durable queue (e.g. Celery/RQ)
            # would be the next step if that ever matters at higher volume.
            background_tasks.add_task(
                handle_incoming_channel_message,
                db, channel, external_id, sender_id, message_id, message_text,
            )

    # Always 200 — Meta retries aggressively on non-2xx, and a bug on our
    # side shouldn't turn into an infinite retry storm.
    return {"received": True}
