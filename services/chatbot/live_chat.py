"""
services/chatbot/live_chat.py
─────────────────────────────
Live-chat plumbing shared by every channel — the website widget
(routers/chat_router.py's widget_ask) and Messenger / Instagram / WhatsApp
(services/apps_integration/channel_connections_service.py):

  - human takeover: whether the business owner has paused the AI for a
    conversation, and storing a visitor message the AI must not answer
  - which channel a conversation came in on, so an owner reply from the
    dashboard can be sent back through the same channel
  - the dashboard websocket events the Chats page renders in real time
"""

from __future__ import annotations

from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from services.chatbot.ws_manager import ws_manager

CHAT_SESSIONS_COLLECTION = "chat_sessions"


async def is_human_takeover(db: AsyncIOMotorDatabase, company_id: str, session_id: str) -> bool:
    doc = await db[CHAT_SESSIONS_COLLECTION].find_one(
        {"company_id": company_id, "session_id": session_id},
        {"human_takeover": 1},
    )
    return bool(doc and doc.get("human_takeover"))


async def remember_channel(
    db: AsyncIOMotorDatabase,
    company_id: str,
    session_id: str,
    channel: str,
    channel_external_id: str,
) -> None:
    """Records which channel (and which Page / Instagram account / WhatsApp
    number) a conversation belongs to, so owner replies can be routed back
    to it. Creates the session if this is the visitor's first message —
    with the same insert-only fields chat_router._persist_exchange sets."""
    now = datetime.now(timezone.utc)
    await db[CHAT_SESSIONS_COLLECTION].update_one(
        {"company_id": company_id, "session_id": session_id},
        {
            "$set": {"channel": channel, "channel_external_id": channel_external_id},
            "$setOnInsert": {
                "company_id": company_id,
                "session_id": session_id,
                "created_at": now,
                "updated_at": now,
                "last_intent": None,
            },
        },
        upsert=True,
    )


async def save_visitor_message(
    db: AsyncIOMotorDatabase, company_id: str, session_id: str, content: str,
) -> datetime:
    """Stores a visitor message without asking the AI — used while a human
    has taken over the conversation. Returns the stored timestamp."""
    now = datetime.now(timezone.utc)
    await db[CHAT_SESSIONS_COLLECTION].update_one(
        {"company_id": company_id, "session_id": session_id},
        {
            "$push": {"messages": {"role": "user", "content": content, "timestamp": now}},
            "$set": {"updated_at": now},
        },
    )
    return now


# ── Dashboard websocket events (rendered by the dashboard's Chats page) ─────

async def notify_visitor_message(
    company_id: str,
    session_id: str,
    content: str,
    channel: str,
    timestamp: datetime | None = None,
) -> None:
    await ws_manager.notify_dashboard(company_id, {
        "type": "new_message",
        "session_id": session_id,
        "channel": channel,
        "content": content,
        "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
    })


async def notify_ai_typing(company_id: str, session_id: str) -> None:
    await ws_manager.notify_dashboard(company_id, {
        "type": "typing_start",
        "session_id": session_id,
    })


async def notify_ai_reply(company_id: str, session_id: str, content: str, channel: str) -> None:
    await ws_manager.notify_dashboard(company_id, {
        "type": "ai_reply",
        "session_id": session_id,
        "channel": channel,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
