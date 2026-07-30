"""
services/apps_integration/service.py
──────────────────────────────────────
Business logic for third-party channel integrations — WhatsApp and Messenger
so far, Instagram to follow as a sibling function group in this same file,
per the shared "channel adapter" plan in
backend/docs/social-channels-integration-plan.md — no change to the AI/
knowledge-base logic here, just webhook-in / Graph-API-out plumbing that
bridges to the existing chat() endpoint.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from motor.motor_asyncio import AsyncIOMotorDatabase

from config import settings
from model.apps_integration import MessengerSettings, WhatsAppSettings

logger = logging.getLogger(__name__)

WHATSAPP_SETTINGS_COLLECTION = "whatsapp_settings"
MESSENGER_SETTINGS_COLLECTION = "messenger_settings"
# Records message_ids we've already handled — Meta retries webhook delivery
# on slow/non-2xx responses, so without this a slow AI reply can cause the
# same customer message to be answered twice.
WHATSAPP_PROCESSED_MESSAGES_COLLECTION = "whatsapp_processed_messages"
MESSENGER_PROCESSED_MESSAGES_COLLECTION = "messenger_processed_messages"
GRAPH_API_BASE_URL = "https://graph.facebook.com/v21.0"


class ChannelAPIError(Exception):
    """Base for a channel's Graph API / chat-bridge failures."""


class WhatsAppAPIError(ChannelAPIError):
    pass


class MessengerAPIError(ChannelAPIError):
    pass


# ── Settings CRUD (per company — each customer connects their own number) ───

def _serialize_settings(doc: dict[str, Any] | None) -> WhatsAppSettings:
    if not doc:
        return WhatsAppSettings()
    return WhatsAppSettings(
        phone_number_id=doc.get("phone_number_id", ""),
        access_token=doc.get("access_token", ""),
        business_account_id=doc.get("business_account_id", ""),
        verified_name=doc.get("verified_name", ""),
        display_phone_number=doc.get("display_phone_number", ""),
    )


async def get_company_whatsapp_settings(
    db: AsyncIOMotorDatabase, company_id: str
) -> WhatsAppSettings:
    doc = await db[WHATSAPP_SETTINGS_COLLECTION].find_one({"company_id": company_id})
    return _serialize_settings(doc)


async def save_company_whatsapp_settings(
    db: AsyncIOMotorDatabase,
    company_id: str,
    phone_number_id: str,
    access_token: str,
    business_account_id: str,
    verified_name: str,
    display_phone_number: str = "",
) -> WhatsAppSettings:
    now = datetime.now(timezone.utc)
    await db[WHATSAPP_SETTINGS_COLLECTION].update_one(
        {"company_id": company_id},
        {
            "$set": {
                "phone_number_id": phone_number_id,
                "access_token": access_token,
                "business_account_id": business_account_id,
                "verified_name": verified_name,
                "display_phone_number": display_phone_number,
                "updated_at": now,
            },
            "$setOnInsert": {
                "company_id": company_id,
                "created_at": now,
            },
        },
        upsert=True,
    )
    return WhatsAppSettings(
        phone_number_id=phone_number_id,
        access_token=access_token,
        business_account_id=business_account_id,
        verified_name=verified_name,
        display_phone_number=display_phone_number,
    )


async def delete_company_whatsapp_settings(db: AsyncIOMotorDatabase, company_id: str) -> bool:
    result = await db[WHATSAPP_SETTINGS_COLLECTION].delete_one({"company_id": company_id})
    return result.deleted_count > 0


async def find_company_id_by_phone_number_id(
    db: AsyncIOMotorDatabase, phone_number_id: str
) -> str | None:
    """Reverse lookup used by the webhook — Meta's payload only carries its
    own phone_number_id, so this is how an incoming message finds *whose*
    knowledge base to use. Unlike Calendly's per-company webhook URL, Meta
    webhooks are registered once per App and shared across every connected
    number, so this lookup replaces the URL-based routing Calendly gets for
    free."""
    doc = await db[WHATSAPP_SETTINGS_COLLECTION].find_one(
        {"phone_number_id": phone_number_id}, {"company_id": 1}
    )
    return doc.get("company_id") if doc else None


# ── Graph API calls ──────────────────────────────────────────────────────────

async def test_whatsapp_connection(
    phone_number_id: str, access_token: str
) -> tuple[bool, str, str]:
    """Verifies the token/number pair by fetching the number's verified
    business name and display phone number. Returns
    (valid, verified_name, display_phone_number)."""
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                f"{GRAPH_API_BASE_URL}/{phone_number_id}",
                headers=headers,
                params={"fields": "verified_name,display_phone_number"},
            )
    except httpx.HTTPError as exc:
        logger.error("whatsapp.test_connection.network_error error=%s", exc)
        return False, "", ""

    if resp.status_code >= 400:
        logger.warning(
            "whatsapp.test_connection.rejected status=%d body=%s", resp.status_code, resp.text,
        )
        return False, "", ""

    data = resp.json()
    return (
        True,
        str(data.get("verified_name", "")),
        str(data.get("display_phone_number", "")),
    )


# ── Embedded Signup / Login for Business ("Continue with Facebook") ────────
# The frontend's FB.login() only ever hands us a short-lived authorization
# `code` — never a token. These two helpers are the one place a raw access
# token gets minted, exactly mirroring Meta's documented exchange: code ->
# short-lived user token -> long-lived user token (~60 days). Shared by every
# channel that authenticates through Facebook Login for Business (WhatsApp's
# Embedded Signup and Messenger both use it) — each channel wraps
# MetaOAuthExchangeError in its own typed error so the router's except clause
# stays channel-specific.

class MetaOAuthExchangeError(Exception):
    pass


class WhatsAppEmbeddedSignupError(Exception):
    pass


class MessengerEmbeddedSignupError(Exception):
    pass


async def _exchange_code_for_short_lived_user_token(code: str) -> str:
    params = {
        "client_id": settings.META_APP_ID,
        "client_secret": settings.META_APP_SECRET,
        "code": code,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(f"{GRAPH_API_BASE_URL}/oauth/access_token", params=params)

    if resp.status_code >= 400:
        logger.error(
            "meta.oauth.code_exchange_failed status=%d body=%s",
            resp.status_code, resp.text,
        )
        raise MetaOAuthExchangeError("Failed to exchange authorization code")

    token = resp.json().get("access_token")
    if not token:
        raise MetaOAuthExchangeError("No access_token in Meta's response")
    return str(token)


async def _exchange_for_long_lived_user_token(short_lived_token: str) -> str:
    params = {
        "grant_type": "fb_exchange_token",
        "client_id": settings.META_APP_ID,
        "client_secret": settings.META_APP_SECRET,
        "fb_exchange_token": short_lived_token,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(f"{GRAPH_API_BASE_URL}/oauth/access_token", params=params)

    if resp.status_code >= 400:
        logger.error(
            "meta.oauth.long_lived_exchange_failed status=%d body=%s",
            resp.status_code, resp.text,
        )
        raise MetaOAuthExchangeError("Failed to exchange for a long-lived token")

    token = resp.json().get("access_token")
    if not token:
        raise MetaOAuthExchangeError("No access_token in Meta's response")
    return str(token)


async def connect_whatsapp_embedded(
    db: AsyncIOMotorDatabase,
    company_id: str,
    code: str,
    phone_number_id: str,
    business_account_id: str,
) -> WhatsAppSettings:
    """Full Embedded Signup pipeline: code -> long-lived token -> verify ->
    save. Raises WhatsAppEmbeddedSignupError on any step failing, so the
    router can turn it into a clean 400 for the dashboard to show."""
    try:
        short_lived = await _exchange_code_for_short_lived_user_token(code)
        access_token = await _exchange_for_long_lived_user_token(short_lived)
    except MetaOAuthExchangeError as exc:
        raise WhatsAppEmbeddedSignupError(str(exc)) from exc

    valid, verified_name, display_phone_number = await test_whatsapp_connection(
        phone_number_id, access_token
    )
    if not valid:
        raise WhatsAppEmbeddedSignupError(
            "Connected to Facebook, but couldn't verify the WhatsApp number"
        )

    return await save_company_whatsapp_settings(
        db,
        company_id,
        phone_number_id=phone_number_id,
        access_token=access_token,
        business_account_id=business_account_id,
        verified_name=verified_name,
        display_phone_number=display_phone_number,
    )


async def send_whatsapp_message(
    phone_number_id: str, access_token: str, to: str, text: str
) -> None:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    body = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"{GRAPH_API_BASE_URL}/{phone_number_id}/messages",
            headers=headers,
            json=body,
        )
    if resp.status_code >= 400:
        logger.error(
            "whatsapp.send_message.failed status=%d body=%s", resp.status_code, resp.text,
        )
        raise WhatsAppAPIError(f"WhatsApp send failed: {resp.status_code} — {resp.text}")


# ── Webhook signature verification ───────────────────────────────────────────
# One shared App Secret for all companies (unlike Calendly's per-subscription
# signing key) — Meta signs every webhook POST with this App's secret
# regardless of which connected number/page the message is for.

def verify_meta_webhook_signature(
    app_secret: str, raw_body: bytes, signature_header: str | None
) -> bool:
    """Verifies Meta's `X-Hub-Signature-256: sha256=<hex hmac>` header —
    HMAC-SHA256 of the raw request body using the Meta App Secret."""
    if not app_secret or not signature_header:
        return False
    if not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    provided = signature_header.removeprefix("sha256=")
    return hmac.compare_digest(expected, provided)


# ── Incoming message dedup ───────────────────────────────────────────────────
# Shared by every channel — each keeps its own collection since message_ids
# are only unique per-platform, not globally.

async def is_duplicate_message(
    db: AsyncIOMotorDatabase, collection: str, message_id: str
) -> bool:
    """First call for a given message_id returns False and records it;
    every subsequent call (Meta's webhook retries) returns True."""
    result = await db[collection].update_one(
        {"_id": message_id},
        {"$setOnInsert": {"processed_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
    # upserted_id is only set when this call actually created the doc — i.e.
    # this is the first time this message_id has been seen.
    return result.upserted_id is None


# ── Bridge to the existing chat brain ────────────────────────────────────────
# No AI/knowledge-base logic lives here — this calls the same POST
# /chat/{company_id} endpoint the website widget already uses, so every
# channel gets identical answers without duplicating any agent/session logic.

async def get_ai_reply(channel: str, company_id: str, session_id: str, message: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{settings.INTERNAL_API_BASE_URL}/api/v1/chat/{company_id}",
            json={"session_id": session_id, "message": message},
        )
    if resp.status_code >= 400:
        logger.error(
            "%s.chat_bridge.failed company_id=%s status=%d body=%s",
            channel, company_id, resp.status_code, resp.text,
        )
        raise ChannelAPIError(f"Chat bridge failed: {resp.status_code}")
    return resp.json().get("reply", "")


async def handle_incoming_whatsapp_message(
    db: AsyncIOMotorDatabase,
    phone_number_id: str,
    from_number: str,
    message_id: str,
    message_text: str,
) -> None:
    """Full pipeline for one incoming WhatsApp message: dedupe, find the
    owning company, ask the AI, send the reply back. Never raises — a
    forged/unmatched/errored message should be dropped, not surfaced as a
    500 (Meta retries aggressively on non-2xx responses)."""
    try:
        if await is_duplicate_message(db, WHATSAPP_PROCESSED_MESSAGES_COLLECTION, message_id):
            logger.info("whatsapp.message.duplicate_skipped message_id=%s", message_id)
            return

        company_id = await find_company_id_by_phone_number_id(db, phone_number_id)
        if not company_id:
            logger.warning(
                "whatsapp.message.unmatched_phone_number_id phone_number_id=%s", phone_number_id,
            )
            return

        company_settings = await get_company_whatsapp_settings(db, company_id)
        if not company_settings.access_token:
            logger.warning("whatsapp.message.no_access_token company_id=%s", company_id)
            return

        # WhatsApp has no separate "session" concept — the sender's phone
        # number itself is the session, so the same conversation always
        # resumes with the same history.
        reply = await get_ai_reply(
            "whatsapp", company_id, session_id=from_number, message=message_text
        )
        if not reply:
            return

        await send_whatsapp_message(
            phone_number_id, company_settings.access_token, from_number, reply,
        )
    except Exception:
        logger.exception(
            "whatsapp.message.handler_error phone_number_id=%s message_id=%s",
            phone_number_id, message_id,
        )


# ═══ Messenger ════════════════════════════════════════════════════════════════

# ── Settings CRUD (per company — each customer connects their own Page) ────

def _serialize_messenger_settings(doc: dict[str, Any] | None) -> MessengerSettings:
    if not doc:
        return MessengerSettings()
    return MessengerSettings(
        page_id=doc.get("page_id", ""),
        access_token=doc.get("access_token", ""),
        page_name=doc.get("page_name", ""),
    )


async def get_company_messenger_settings(
    db: AsyncIOMotorDatabase, company_id: str
) -> MessengerSettings:
    doc = await db[MESSENGER_SETTINGS_COLLECTION].find_one({"company_id": company_id})
    return _serialize_messenger_settings(doc)


async def save_company_messenger_settings(
    db: AsyncIOMotorDatabase,
    company_id: str,
    page_id: str,
    access_token: str,
    page_name: str = "",
) -> MessengerSettings:
    now = datetime.now(timezone.utc)
    await db[MESSENGER_SETTINGS_COLLECTION].update_one(
        {"company_id": company_id},
        {
            "$set": {
                "page_id": page_id,
                "access_token": access_token,
                "page_name": page_name,
                "updated_at": now,
            },
            "$setOnInsert": {
                "company_id": company_id,
                "created_at": now,
            },
        },
        upsert=True,
    )
    return MessengerSettings(page_id=page_id, access_token=access_token, page_name=page_name)


async def delete_company_messenger_settings(db: AsyncIOMotorDatabase, company_id: str) -> bool:
    result = await db[MESSENGER_SETTINGS_COLLECTION].delete_one({"company_id": company_id})
    return result.deleted_count > 0


async def find_company_id_by_page_id(db: AsyncIOMotorDatabase, page_id: str) -> str | None:
    """Reverse lookup used by the webhook — same shape as
    find_company_id_by_phone_number_id: Meta's Messenger webhook is
    registered once per App and shared across every connected Page, so an
    incoming message's Page ID is the only way to find whose knowledge base
    to use."""
    doc = await db[MESSENGER_SETTINGS_COLLECTION].find_one(
        {"page_id": page_id}, {"company_id": 1}
    )
    return doc.get("company_id") if doc else None


# ── Graph API calls ──────────────────────────────────────────────────────────

async def test_messenger_connection(page_id: str, access_token: str) -> tuple[bool, str]:
    """Verifies the Page ID / Page Access Token pair by fetching the Page's
    name. Returns (valid, page_name)."""
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                f"{GRAPH_API_BASE_URL}/{page_id}",
                headers=headers,
                params={"fields": "name"},
            )
    except httpx.HTTPError as exc:
        logger.error("messenger.test_connection.network_error error=%s", exc)
        return False, ""

    if resp.status_code >= 400:
        logger.warning(
            "messenger.test_connection.rejected status=%d body=%s", resp.status_code, resp.text,
        )
        return False, ""

    return True, str(resp.json().get("name", ""))


# ── Facebook Login for Business ("Continue with Facebook") ─────────────────
# Unlike WhatsApp's Embedded Signup, Meta doesn't post the selected Page back
# to the frontend through window.postMessage here — FB.login() only ever
# hands us a `code`. The backend exchanges that for a long-lived *user*
# token (via the shared helpers above), then reads the available Pages off
# GET /me/accounts — each entry already carries its own (long-lived) Page
# Access Token. `page_id` is optional: most businesses have exactly one Page
# connected, so we auto-select it; it's only required to disambiguate when
# the login grants access to more than one.

async def _resolve_messenger_page(
    user_access_token: str, page_id: str | None
) -> tuple[str, str, str]:
    """Returns (resolved_page_id, page_access_token, page_name)."""
    headers = {"Authorization": f"Bearer {user_access_token}"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{GRAPH_API_BASE_URL}/me/accounts",
            headers=headers,
            params={"fields": "id,name,access_token", "limit": 200},
        )

    if resp.status_code >= 400:
        logger.error(
            "messenger.login.accounts_lookup_failed status=%d body=%s",
            resp.status_code, resp.text,
        )
        raise MessengerEmbeddedSignupError("Failed to look up this user's Facebook Pages")

    pages = resp.json().get("data", [])
    if not pages:
        raise MessengerEmbeddedSignupError(
            "No Facebook Pages found for this login — connect a Page in Business Manager first."
        )

    if page_id:
        match = next((p for p in pages if str(p.get("id")) == page_id), None)
        if not match:
            raise MessengerEmbeddedSignupError(
                "That Page wasn't found among the Pages this Facebook login manages."
            )
    elif len(pages) == 1:
        match = pages[0]
    else:
        names = ", ".join(str(p.get("name") or p.get("id")) for p in pages)
        raise MessengerEmbeddedSignupError(
            f"This Facebook login manages multiple Pages ({names}) — "
            "reconnect and select just one to continue."
        )

    token = match.get("access_token")
    if not token:
        raise MessengerEmbeddedSignupError("Facebook didn't return an access token for this Page.")
    return str(match["id"]), str(token), str(match.get("name", ""))


async def connect_messenger_embedded(
    db: AsyncIOMotorDatabase,
    company_id: str,
    code: str,
    page_id: str | None = None,
) -> MessengerSettings:
    """Full Login for Business pipeline: code -> long-lived user token ->
    resolve Page -> save. Raises MessengerEmbeddedSignupError on any step
    failing, so the router can turn it into a clean 400 for the dashboard
    to show. No separate verify call needed — a Page Access Token read
    straight off GET /me/accounts is already proven live."""
    try:
        short_lived = await _exchange_code_for_short_lived_user_token(code)
        user_access_token = await _exchange_for_long_lived_user_token(short_lived)
    except MetaOAuthExchangeError as exc:
        raise MessengerEmbeddedSignupError(str(exc)) from exc

    resolved_page_id, page_access_token, page_name = await _resolve_messenger_page(
        user_access_token, page_id
    )

    return await save_company_messenger_settings(
        db,
        company_id,
        page_id=resolved_page_id,
        access_token=page_access_token,
        page_name=page_name,
    )


async def send_messenger_message(access_token: str, to: str, text: str) -> None:
    """No page_id in the URL — unlike WhatsApp's per-number endpoint,
    Messenger's Send API resolves the sending Page from the Page Access
    Token itself, always via the shared `me/messages` path."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    body = {
        "recipient": {"id": to},
        "messaging_type": "RESPONSE",
        "message": {"text": text},
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"{GRAPH_API_BASE_URL}/me/messages",
            headers=headers,
            json=body,
        )
    if resp.status_code >= 400:
        logger.error(
            "messenger.send_message.failed status=%d body=%s", resp.status_code, resp.text,
        )
        raise MessengerAPIError(f"Messenger send failed: {resp.status_code} — {resp.text}")


async def handle_incoming_messenger_message(
    db: AsyncIOMotorDatabase,
    page_id: str,
    sender_id: str,
    message_id: str,
    message_text: str,
) -> None:
    """Full pipeline for one incoming Messenger message: dedupe, find the
    owning company, ask the AI, send the reply back. Never raises — same
    contract as handle_incoming_whatsapp_message (Meta retries aggressively
    on non-2xx webhook responses)."""
    try:
        if await is_duplicate_message(db, MESSENGER_PROCESSED_MESSAGES_COLLECTION, message_id):
            logger.info("messenger.message.duplicate_skipped message_id=%s", message_id)
            return

        company_id = await find_company_id_by_page_id(db, page_id)
        if not company_id:
            logger.warning("messenger.message.unmatched_page_id page_id=%s", page_id)
            return

        company_settings = await get_company_messenger_settings(db, company_id)
        if not company_settings.access_token:
            logger.warning("messenger.message.no_access_token company_id=%s", company_id)
            return

        # Messenger has no separate "session" concept — the sender's PSID
        # (page-scoped user id) itself is the session, so the same
        # conversation always resumes with the same history.
        reply = await get_ai_reply(
            "messenger", company_id, session_id=sender_id, message=message_text
        )
        if not reply:
            return

        await send_messenger_message(company_settings.access_token, sender_id, reply)
    except Exception:
        logger.exception(
            "messenger.message.handler_error page_id=%s message_id=%s",
            page_id, message_id,
        )
