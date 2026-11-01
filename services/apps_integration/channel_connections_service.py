"""
services/apps_integration/channel_connections_service.py
────────────────────────────────────────────────────────
Business logic for the unified, multi-connection channel model
(`channel_connections`): server-redirect OAuth (initiate/callback/confirm),
connection listing/disconnect, the shared Send API call, and the incoming-
message pipeline used by routers/meta_webhook_router.py. Coexists with
apps_integration_service.py's older single-connection Messenger flow rather
than replacing it — see that file's docstring and
routers/apps_integration_router.py for what still uses which.

Security notes:
  - Every stored access token (including in the short-lived pending-
    selection stash) is encrypted with utils.token_encryption before it
    touches Mongo — never write or log a raw token.
  - The OAuth `state` param is a signed, short-lived JWT (same SECRET_KEY/
    ALGORITHM as user auth, but a distinct "purpose" claim and no "sub"),
    since Meta's callback redirect is a bare browser GET with no
    Authorization header to identify the logged-in company otherwise.
"""

from __future__ import annotations

import logging
import secrets as secrets_module
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import httpx
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorDatabase

from config import settings
from model.apps_integration_model import (
    ChannelConnectionDoc,
    ChannelType,
    ConnectionStatus,
    OAuthCandidatePage,
)
from services.apps_integration.apps_integration_service import (
    MetaOAuthExchangeError,
    _exchange_code_for_short_lived_user_token,
    _exchange_for_long_lived_user_token,
    get_ai_reply,
    is_duplicate_message,
)
from utils.token_encryption import TokenEncryptionError, decrypt_token, encrypt_token

logger = logging.getLogger(__name__)

CHANNEL_CONNECTIONS_COLLECTION = "channel_connections"
OAUTH_PENDING_SELECTIONS_COLLECTION = "oauth_pending_selections"
CHANNEL_PROCESSED_MESSAGES_COLLECTION = "channel_connections_processed_messages"

GRAPH_API_BASE_URL = f"https://graph.facebook.com/{settings.META_GRAPH_API_VERSION}"

_OAUTH_SCOPES = (
    "pages_show_list,pages_messaging,pages_manage_metadata,"
    "pages_read_engagement,instagram_basic,instagram_manage_messages"
)
_STATE_TTL_MINUTES = 10
_PENDING_SELECTION_TTL_MINUTES = 10


class ChannelOAuthError(Exception):
    """`code` is a short machine-readable reason the OAuth callback passes
    to the dashboard as ?oauth_error=<code>, so the user sees *why* the
    connect failed instead of one generic message."""

    def __init__(self, message: str, code: str = "failed") -> None:
        super().__init__(message)
        self.code = code


class MetaSendError(Exception):
    pass


class MetaTokenExpiredError(MetaSendError):
    """Raised when Meta rejects a Send API call for an auth/token reason —
    the caller should mark the connection token_expired, not just log it."""


# ── OAuth state (identifies which company an inbound callback belongs to) ──

def _mint_oauth_state(company_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=_STATE_TTL_MINUTES)
    payload = {"company_id": company_id, "purpose": "meta_oauth_state", "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_oauth_state(state: str) -> str:
    """Returns the company_id embedded in an initiate-minted state token.
    Raises ChannelOAuthError if missing, expired, forged, or not actually an
    OAuth-state token (defense against someone replaying an unrelated JWT)."""
    try:
        payload = jwt.decode(state, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError as exc:
        raise ChannelOAuthError("Invalid or expired OAuth state", code="invalid_state") from exc

    if payload.get("purpose") != "meta_oauth_state":
        raise ChannelOAuthError("Invalid OAuth state", code="invalid_state")

    company_id = payload.get("company_id")
    if not isinstance(company_id, str) or not company_id:
        raise ChannelOAuthError("Invalid OAuth state", code="invalid_state")
    return company_id


def build_authorize_url(company_id: str) -> str:
    """Step 1 (initiate): the URL the dashboard sends the browser to."""
    params = {
        "client_id": settings.META_APP_ID,
        "redirect_uri": settings.META_OAUTH_REDIRECT_URI,
        "scope": _OAUTH_SCOPES,
        "response_type": "code",
        "state": _mint_oauth_state(company_id),
    }
    return f"https://www.facebook.com/{settings.META_GRAPH_API_VERSION}/dialog/oauth?{urlencode(params)}"


# ── Callback: code -> user token -> candidate Pages (+ linked Instagram) ───

async def _list_oauth_candidate_pages(user_access_token: str) -> list[dict[str, Any]]:
    """GET /me/accounts for the Pages this login manages, then for each Page
    checks for a linked Instagram Business Account. Returns raw dicts (not
    yet encrypted/stashed) — each with a plaintext page access_token, so
    callers must hand this straight to create_pending_selection and not
    persist it themselves."""
    headers = {"Authorization": f"Bearer {user_access_token}"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{GRAPH_API_BASE_URL}/me/accounts",
            headers=headers,
            params={"fields": "id,name,access_token", "limit": 200},
        )
    if resp.status_code >= 400:
        logger.error(
            "channel_oauth.accounts_lookup_failed status=%d body=%s",
            resp.status_code, resp.text,
        )
        raise ChannelOAuthError("Failed to look up this user's Facebook Pages", code="pages_lookup_failed")

    pages = resp.json().get("data", [])
    candidates: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=20.0) as client:
        for page in pages:
            page_id = str(page.get("id") or "")
            page_token = page.get("access_token")
            if not page_id or not page_token:
                continue

            linked_instagram_id: str | None = None
            linked_instagram_name: str | None = None
            try:
                ig_resp = await client.get(
                    f"{GRAPH_API_BASE_URL}/{page_id}",
                    headers={"Authorization": f"Bearer {page_token}"},
                    params={"fields": "instagram_business_account"},
                )
                if ig_resp.status_code < 400:
                    ig_account = (ig_resp.json().get("instagram_business_account") or {}).get("id")
                    if ig_account:
                        linked_instagram_id = str(ig_account)
                        name_resp = await client.get(
                            f"{GRAPH_API_BASE_URL}/{linked_instagram_id}",
                            headers={"Authorization": f"Bearer {page_token}"},
                            params={"fields": "username"},
                        )
                        if name_resp.status_code < 400:
                            linked_instagram_name = str(name_resp.json().get("username", ""))
            except httpx.HTTPError as exc:
                # A failed Instagram lookup shouldn't block connecting the
                # Page itself — just means no linked IG shows up as an option.
                logger.warning(
                    "channel_oauth.instagram_lookup_failed page_id=%s error=%s", page_id, exc,
                )

            candidates.append({
                "id": page_id,
                "name": str(page.get("name", "")),
                "access_token": str(page_token),
                "linked_instagram_id": linked_instagram_id,
                "linked_instagram_name": linked_instagram_name,
            })

    return candidates


async def exchange_code_and_list_pages(db: AsyncIOMotorDatabase, company_id: str, code: str) -> str:
    """Full callback pipeline: code -> long-lived user token -> candidate
    Pages -> stashed (encrypted) pending selection. Returns a selection_id
    for the dashboard to fetch via get_pending_selection."""
    try:
        short_lived = await _exchange_code_for_short_lived_user_token(
            code, settings.META_OAUTH_REDIRECT_URI
        )
        user_access_token = await _exchange_for_long_lived_user_token(short_lived)
    except MetaOAuthExchangeError as exc:
        raise ChannelOAuthError(str(exc), code="token_exchange_failed") from exc

    candidates = await _list_oauth_candidate_pages(user_access_token)
    if not candidates:
        raise ChannelOAuthError(
            "No Facebook Pages found for this login — connect a Page in Business Manager first.",
            code="no_pages",
        )

    return await _create_pending_selection(db, company_id, candidates)


# ── Pending selection stash (short-lived, encrypted, single-use) ───────────

async def _create_pending_selection(
    db: AsyncIOMotorDatabase, company_id: str, candidates: list[dict[str, Any]]
) -> str:
    selection_id = secrets_module.token_urlsafe(24)
    now = datetime.now(timezone.utc)
    stashed = [
        {
            "id": c["id"],
            "name": c["name"],
            "encrypted_access_token": encrypt_token(c["access_token"]),
            "linked_instagram_id": c["linked_instagram_id"],
            "linked_instagram_name": c["linked_instagram_name"],
        }
        for c in candidates
    ]
    await db[OAUTH_PENDING_SELECTIONS_COLLECTION].insert_one({
        "_id": selection_id,
        "company_id": company_id,
        "candidates": stashed,
        "created_at": now,
        "expires_at": now + timedelta(minutes=_PENDING_SELECTION_TTL_MINUTES),
    })
    return selection_id


async def _get_pending_selection_doc(
    db: AsyncIOMotorDatabase, company_id: str, selection_id: str
) -> dict[str, Any]:
    doc = await db[OAUTH_PENDING_SELECTIONS_COLLECTION].find_one({"_id": selection_id})
    if not doc or doc.get("company_id") != company_id:
        raise ChannelOAuthError("This connection request wasn't found — start over.")

    expires_at = doc["expires_at"]
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        await db[OAUTH_PENDING_SELECTIONS_COLLECTION].delete_one({"_id": selection_id})
        raise ChannelOAuthError("This connection request expired — start over.")

    return doc


async def get_pending_selection(
    db: AsyncIOMotorDatabase, company_id: str, selection_id: str
) -> list[OAuthCandidatePage]:
    """Step 3 (list candidates for the picker UI) — never returns tokens,
    encrypted or otherwise."""
    doc = await _get_pending_selection_doc(db, company_id, selection_id)
    return [
        OAuthCandidatePage(
            external_id=c["id"],
            page_name=c["name"],
            linked_instagram_id=c.get("linked_instagram_id"),
            linked_instagram_name=c.get("linked_instagram_name"),
        )
        for c in doc["candidates"]
    ]


async def confirm_channel_connections(
    db: AsyncIOMotorDatabase, company_id: str, selection_id: str, external_ids: list[str]
) -> list[ChannelConnectionDoc]:
    """Step 4 (confirm/save selection): for each chosen Page, save + subscribe
    it, and its linked Instagram account if it has one (same Page Access
    Token authenticates both). Single-use — deletes the pending stash after."""
    doc = await _get_pending_selection_doc(db, company_id, selection_id)
    chosen = {c["id"]: c for c in doc["candidates"] if c["id"] in set(external_ids)}
    if not chosen:
        raise ChannelOAuthError("None of the selected Pages were found in this request.")

    saved: list[ChannelConnectionDoc] = []
    for candidate in chosen.values():
        try:
            page_token = decrypt_token(candidate["encrypted_access_token"])
        except TokenEncryptionError as exc:
            raise ChannelOAuthError("Could not read back this Page's access token.") from exc

        try:
            await subscribe_page_to_app(candidate["id"], page_token)
        except MetaSendError as exc:
            raise ChannelOAuthError(
                f"Connected to Facebook, but couldn't subscribe the Page to webhooks: {exc}"
            ) from exc

        saved.append(await save_channel_connection(
            db, company_id, ChannelType.messenger, candidate["id"], page_token, candidate["name"],
            # Lets the dashboard offer "connect the WhatsApp number on this
            # Page" and prefill WhatsApp Embedded Signup with the business
            # details — WhatsApp itself still has to be verified in Meta's
            # own popup.
            extra_fields=await _fetch_page_contact_info(candidate["id"], page_token),
        ))

        if candidate.get("linked_instagram_id"):
            saved.append(await save_channel_connection(
                db,
                company_id,
                ChannelType.instagram,
                candidate["linked_instagram_id"],
                page_token,
                candidate.get("linked_instagram_name") or candidate["name"],
                # IG DMs arrive through the Page's webhook subscription, so
                # disconnect needs to know which Page this account rides on.
                extra_fields={"linked_page_id": candidate["id"]},
            ))

    await db[OAUTH_PENDING_SELECTIONS_COLLECTION].delete_one({"_id": selection_id})
    return saved


async def _fetch_page_contact_info(page_id: str, page_access_token: str) -> dict[str, str]:
    """Best-effort read of the WhatsApp number and website shown on the
    Page. Never raises — a Page without them (or a lookup Meta rejects)
    just means the dashboard has nothing to suggest."""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                f"{GRAPH_API_BASE_URL}/{page_id}",
                headers={"Authorization": f"Bearer {page_access_token}"},
                params={"fields": "whatsapp_number,website"},
            )
    except httpx.HTTPError as exc:
        logger.warning("channel_oauth.page_contact_lookup_failed page_id=%s error=%s", page_id, exc)
        return {}
    if resp.status_code >= 400:
        logger.warning(
            "channel_oauth.page_contact_lookup_failed page_id=%s status=%d body=%s",
            page_id, resp.status_code, resp.text,
        )
        return {}
    body = resp.json()
    return {
        # Most Pages show WhatsApp as their action button rather than
        # filling the whatsapp_number field, so fall back to that.
        "whatsapp_number": str(body.get("whatsapp_number") or "")
        or await _fetch_whatsapp_action_button_number(page_id, page_access_token),
        "website": str(body.get("website") or ""),
    }


async def _fetch_whatsapp_action_button_number(page_id: str, page_access_token: str) -> str:
    """The number behind the Page's "WhatsApp" action button, if it has one.
    Best-effort like _fetch_page_contact_info — returns "" on any failure."""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                f"{GRAPH_API_BASE_URL}/{page_id}/call_to_actions",
                headers={"Authorization": f"Bearer {page_access_token}"},
                params={"fields": "type,intl_number_with_plus"},
            )
    except httpx.HTTPError as exc:
        logger.warning("channel_oauth.page_cta_lookup_failed page_id=%s error=%s", page_id, exc)
        return ""
    if resp.status_code >= 400:
        logger.warning(
            "channel_oauth.page_cta_lookup_failed page_id=%s status=%d body=%s",
            page_id, resp.status_code, resp.text,
        )
        return ""
    for cta in resp.json().get("data", []):
        if "WHATSAPP" in str(cta.get("type", "")).upper() and cta.get("intl_number_with_plus"):
            return str(cta["intl_number_with_plus"])
    return ""


# ── Graph API: subscribe / unsubscribe a Page's webhook ─────────────────────

async def subscribe_page_to_app(page_id: str, page_access_token: str) -> None:
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"{GRAPH_API_BASE_URL}/{page_id}/subscribed_apps",
            params={"subscribed_fields": "messages", "access_token": page_access_token},
        )
    if resp.status_code >= 400:
        logger.error(
            "channel_oauth.subscribe_failed page_id=%s status=%d body=%s",
            page_id, resp.status_code, resp.text,
        )
        raise MetaSendError(f"Failed to subscribe Page {page_id} to webhooks")


async def subscribe_waba_to_app(waba_id: str, access_token: str) -> None:
    """WhatsApp's /{waba_id}/subscribed_apps takes no subscribed_fields —
    which fields arrive is set once on the App's webhook config instead."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"{GRAPH_API_BASE_URL}/{waba_id}/subscribed_apps",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if resp.status_code >= 400:
        logger.error(
            "whatsapp_connect.subscribe_failed waba_id=%s status=%d body=%s",
            waba_id, resp.status_code, resp.text,
        )
        raise MetaSendError(f"Failed to subscribe WhatsApp account {waba_id} to webhooks")


async def unsubscribe_page_from_app(page_id: str, page_access_token: str) -> None:
    """Works for a Page id or a WABA id — both use DELETE /{id}/subscribed_apps."""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.delete(
                f"{GRAPH_API_BASE_URL}/{page_id}/subscribed_apps",
                params={"access_token": page_access_token},
            )
    except httpx.HTTPError as exc:
        logger.warning("channel_oauth.unsubscribe_network_error id=%s error=%s", page_id, exc)
        return
    if resp.status_code >= 400:
        # Best-effort — don't block a user from disconnecting in our system
        # just because Meta's side failed (token may already be dead, which
        # is often *why* they're disconnecting).
        logger.warning(
            "channel_oauth.unsubscribe_failed page_id=%s status=%d body=%s",
            page_id, resp.status_code, resp.text,
        )


# ── channel_connections CRUD ─────────────────────────────────────────────────

async def save_channel_connection(
    db: AsyncIOMotorDatabase,
    company_id: str,
    channel: ChannelType,
    external_id: str,
    access_token_plaintext: str,
    page_name: str,
    extra_fields: dict[str, Any] | None = None,
) -> ChannelConnectionDoc:
    """`extra_fields` holds channel-specific ids disconnect needs later —
    `linked_page_id` on Instagram rows, `waba_id` on WhatsApp rows."""
    now = datetime.now(timezone.utc)
    encrypted = encrypt_token(access_token_plaintext)
    await db[CHANNEL_CONNECTIONS_COLLECTION].update_one(
        {"company_id": company_id, "channel": channel.value, "external_id": external_id},
        {
            "$set": {
                **(extra_fields or {}),
                "encrypted_access_token": encrypted,
                "page_name": page_name,
                "connected_at": now,
                "status": ConnectionStatus.active.value,
            },
        },
        upsert=True,
    )
    return ChannelConnectionDoc(
        company_id=company_id,
        channel=channel,
        external_id=external_id,
        encrypted_access_token=encrypted,
        page_name=page_name,
        connected_at=now,
        status=ConnectionStatus.active,
        whatsapp_number=(extra_fields or {}).get("whatsapp_number", ""),
        website=(extra_fields or {}).get("website", ""),
    )


def _doc_to_model(doc: dict[str, Any]) -> ChannelConnectionDoc:
    return ChannelConnectionDoc(
        id=str(doc["_id"]),
        company_id=doc["company_id"],
        channel=ChannelType(doc["channel"]),
        external_id=doc["external_id"],
        encrypted_access_token=doc["encrypted_access_token"],
        page_name=doc.get("page_name", ""),
        connected_at=doc["connected_at"],
        status=ConnectionStatus(doc.get("status", ConnectionStatus.active.value)),
        whatsapp_number=doc.get("whatsapp_number", ""),
        website=doc.get("website", ""),
    )


async def list_company_channel_connections(
    db: AsyncIOMotorDatabase, company_id: str
) -> list[ChannelConnectionDoc]:
    cursor = db[CHANNEL_CONNECTIONS_COLLECTION].find({"company_id": company_id})
    return [_doc_to_model(doc) async for doc in cursor]


async def get_channel_connection_by_external_id(
    db: AsyncIOMotorDatabase, channel: ChannelType, external_id: str
) -> ChannelConnectionDoc | None:
    """Reverse lookup used by the webhook — Meta's payload only carries the
    Page/IG id, so this is how an incoming message finds *whose* knowledge
    base to use.

    The same Page/number can have rows under more than one company (e.g.
    company A disconnected it — soft delete keeps the row — and company B
    connected it afterwards), so an active row always wins, newest first.
    Only when no active row exists does it fall back to any row, so the
    handler can still log the specific reason (expired vs. disconnected)."""
    query = {"channel": channel.value, "external_id": external_id}
    newest_first = [("connected_at", -1)]
    collection = db[CHANNEL_CONNECTIONS_COLLECTION]
    doc = await collection.find_one(
        {**query, "status": ConnectionStatus.active.value}, sort=newest_first
    )
    if not doc:
        doc = await collection.find_one(query, sort=newest_first)
    return _doc_to_model(doc) if doc else None


async def mark_channel_connection_status(
    db: AsyncIOMotorDatabase,
    company_id: str,
    channel: ChannelType,
    external_id: str,
    status: ConnectionStatus,
) -> None:
    await db[CHANNEL_CONNECTIONS_COLLECTION].update_one(
        {"company_id": company_id, "channel": channel.value, "external_id": external_id},
        {"$set": {"status": status.value}},
    )


async def ensure_channel_connection_indexes(db: AsyncIOMotorDatabase) -> None:
    """Called once at startup. create_index is a no-op when the index
    already exists, so this is safe to run on every boot."""
    connections = db[CHANNEL_CONNECTIONS_COLLECTION]
    try:
        # One row per (company, channel, Page/number) — matches the upsert
        # key in save_channel_connection, so two concurrent confirms can't
        # insert duplicates.
        await connections.create_index(
            [("company_id", 1), ("channel", 1), ("external_id", 1)], unique=True,
        )
    except Exception:
        # Existing duplicate rows make the unique build fail — log and keep
        # booting rather than taking the whole API down over it.
        logger.exception("channel_connections.unique_index_failed")
    # Webhook lookup (get_channel_connection_by_external_id).
    await connections.create_index([("channel", 1), ("external_id", 1), ("status", 1)])
    # Mongo's TTL monitor deletes abandoned pending selections (and the
    # encrypted tokens inside them) once expires_at passes — otherwise they
    # only get cleaned up if someone happens to open them again.
    await db[OAUTH_PENDING_SELECTIONS_COLLECTION].create_index(
        "expires_at", expireAfterSeconds=0,
    )


async def disconnect_channel_connection(
    db: AsyncIOMotorDatabase, company_id: str, channel: ChannelType, external_id: str
) -> bool:
    """Soft delete — sets status=disconnected, keeps the row for history.
    Then best-effort unsubscribes on Meta's side, but only once nothing
    else still depends on that subscription (see _unsubscribe_if_unused;
    failures there don't block this)."""
    doc = await db[CHANNEL_CONNECTIONS_COLLECTION].find_one(
        {"company_id": company_id, "channel": channel.value, "external_id": external_id}
    )
    if not doc:
        return False

    await db[CHANNEL_CONNECTIONS_COLLECTION].update_one(
        {"_id": doc["_id"]},
        {"$set": {"status": ConnectionStatus.disconnected.value}},
    )
    await _unsubscribe_if_unused(db, doc)
    return True


async def _page_still_in_use(
    db: AsyncIOMotorDatabase, company_id: str, page_id: str, page_token: str
) -> bool:
    """True while any active Messenger connection for this Page, or any
    active Instagram connection riding on it, remains — Meta delivers both
    through the one Page subscription, so unsubscribing it would silently
    cut off the other."""
    collection = db[CHANNEL_CONNECTIONS_COLLECTION]
    active = ConnectionStatus.active.value
    if await collection.find_one(
        {
            "status": active,
            "$or": [
                {"channel": ChannelType.messenger.value, "external_id": page_id},
                {"channel": ChannelType.instagram.value, "linked_page_id": page_id},
            ],
        },
        {"_id": 1},
    ):
        return True

    # Instagram rows saved before linked_page_id existed: they were saved
    # with the same Page token as their Page, so match on that instead.
    async for ig_doc in collection.find(
        {
            "company_id": company_id,
            "channel": ChannelType.instagram.value,
            "status": active,
            "linked_page_id": {"$exists": False},
        },
        {"encrypted_access_token": 1},
    ):
        try:
            if decrypt_token(ig_doc["encrypted_access_token"]) == page_token:
                return True
        except TokenEncryptionError:
            continue
    return False


async def _unsubscribe_if_unused(db: AsyncIOMotorDatabase, doc: dict[str, Any]) -> None:
    channel = ChannelType(doc["channel"])
    external_id = doc["external_id"]

    if channel == ChannelType.whatsapp:
        # Webhooks are subscribed on the WABA, not the phone number.
        target_id = doc.get("waba_id")
    elif channel == ChannelType.instagram:
        target_id = doc.get("linked_page_id")
    else:
        target_id = external_id

    if not target_id:
        logger.info(
            "channel_oauth.disconnect_unsubscribe_skipped channel=%s external_id=%s reason=no_target_id",
            channel.value, external_id,
        )
        return

    try:
        token = decrypt_token(doc["encrypted_access_token"])
    except TokenEncryptionError as exc:
        logger.warning(
            "channel_oauth.disconnect_unsubscribe_skipped channel=%s external_id=%s error=%s",
            channel.value, external_id, exc,
        )
        return

    if channel == ChannelType.whatsapp:
        in_use = await db[CHANNEL_CONNECTIONS_COLLECTION].find_one(
            {
                "channel": ChannelType.whatsapp.value,
                "waba_id": target_id,
                "status": ConnectionStatus.active.value,
            },
            {"_id": 1},
        ) is not None
    else:
        in_use = await _page_still_in_use(db, doc["company_id"], target_id, token)

    if in_use:
        logger.info(
            "channel_oauth.disconnect_unsubscribe_skipped channel=%s external_id=%s reason=still_in_use",
            channel.value, external_id,
        )
        return

    await unsubscribe_page_from_app(target_id, token)


# ── Send API ──────────────────────────────────────────────────────────────

async def send_meta_message(access_token: str, recipient_id: str, text: str) -> None:
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"{GRAPH_API_BASE_URL}/me/messages",
            params={"access_token": access_token},
            json={"recipient": {"id": recipient_id}, "message": {"text": text}},
        )
    if resp.status_code >= 400:
        body = resp.json() if "application/json" in resp.headers.get("content-type", "") else {}
        error = body.get("error", {})
        logger.error(
            "channel_send.failed status=%d error_type=%s error_code=%s",
            resp.status_code, error.get("type"), error.get("code"),
        )
        # OAuthException (and the related "invalid session"/190 code) is
        # Meta's signal that the token itself is the problem, not the
        # request — anything else (rate limit, bad recipient, etc.) is a
        # transient/one-off failure that shouldn't flip the connection to
        # token_expired.
        if error.get("type") == "OAuthException" or error.get("code") == 190:
            raise MetaTokenExpiredError("Page access token is invalid or expired")
        raise MetaSendError(f"Meta send failed: {resp.status_code}")


# ── Incoming message pipeline (used by routers/meta_webhook_router.py) ─────

async def handle_incoming_channel_message(
    db: AsyncIOMotorDatabase,
    channel: ChannelType,
    external_id: str,
    sender_id: str,
    message_id: str,
    message_text: str,
) -> None:
    """Full pipeline for one incoming Messenger/Instagram message: dedupe,
    find the owning connection, ask the AI, send the reply back. Never
    raises — a forged/unmatched/errored message should be dropped, not
    surfaced as a 500 (Meta retries aggressively on non-2xx responses)."""
    try:
        if await is_duplicate_message(db, CHANNEL_PROCESSED_MESSAGES_COLLECTION, message_id):
            logger.info(
                "channel_message.duplicate_skipped channel=%s message_id=%s",
                channel.value, message_id,
            )
            return

        connection = await get_channel_connection_by_external_id(db, channel, external_id)
        if not connection:
            logger.warning(
                "channel_message.unmatched_external_id channel=%s external_id=%s",
                channel.value, external_id,
            )
            return

        if connection.status != ConnectionStatus.active:
            logger.info(
                "channel_message.inactive_connection channel=%s external_id=%s status=%s",
                channel.value, external_id, connection.status.value,
            )
            return

        try:
            access_token = decrypt_token(connection.encrypted_access_token)
        except TokenEncryptionError:
            logger.error(
                "channel_message.token_decrypt_failed channel=%s external_id=%s",
                channel.value, external_id,
            )
            return

        # Neither Messenger nor Instagram has a separate "session" concept —
        # the sender's PSID/IGSID itself is the session, so the same
        # conversation always resumes with the same history.
        reply = await get_ai_reply(
            channel.value, connection.company_id, session_id=sender_id, message=message_text
        )
        if not reply:
            return

        try:
            await send_meta_message(access_token, sender_id, reply)
            logger.info(
                "channel_message.reply_sent channel=%s external_id=%s recipient_id=%s",
                channel.value, external_id, sender_id,
            )
        except MetaTokenExpiredError:
            logger.warning(
                "channel_message.token_expired channel=%s external_id=%s",
                channel.value, external_id,
            )
            await mark_channel_connection_status(
                db, connection.company_id, channel, external_id, ConnectionStatus.token_expired
            )
        except MetaSendError:
            logger.exception(
                "channel_message.send_failed channel=%s external_id=%s",
                channel.value, external_id,
            )
    except Exception:
        logger.exception(
            "channel_message.handler_error channel=%s external_id=%s message_id=%s",
            channel.value, external_id, message_id,
        )


# ═══ WhatsApp (Embedded Signup) ═════════════════════════════════════════════
# Structurally different from Messenger/Instagram: WhatsApp's Send API and
# webhook payload shape aren't the shared /me/messages + entry[].messaging[]
# format, so it gets its own send/receive functions rather than being forced
# through send_meta_message/handle_incoming_channel_message. Everything else
# (ChannelConnectionDoc, token encryption, connection listing/disconnect) is
# already channel-agnostic and reused as-is.


async def connect_whatsapp_embedded(
    db: AsyncIOMotorDatabase, company_id: str, code: str, phone_number_id: str, waba_id: str,
) -> ChannelConnectionDoc:
    """Full Embedded Signup pipeline: code -> long-lived token -> subscribe
    the WABA -> save. `phone_number_id`/`waba_id` come from the frontend's
    WA_EMBEDDED_SIGNUP postMessage event, not looked up here — Meta's own
    signup wizard is what let the user pick/create them."""
    try:
        short_lived = await _exchange_code_for_short_lived_user_token(code)
        access_token = await _exchange_for_long_lived_user_token(short_lived)
    except MetaOAuthExchangeError as exc:
        raise ChannelOAuthError(str(exc)) from exc

    # Required, not best-effort: without this subscription no incoming
    # message ever reaches our webhook, so saving the connection anyway
    # would show "Connected" on a number the bot can never answer.
    try:
        await subscribe_waba_to_app(waba_id, access_token)
    except MetaSendError as exc:
        raise ChannelOAuthError(
            "Couldn't finish connecting WhatsApp — Meta didn't let us subscribe this number "
            "to incoming messages. Please try again."
        ) from exc

    display_name = await _fetch_whatsapp_display_name(phone_number_id, access_token)

    return await save_channel_connection(
        db, company_id, ChannelType.whatsapp, phone_number_id, access_token,
        display_name or phone_number_id,
        extra_fields={"waba_id": waba_id},
    )


async def _fetch_whatsapp_display_name(phone_number_id: str, access_token: str) -> str:
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{GRAPH_API_BASE_URL}/{phone_number_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"fields": "verified_name,display_phone_number"},
        )
    if resp.status_code >= 400:
        logger.warning(
            "whatsapp_connect.name_lookup_failed phone_number_id=%s status=%d body=%s",
            phone_number_id, resp.status_code, resp.text,
        )
        return ""
    body = resp.json()
    return str(body.get("verified_name") or body.get("display_phone_number") or "")


async def send_whatsapp_message(access_token: str, phone_number_id: str, to: str, text: str) -> None:
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"{GRAPH_API_BASE_URL}/{phone_number_id}/messages",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {"body": text},
            },
        )
    if resp.status_code >= 400:
        body = resp.json() if "application/json" in resp.headers.get("content-type", "") else {}
        error = body.get("error", {})
        logger.error(
            "whatsapp_send.failed status=%d error_type=%s error_code=%s",
            resp.status_code, error.get("type"), error.get("code"),
        )
        if error.get("type") == "OAuthException" or error.get("code") == 190:
            raise MetaTokenExpiredError("WhatsApp access token is invalid or expired")
        raise MetaSendError(f"WhatsApp send failed: {resp.status_code}")


async def handle_incoming_whatsapp_message(
    db: AsyncIOMotorDatabase,
    phone_number_id: str,
    sender_wa_id: str,
    message_id: str,
    message_text: str,
) -> None:
    """Full pipeline for one incoming WhatsApp message: dedupe, find the
    owning connection, ask the AI, send the reply back. Never raises — a
    forged/unmatched/errored message should be dropped, not surfaced as a
    500 (Meta retries aggressively on non-2xx responses)."""
    try:
        if await is_duplicate_message(db, CHANNEL_PROCESSED_MESSAGES_COLLECTION, message_id):
            logger.info("whatsapp_message.duplicate_skipped message_id=%s", message_id)
            return

        connection = await get_channel_connection_by_external_id(
            db, ChannelType.whatsapp, phone_number_id
        )
        if not connection:
            logger.warning(
                "whatsapp_message.unmatched_external_id phone_number_id=%s", phone_number_id,
            )
            return

        if connection.status != ConnectionStatus.active:
            logger.info(
                "whatsapp_message.inactive_connection phone_number_id=%s status=%s",
                phone_number_id, connection.status.value,
            )
            return

        try:
            access_token = decrypt_token(connection.encrypted_access_token)
        except TokenEncryptionError:
            logger.error(
                "whatsapp_message.token_decrypt_failed phone_number_id=%s", phone_number_id,
            )
            return

        # WhatsApp has no separate "session" concept — the sender's wa_id
        # itself is the session, so the same conversation always resumes
        # with the same history.
        reply = await get_ai_reply(
            "whatsapp", connection.company_id, session_id=sender_wa_id, message=message_text
        )
        if not reply:
            return

        try:
            await send_whatsapp_message(access_token, phone_number_id, sender_wa_id, reply)
            logger.info(
                "whatsapp_message.reply_sent phone_number_id=%s recipient_id=%s",
                phone_number_id, sender_wa_id,
            )
        except MetaTokenExpiredError:
            logger.warning(
                "whatsapp_message.token_expired phone_number_id=%s", phone_number_id,
            )
            await mark_channel_connection_status(
                db, connection.company_id, ChannelType.whatsapp, phone_number_id,
                ConnectionStatus.token_expired,
            )
        except MetaSendError:
            logger.exception(
                "whatsapp_message.send_failed phone_number_id=%s", phone_number_id,
            )
    except Exception:
        logger.exception(
            "whatsapp_message.handler_error phone_number_id=%s message_id=%s",
            phone_number_id, message_id,
        )
