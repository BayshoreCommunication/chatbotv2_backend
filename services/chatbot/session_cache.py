"""
services/chatbot/session_cache.py
──────────────────────────────────
Layer-2 in-memory session cache, keyed by thread_id ("company_id:session_id").

Stores per-visitor state so that:
  • Lead info (name / phone / email) captured in message N is instantly
    available for message N+1 — no round-trip to MongoDB.
  • Company context is reused within the session — no extra DB hit after
    the first message.
  • User timezone is remembered for the lifetime of the session.

TTL: 30 minutes of inactivity.  Expired entries are evicted lazily on every
write.  No background thread required.

Thread safety: one asyncio.Lock per session (same pattern as agent.py).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_SESSION_TTL = 1800  # 30 minutes of inactivity
_SESSION_MAX_SIZE = 5000

# ── In-memory store ───────────────────────────────────────────────────────────
# { thread_id: SessionData }
_session_store: dict[str, "SessionData"] = {}

# Per-session asyncio locks (same double-lock pattern as agent.py)
_session_locks: dict[str, asyncio.Lock] = {}
_locks_meta_lock = asyncio.Lock()


def _mask_value(value: str | None, keep_tail: int = 2) -> str:
    if not value:
        return "none"
    if len(value) <= keep_tail:
        return "*" * len(value)
    return ("*" * (len(value) - keep_tail)) + value[-keep_tail:]


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class SessionData:
    thread_id:     str
    company_id:    str
    company_ctx:   dict[str, Any]   = field(default_factory=dict)
    lead_name:     str | None       = None
    lead_phone:    str | None       = None
    lead_email:    str | None       = None
    lead_captured: bool             = False
    user_timezone: str | None       = None
    expires_at:    float            = field(default_factory=lambda: time.monotonic() + _SESSION_TTL)

    def is_alive(self) -> bool:
        return time.monotonic() < self.expires_at

    def touch(self) -> None:
        """Reset TTL on activity."""
        self.expires_at = time.monotonic() + _SESSION_TTL

    def to_lead_dict(self) -> dict[str, Any]:
        """Return lead fields as a plain dict (for router convenience)."""
        return {
            "lead_captured": self.lead_captured,
            "lead_name":     self.lead_name,
            "lead_phone":    self.lead_phone,
            "lead_email":    self.lead_email,
        }


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _get_lock(thread_id: str) -> asyncio.Lock:
    """Return (or create) a per-session asyncio.Lock. Created lazily so it
    is always bound to the currently running event loop.
    Uses _locks_meta_lock to prevent a race where two concurrent coroutines
    both pass the `if` check and overwrite each other's lock object."""
    async with _locks_meta_lock:
        if thread_id not in _session_locks:
            _session_locks[thread_id] = asyncio.Lock()
        return _session_locks[thread_id]


def _cleanup_expired() -> None:
    """Lazily evict expired sessions.  Called on every write."""
    dead = [tid for tid, s in _session_store.items() if not s.is_alive()]
    for tid in dead:
        _session_store.pop(tid, None)
        _session_locks.pop(tid, None)
        logger.debug("session_cache.evicted thread_id=%s reason=ttl_expired", tid)
    if dead:
        logger.info("session_cache.cleanup evicted=%d remaining=%d", len(dead), len(_session_store))


def _trim_max_sessions_if_needed() -> None:
    while len(_session_store) > _SESSION_MAX_SIZE:
        oldest_thread_id = next(iter(_session_store))
        _session_store.pop(oldest_thread_id, None)
        _session_locks.pop(oldest_thread_id, None)
        logger.info(
            "session_cache.evicted thread_id=%s reason=max_size remaining=%d",
            oldest_thread_id,
            len(_session_store),
        )


# ── Public API ────────────────────────────────────────────────────────────────

def get_session(thread_id: str) -> SessionData | None:
    """
    Return the cached SessionData if it exists and is still alive.
    Returns None on cache miss or TTL expiry (caller should query MongoDB).
    """
    session = _session_store.get(thread_id)
    if session is None:
        logger.debug("session_cache.miss thread_id=%s", thread_id)
        return None
    if not session.is_alive():
        _session_store.pop(thread_id, None)
        _session_locks.pop(thread_id, None)
        logger.debug("session_cache.expired thread_id=%s", thread_id)
        return None
    logger.debug(
        "session_cache.hit thread_id=%s lead_captured=%s",
        thread_id, session.lead_captured,
    )
    return session


async def create_or_refresh_session(
    thread_id:     str,
    company_id:    str,
    company_ctx:   dict[str, Any],
    lead_state:    dict[str, Any] | None = None,
    user_timezone: str | None            = None,
) -> SessionData:
    """
    Create a new session entry or refresh an existing one.

    Args:
        thread_id:     "company_id:session_id" composite key.
        company_id:    MongoDB ObjectId string.
        company_ctx:   Dict from get_company_context().
        lead_state:    Optional dict with lead_captured/lead_name/phone/email
                       loaded from MongoDB on the first message.
        user_timezone: Auto-detected timezone string from the request.

    Returns:
        The new or refreshed SessionData.
    """
    lock = await _get_lock(thread_id)
    async with lock:
        _cleanup_expired()

        existing = _session_store.get(thread_id)
        if existing and existing.is_alive():
            # Refresh timezone if newly provided
            if user_timezone and not existing.user_timezone:
                existing.user_timezone = user_timezone
            existing.touch()
            logger.debug("session_cache.refresh thread_id=%s", thread_id)
            return existing

        ls = lead_state or {}
        session = SessionData(
            thread_id=thread_id,
            company_id=company_id,
            company_ctx=company_ctx,
            lead_name=ls.get("lead_name"),
            lead_phone=ls.get("lead_phone"),
            lead_email=ls.get("lead_email"),
            lead_captured=bool(ls.get("lead_captured", False)),
            user_timezone=user_timezone,
        )
        _session_store[thread_id] = session
        _trim_max_sessions_if_needed()

        logger.info(
            "session_cache.created thread_id=%s company_id=%s "
            "lead_captured=%s user_timezone=%r",
            thread_id, company_id, session.lead_captured, user_timezone,
        )
        return session


def update_session_lead(
    thread_id:  str,
    name:       str | None = None,
    phone:      str | None = None,
    email:      str | None = None,
) -> bool:
    """
    Merge newly extracted lead fields into the cached session.
    Resets the TTL on any successful write.

    Returns True if the session existed and was updated, False otherwise.
    """
    session = _session_store.get(thread_id)
    if session is None or not session.is_alive():
        logger.debug(
            "session_cache.update_lead_noop thread_id=%s reason=not_cached", thread_id
        )
        return False

    updated = False
    # Allow overwriting so a user can correct a previously given value
    # (e.g., they mistyped their phone number and re-send the correct one).
    if name:
        if session.lead_name != name:
            session.lead_name = name
            updated = True
    if phone:
        if session.lead_phone != phone:
            session.lead_phone = phone
            updated = True
    if email:
        if session.lead_email != email:
            session.lead_email = email
            updated = True

    if updated:
        # A name alone is NOT enough — only mark lead_captured once we have
        # a real contact method (phone or email) that lets the team reach out.
        if session.lead_phone or session.lead_email:
            session.lead_captured = True
        session.touch()
        logger.debug(
            "session_cache.lead_updated thread_id=%s lead_captured=%s name=%r phone=%r email=%r",
            thread_id,
            session.lead_captured,
            _mask_value(session.lead_name),
            _mask_value(session.lead_phone),
            _mask_value(session.lead_email),
        )

    return True


def invalidate_session(thread_id: str) -> None:
    """Evict a session from the cache (e.g., on explicit reset or test teardown)."""
    evicted = _session_store.pop(thread_id, None)
    _session_locks.pop(thread_id, None)
    if evicted:
        logger.info("session_cache.invalidated thread_id=%s", thread_id)
    else:
        logger.debug("session_cache.invalidate_noop thread_id=%s reason=not_cached", thread_id)
