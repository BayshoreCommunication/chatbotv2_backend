"""
services/chatbot/company_context.py
────────────────────────────────────
Loads a company's full profile from MongoDB and returns a context dict
used to build the dynamic system prompt and the knowledge-base tool.

Data sources (both joined into one dict):
  • users       collection  — company_name, company_type, company_website, train_data
  • knowledge_base collection — entries_stored, quality_score, categories, namespace

Result is cached in-process with a 5-minute TTL so we do not hit Mongo on
every single chat message.  Call `invalidate_context(company_id)` after
re-training to force a fresh load.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId

from database import get_database

logger = logging.getLogger(__name__)

# ── In-process context cache ──────────────────────────────────────────────────
# { company_id: {"data": {...}, "expires": float} }
_ctx_cache: dict[str, dict[str, Any]] = {}
_ctx_lock  = asyncio.Lock()
_CTX_TTL   = 300  # seconds (5 minutes)


def invalidate_context(company_id: str) -> None:
    """Remove a company's cached context (call after re-training)."""
    _ctx_cache.pop(company_id, None)
    logger.info("company_context.invalidated company_id=%s", company_id)


async def get_company_context(company_id: str) -> dict[str, Any] | None:
    """
    Return a context dict for the given company_id, or None if not found.

    Returned dict keys:
        company_id      str
        company_name    str
        company_type    str
        company_website str | None
        is_trained      bool
        entries_stored  int
        quality_score   float
        categories      list[str]
        namespace       str          ← same as company_id when trained
        is_active       bool
    """
    t0 = time.monotonic()

    # ── Cache hit ─────────────────────────────────────────────────────────────
    cached = _ctx_cache.get(company_id)
    if cached and cached["expires"] > time.monotonic():
        elapsed = int((time.monotonic() - t0) * 1000)
        logger.debug(
            "company_context.cache_hit company_id=%s elapsed_ms=%d",
            company_id, elapsed,
        )
        return cached["data"]

    # ── Validate ObjectId ─────────────────────────────────────────────────────
    try:
        oid = ObjectId(company_id)
    except (InvalidId, TypeError):
        logger.warning(
            "company_context.invalid_id company_id=%s reason=not_a_valid_objectid",
            company_id,
        )
        return None

    # ── Load from MongoDB ─────────────────────────────────────────────────────
    async with _ctx_lock:
        # Double-check after acquiring lock
        cached = _ctx_cache.get(company_id)
        if cached and cached["expires"] > time.monotonic():
            return cached["data"]

        db = get_database()

        # 1) users collection — company profile
        user_doc = await db["users"].find_one({"_id": oid})
        if not user_doc:
            logger.warning(
                "company_context.not_found company_id=%s collection=users",
                company_id,
            )
            return None

        train_data = user_doc.get("train_data") or {}

        # 2) knowledge_base collection — KB stats (optional, may not exist yet)
        kb_doc = await db["knowledge_base"].find_one({"company_id": company_id})

        is_trained     = bool(train_data.get("is_trained", False))
        entries_stored = int(train_data.get("entries_stored", 0))
        quality_score  = float(train_data.get("score", 0.0))
        categories     = train_data.get("categories") or []
        namespace      = train_data.get("namespace") or company_id

        # Prefer KB doc stats when available (more detailed)
        if kb_doc:
            entries_stored = int(kb_doc.get("entries_stored", entries_stored))
            quality_score  = float(kb_doc.get("quality_score", quality_score))
            categories     = kb_doc.get("categories", categories)

        ctx: dict[str, Any] = {
            "company_id":      company_id,
            "company_name":    user_doc.get("company_name", "the company"),
            "company_type":    user_doc.get("company_type", "other"),
            "company_website": user_doc.get("company_website"),
            "is_trained":      is_trained,
            "entries_stored":  entries_stored,
            "quality_score":   quality_score,
            "categories":      categories,
            "namespace":       namespace,
            "is_active":       bool(user_doc.get("is_active", True)),
        }

        _ctx_cache[company_id] = {"data": ctx, "expires": time.monotonic() + _CTX_TTL}

        elapsed = int((time.monotonic() - t0) * 1000)
        logger.info(
            "company_context.loaded company_id=%s company_name=%r company_type=%s "
            "is_trained=%s entries_stored=%d quality_score=%.1f categories=%s elapsed_ms=%d",
            ctx["company_id"],
            ctx["company_name"],
            ctx["company_type"],
            ctx["is_trained"],
            ctx["entries_stored"],
            ctx["quality_score"],
            ctx["categories"],
            elapsed,
        )
        return ctx
