"""
services/chatbot/agent.py
──────────────────────────
Provides `build_company_agent(company_id, company_ctx)` which returns a
compiled LangGraph ReAct agent scoped to a specific company.

Changes vs original:
  • company_ctx  — dynamic per-company prompt, company-aware tool descriptions
  • Async lock   — per-company asyncio.Lock prevents race conditions on cache miss
  • invalidate_company_agent() — evicts cache after re-training
  • get_cached_tool_names() — fixed: fallback now returns [] not company IDs
  • All log lines include company_id and thread_id for full traceability
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from .llm import llm
from .tools import build_tools
from .prompts import build_system_prompt, FALLBACK_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ── Per-session memory store (shared across all companies) ────────────────────
# In production with multiple workers swap for a persistent store
# e.g. langgraph.checkpoint.sqlite.SqliteSaver or a Redis-backed checkpointer
_memory = MemorySaver()
_MAX_MEMORY_THREADS = 5000
_memory_thread_order: dict[str, None] = {}

# ── Agent cache with per-company asyncio locks ────────────────────────────────
# { company_id: compiled_agent }
_agent_cache: dict[str, Any]          = {}
# { company_id: asyncio.Lock }
_agent_locks: dict[str, asyncio.Lock] = {}
_locks_meta_lock = asyncio.Lock()   # protects the _agent_locks dict itself
_MAX_AGENT_CACHE_SIZE = 256


async def _get_lock(company_id: str) -> asyncio.Lock:
    """Return (or create) a per-company asyncio.Lock."""
    async with _locks_meta_lock:
        if company_id not in _agent_locks:
            _agent_locks[company_id] = asyncio.Lock()
        return _agent_locks[company_id]


# ── Public API ────────────────────────────────────────────────────────────────

async def build_company_agent(
    company_id: str,
    company_ctx: dict[str, Any] | None = None,
) -> Any:
    """
    Return a compiled LangGraph ReAct agent for the given company.

    • If already cached, returns it immediately (fast path).
    • Otherwise builds tools + prompt from company_ctx and compiles the graph.
    • Uses a per-company asyncio.Lock to prevent duplicate builds under concurrency.

    Args:
        company_id:  MongoDB ObjectId string for the company.
        company_ctx: Dict from `company_context.get_company_context()`.
                     If None, a generic fallback prompt is used.
    """
    # ── Fast path: cache hit ──────────────────────────────────────────────────
    if company_id in _agent_cache:
        cached = _agent_cache.pop(company_id)
        _agent_cache[company_id] = cached
        logger.debug("chatbot.agent.cache_hit company_id=%s", company_id)
        return cached

    # ── Slow path: build under per-company lock ───────────────────────────────
    lock = await _get_lock(company_id)
    async with lock:
        # Double-check after acquiring lock
        if company_id in _agent_cache:
            logger.debug(
                "chatbot.agent.cache_hit_after_lock company_id=%s", company_id
            )
            cached = _agent_cache.pop(company_id)
            _agent_cache[company_id] = cached
            return cached

        is_trained     = (company_ctx or {}).get("is_trained", False)
        entries_stored = (company_ctx or {}).get("entries_stored", 0)
        company_name   = (company_ctx or {}).get("company_name", "unknown")

        logger.info(
            "chatbot.agent.build company_id=%s company_name=%r "
            "is_trained=%s entries_stored=%d",
            company_id, company_name, is_trained, entries_stored,
        )

        # Build tools with company context so descriptions are personalised
        tools = build_tools(company_id=company_id, company_ctx=company_ctx)

        # Build the system prompt from company context
        if company_ctx:
            system_prompt = build_system_prompt(company_ctx)
            logger.debug(
                "chatbot.agent.prompt_built company_id=%s prompt_len=%d",
                company_id, len(system_prompt),
            )
        else:
            system_prompt = FALLBACK_SYSTEM_PROMPT
            logger.warning(
                "chatbot.agent.using_fallback_prompt company_id=%s "
                "reason=no_company_ctx",
                company_id,
            )

        agent = create_react_agent(
            model=llm,
            tools=tools,
            checkpointer=_memory,
            prompt=system_prompt,
        )
        _agent_cache[company_id] = agent
        _trim_agent_cache_if_needed()

        logger.info(
            "chatbot.agent.ready company_id=%s company_name=%r "
            "tool_count=%d tools=%s",
            company_id, company_name, len(tools), [t.name for t in tools],
        )
        return agent


def register_thread_activity(thread_id: str) -> None:
    """
    Track active LangGraph thread_ids and evict oldest checkpoint data when needed.
    """
    if not thread_id:
        return

    if thread_id in _memory_thread_order:
        _memory_thread_order.pop(thread_id, None)
    _memory_thread_order[thread_id] = None

    while len(_memory_thread_order) > _MAX_MEMORY_THREADS:
        oldest_thread_id = next(iter(_memory_thread_order))
        _memory_thread_order.pop(oldest_thread_id, None)
        try:
            _memory.delete_thread(oldest_thread_id)
            logger.info(
                "chatbot.agent.memory_thread_evicted thread_id=%s reason=max_size",
                oldest_thread_id,
            )
        except Exception as exc:
            logger.warning(
                "chatbot.agent.memory_thread_evict_failed thread_id=%s error=%s",
                oldest_thread_id,
                exc,
            )


def invalidate_company_agent(company_id: str) -> None:
    """
    Evict a company's agent from the cache.
    Call this after re-training so the next request rebuilds with fresh KB data.
    """
    evicted = _agent_cache.pop(company_id, None)
    _agent_locks.pop(company_id, None)
    if evicted is not None:
        logger.info(
            "chatbot.agent.cache_evicted company_id=%s reason=invalidated",
            company_id,
        )
    else:
        logger.debug(
            "chatbot.agent.cache_evict_noop company_id=%s reason=not_cached",
            company_id,
        )


def _trim_agent_cache_if_needed() -> None:
    while len(_agent_cache) > _MAX_AGENT_CACHE_SIZE:
        evicted_company_id = next(iter(_agent_cache))
        _agent_cache.pop(evicted_company_id, None)
        _agent_locks.pop(evicted_company_id, None)
        logger.info(
            "chatbot.agent.cache_evicted company_id=%s reason=max_size",
            evicted_company_id,
        )


def get_cached_tool_names(company_id: str) -> list[str]:
    """
    Return tool names for a cached agent, or empty list if not yet built.
    Safe fallback: returns [] (not company IDs) if agent.tools is missing.
    """
    agent = _agent_cache.get(company_id)
    if agent is None:
        logger.debug(
            "chatbot.agent.get_tools_noop company_id=%s reason=not_cached",
            company_id,
        )
        return []
    try:
        names = [t.name for t in agent.tools]  # type: ignore[attr-defined]
        logger.debug(
            "chatbot.agent.get_tools company_id=%s tools=%s", company_id, names
        )
        return names
    except Exception as exc:
        logger.warning(
            "chatbot.agent.get_tools_error company_id=%s error=%s",
            company_id, exc,
        )
        return []
