"""
services/chatbot/tools.py
──────────────────────────
Builds LangChain tools for the ReAct agent.

All tools are optional — a missing package logs a warning, never crashes.
Tools in priority order (agent tries first → last):
  1. knowledge_base  — Pinecone RAG scoped to company_id namespace
  2. web_search      — DuckDuckGo real-time internet search
  3. wikipedia       — Factual background look-ups

Pass `company_id` and optionally the full `company_ctx` dict to get a
retriever scoped to THAT company's vectors, with descriptions personalised
to the company name and domain.
"""

from __future__ import annotations
import logging
from typing import Any

from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from config import settings
from database import get_database
from services.appointments.service import (
    CalendlyAPIError,
    get_calendly_availability,
    get_user_calendly_settings,
)
from .llm import embeddings

logger = logging.getLogger(__name__)


def build_tools(
    company_id: str | None = None,
    company_ctx: dict[str, Any] | None = None,
) -> list[Any]:
    """
    Build the tool list for the agent.

    Args:
        company_id:  When provided, the Pinecone knowledge_base tool will
                     query only this company's namespace.
        company_ctx: Optional company context dict (from company_context.py).
                     Used to personalise tool descriptions and log KB health.

    Returns:
        List of successfully initialised LangChain tools.
    """
    tools: list[Any] = []
    ctx             = company_ctx or {}
    company_name    = ctx.get("company_name", "the company")
    company_type    = ctx.get("company_type", "")
    is_trained      = ctx.get("is_trained", False)
    entries_stored  = ctx.get("entries_stored", 0)
    quality_score   = ctx.get("quality_score", 0.0)
    categories      = ctx.get("categories", [])
    namespace       = ctx.get("namespace") or company_id

    # ── 1. Pinecone knowledge-base RAG ────────────────────────────────────────
    # Namespace = company_id → each company's vectors are fully isolated.
    if company_id:
        if not is_trained or entries_stored == 0:
            logger.warning(
                "chatbot.tool.kb_not_trained company_id=%s is_trained=%s "
                "entries_stored=%d — knowledge_base tool will return empty results",
                company_id, is_trained, entries_stored,
            )
        try:
            from langchain_pinecone import PineconeVectorStore

            vectorstore = PineconeVectorStore(
                index_name=settings.PINECONE_INDEX,
                embedding=embeddings,
                pinecone_api_key=settings.PINECONE_API_KEY,
                namespace=namespace,   # ← scoped to this company's configured namespace
            )
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6},   # increased from 4 for richer context
            )

            # Build a description that tells the LLM exactly what's in the KB
            categories_hint = (
                f" It covers: {', '.join(categories)}." if categories else ""
            )
            kb_description = (
                f"Search {company_name}'s private knowledge base. "
                f"Use this FIRST for any questions about {company_name}'s services, "
                f"team, pricing, process, or policies.{categories_hint} "
                f"Contains {entries_stored} verified facts "
                f"(quality score: {quality_score:.0f}/100)."
            )

            kb_tool = create_retriever_tool(
                retriever,
                name="knowledge_base",
                description=kb_description,
            )
            tools.insert(0, kb_tool)
            logger.info(
                "chatbot.tool.loaded name=knowledge_base company_id=%s "
                "company_name=%r namespace=%s index=%s entries=%d "
                "quality=%.1f categories=%s k=6",
                company_id, company_name, namespace,
                settings.PINECONE_INDEX, entries_stored,
                quality_score, categories,
            )
        except Exception as exc:
            logger.warning(
                "chatbot.tool.skip name=knowledge_base company_id=%s reason=%s",
                company_id, exc,
            )

    # ── 2. DuckDuckGo web search ──────────────────────────────────────────────
    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        web_description = (
            "Search the internet for up-to-date information. "
            "Use for current news, recent events, general facts, "
            "or anything not found in the knowledge base."
        )
        if company_type:
            domain_hints = {
                "law-firm":              "current laws, recent court verdicts, or legal news",
                "healthcare-company":    "medical research, treatment guidelines, or health news",
                "realestate-company":    "property market trends, neighbourhood info, or listings",
                "tech-company":          "tech news, API docs, or software updates",
                "consultancy-company":   "business trends, industry reports, or market data",
                "agency-company":        "marketing trends, design inspiration, or campaign ideas",
            }
            hint = domain_hints.get(company_type, "")
            if hint:
                web_description = (
                    f"Search the internet for up-to-date information, especially {hint}. "
                    "Use when the knowledge base does not have the answer."
                )

        web_tool = DuckDuckGoSearchRun(
            name="web_search",
            description=web_description,
        )
        tools.append(web_tool)
        logger.info("chatbot.tool.loaded name=web_search company_id=%s", company_id)
    except Exception as exc:
        logger.warning(
            "chatbot.tool.skip name=web_search company_id=%s reason=%s",
            company_id, exc,
        )

    # ── 3. Wikipedia ──────────────────────────────────────────────────────────
    try:
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper

        wiki_tool = WikipediaQueryRun(
            name="wikipedia",
            description=(
                "Look up well-established factual background — legal concepts, "
                "medical conditions, places, companies, or people. "
                "Use for stable reference facts, not current events."
            ),
            api_wrapper=WikipediaAPIWrapper(
                top_k_results=2,
                doc_content_chars_max=2000,
            ),
        )
        tools.append(wiki_tool)
        logger.info("chatbot.tool.loaded name=wikipedia company_id=%s", company_id)
    except Exception as exc:
        logger.warning(
            "chatbot.tool.skip name=wikipedia company_id=%s reason=%s",
            company_id, exc,
        )

    # 4. Appointment scheduling (Calendly-backed)
    if company_id:
        try:
            appointment_tools = _build_appointment_tools(company_id)
            tools.extend(appointment_tools)
            logger.info(
                "chatbot.tool.loaded names=%s company_id=%s",
                [tool_obj.name for tool_obj in appointment_tools],  # type: ignore[attr-defined]
                company_id,
            )
        except Exception as exc:
            logger.warning(
                "chatbot.tool.skip names=appointment_tools company_id=%s reason=%s",
                company_id, exc,
            )

    logger.info(
        "chatbot.tools.ready company_id=%s count=%d tools=%s",
        company_id, len(tools), [t.name for t in tools],
    )
    return tools



def _utc_iso_to_local(iso_str: str, user_timezone: str) -> str:
    """
    Convert a UTC ISO-8601 string (e.g. "2026-04-16T03:30:00Z") to a
    human-readable local time string using user_timezone (IANA name).
    Returns the original string on any parsing failure.
    """
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime
        dt_utc = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        tz_obj = ZoneInfo(user_timezone)
        dt_local = dt_utc.astimezone(tz_obj)
        return dt_local.strftime("%B %d, %Y at %I:%M %p") + f" ({user_timezone})"
    except Exception:
        return f"{iso_str} (UTC)"


def _preferred_time_to_utc_fragment(preferred_time: str, user_timezone: str) -> str:
    """
    Given a natural-language preferred time (e.g. '7 pm', '19:00', '3:30 pm')
    and the user's IANA timezone, return the equivalent UTC time fragment (HH:MM)
    for substring-matching against ISO timestamps.

    Returns the original preferred_time string (lowercased) on any failure so
    the caller's fallback logic is unharmed.
    """
    if not preferred_time or not user_timezone:
        return preferred_time.strip().lower()
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime, timezone as dt_tz
        tz_obj = ZoneInfo(user_timezone)
        today = datetime.now(tz_obj).date()
        text = preferred_time.strip().upper()
        local_dt: datetime | None = None
        for fmt in ("%I %p", "%I:%M %p", "%H:%M", "%I%p", "%H"):
            try:
                parsed = datetime.strptime(text, fmt)
                local_dt = parsed.replace(
                    year=today.year, month=today.month, day=today.day,
                    tzinfo=tz_obj,
                )
                break
            except ValueError:
                continue
        if local_dt is None:
            return preferred_time.strip().lower()
        utc_dt = local_dt.astimezone(dt_tz.utc)
        return utc_dt.strftime("%H:%M")  # e.g. "13:00" to match inside ISO string
    except Exception:
        return preferred_time.strip().lower()


def _format_slots_for_tool(slots: list[dict[str, str]], user_timezone: str = "") -> str:
    if not slots:
        return "No available slots were found in the next 7 days."

    lines: list[str] = []
    for idx, slot in enumerate(slots, start=1):
        start_iso = slot["start_time"]
        if user_timezone:
            display = _utc_iso_to_local(start_iso, user_timezone)
        else:
            display = f"{start_iso} (UTC)"
        lines.append(
            f"{idx}. {display} | iso_start_time={start_iso} | confirmation_url={slot['scheduling_url']}"
        )
    return (
        "Available slots:\n"
        + "\n".join(lines)
        + "\n"
        + "IMPORTANT: When presenting slots to the user, show only the human-readable local time "
        "(before the pipe). Use the iso_start_time value when calling get_slot_booking_link. "
        "Ask the user to choose one slot, then share the confirmation page."
    )


def _build_appointment_tools(company_id: str) -> list[Any]:
    @tool(
        "check_appointment_setup",
        description=(
            "Check whether appointment scheduling is configured for this company. "
            "Use before offering appointment slots."
        ),
    )
    async def check_appointment_setup() -> str:
        db = get_database()
        settings_doc = await get_user_calendly_settings(db, company_id)

        if not settings_doc.calendly_access_token.strip():
            return (
                "Appointment scheduling is not configured yet (missing Calendly token). "
                "Offer phone call or email follow-up instead."
            )
        if not settings_doc.event_type_uri.strip():
            return (
                "Calendly is connected but no event type is selected. "
                "Offer phone call or email follow-up, or ask admin to select an event type."
            )
        return "Appointment scheduling is configured and ready."

    @tool(
        "get_available_appointment_slots",
        description=(
            "Get real available appointment slots for the configured Calendly event type "
            "for the next 7 days. Returns times in the user's local timezone when user_timezone is provided. "
            "Pass preferred_time as the user's stated preference (e.g. '7 pm', '14:00') and "
            "user_timezone as the IANA timezone string (e.g. 'Asia/Dhaka', 'America/New_York') "
            "so slots are matched and displayed in the user's local time."
        ),
    )
    async def get_available_appointment_slots(
        preferred_time: str = "",
        user_timezone: str = "",
    ) -> str:
        db = get_database()
        settings_doc = await get_user_calendly_settings(db, company_id)

        if not settings_doc.calendly_access_token.strip():
            return (
                "Cannot fetch slots because Calendly token is missing. "
                "Offer phone or email consultation instead."
            )
        if not settings_doc.event_type_uri.strip():
            return (
                "Cannot fetch slots because event type is not configured. "
                "Offer phone or email consultation instead."
            )

        try:
            slots = await get_calendly_availability(
                settings_doc.calendly_access_token,
                settings_doc.event_type_uri,
            )
        except CalendlyAPIError as exc:
            return (
                f"Could not fetch slots right now: {exc}. "
                "Offer phone or email consultation instead."
            )

        payload = [
            {
                "start_time": slot.start_time,
                "scheduling_url": slot.scheduling_url,
            }
            for slot in slots[:8]
        ]

        # Timezone-aware preferred time matching:
        # Convert the user's natural-language preference (e.g. "7 pm") from
        # their local timezone to a UTC time fragment (e.g. "13:00") so it
        # can be matched against ISO timestamps in the payload.
        # Previously this was a raw string search that never matched anything
        # because "7 pm" never appears in "2026-04-16T13:00:00Z".
        preferred_utc = _preferred_time_to_utc_fragment(preferred_time, user_timezone)
        if preferred_utc:
            payload.sort(
                key=lambda item: 0 if preferred_utc in item["start_time"] else 1
            )

        return _format_slots_for_tool(payload, user_timezone=user_timezone)

    @tool(
        "get_slot_booking_link",
        description=(
            "Get the exact Calendly confirmation URL for a selected slot start_time "
            "(ISO format). Use after the user picks a specific time and shares name/email. "
            "Important: this returns a confirmation page link only; appointment is NOT complete "
            "until the user finishes the Calendly form and confirms completion."
        ),
    )
    async def get_slot_booking_link(slot_start_time: str) -> str:
        db = get_database()
        settings_doc = await get_user_calendly_settings(db, company_id)

        if not settings_doc.calendly_access_token.strip() or not settings_doc.event_type_uri.strip():
            return (
                "Scheduling is not fully configured. "
                "Offer phone or email follow-up instead."
            )

        try:
            slots = await get_calendly_availability(
                settings_doc.calendly_access_token,
                settings_doc.event_type_uri,
            )
        except CalendlyAPIError as exc:
            return (
                f"Could not fetch booking link right now: {exc}. "
                "Offer phone or email consultation instead."
            )

        target = slot_start_time.strip()
        if not target:
            return "Please provide a slot_start_time in ISO format."

        for slot in slots:
            if slot.start_time == target:
                return (
                    f"Appointment confirmation page for {slot.start_time}: {slot.scheduling_url}. "
                    "Status: pending user completion."
                )

        return (
            "Selected slot is no longer available. "
            "Call get_available_appointment_slots again and offer fresh options."
        )

    return [check_appointment_setup, get_available_appointment_slots, get_slot_booking_link]
