"""
routers/chat_router.py
───────────────────────
Thin FastAPI router — only HTTP endpoint logic lives here.
All AI logic (LLM, tools, agent, prompts) lives in services/chatbot/.

Endpoints:
  POST /chat/{company_id}         — send a message, get an AI reply
  GET  /chat/{company_id}/status  — agent + tool health for this company
"""

from __future__ import annotations
import logging
import time
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Header, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel

from database import get_database
from services.chatbot import (
    build_company_agent,
    get_cached_tool_names,
    register_thread_activity,
)
from services.chatbot.company_context import get_company_context
from services.chatbot.lead_extractor import LeadInfo, extract_lead_info_async
from services.chatbot.session_cache import (
    create_or_refresh_session,
    get_session,
    update_session_lead,
)
from services.chatbot.tools import build_tools
from services.chatbot.ws_manager import ws_manager

router = APIRouter(prefix="/chat", tags=["Chat"])
widget_router = APIRouter(prefix="/chatbot", tags=["Widget Chat"])
logger = logging.getLogger(__name__)


# ── WebSocket: dashboard owner ────────────────────────────────────────────────

@router.websocket("/{company_id}/ws")
async def dashboard_ws(company_id: str, websocket: WebSocket):
    await ws_manager.connect_dashboard(company_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive pings
    except WebSocketDisconnect:
        ws_manager.disconnect_dashboard(company_id, websocket)


# ── WebSocket: widget visitor ─────────────────────────────────────────────────

@widget_router.websocket("/ws")
async def widget_ws(
    websocket: WebSocket,
    api_key: str = Query(..., alias="apiKey"),
    session_id: str = Query(default="default", alias="sessionId"),
):
    company_id = api_key[4:] if api_key.startswith("org-") else api_key
    if not ObjectId.is_valid(company_id):
        await websocket.close(code=4001)
        return

    session_key = f"{company_id}:{session_id}"
    await ws_manager.connect_widget(session_key, websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive pings from widget
    except WebSocketDisconnect:
        ws_manager.disconnect_widget(session_key)


# ── Takeover: toggle ──────────────────────────────────────────────────────────

class TakeoverRequest(BaseModel):
    active: bool


@router.patch("/{company_id}/{session_id}/takeover")
async def toggle_takeover(company_id: str, session_id: str, payload: TakeoverRequest):
    db = get_database()
    now = datetime.now(timezone.utc)
    await db["chat_sessions"].update_one(
        {"company_id": company_id, "session_id": session_id},
        {"$set": {"human_takeover": payload.active, "updated_at": now}},
        upsert=True,
    )
    session_key = f"{company_id}:{session_id}"
    await ws_manager.push_to_widget(session_key, {
        "type": "takeover_status",
        "active": payload.active,
    })
    return {"ok": True, "human_takeover": payload.active}


# ── Owner reply ───────────────────────────────────────────────────────────────

class OwnerReplyRequest(BaseModel):
    content: str


@router.post("/{company_id}/{session_id}/owner-reply")
async def owner_reply(company_id: str, session_id: str, payload: OwnerReplyRequest):
    if not payload.content.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Reply cannot be empty.")
    db = get_database()
    now = datetime.now(timezone.utc)
    msg = {
        "role": "assistant",
        "content": payload.content.strip(),
        "timestamp": now,
        "source": "human",
    }
    await db["chat_sessions"].update_one(
        {"company_id": company_id, "session_id": session_id},
        {"$push": {"messages": msg}, "$set": {"updated_at": now}},
    )
    session_key = f"{company_id}:{session_id}"
    await ws_manager.push_to_widget(session_key, {
        "type": "owner_reply",
        "content": payload.content.strip(),
        "timestamp": now.isoformat(),
    })
    return {"ok": True, "timestamp": now.isoformat()}


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = "default"
    visitor_id: str | None = None
    message: str
    user_timezone: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    company_id: str
    tools_used: list[str] = []


class StatusResponse(BaseModel):
    company_id: str
    company_name: str
    company_type: str
    is_trained: bool
    entries_stored: int
    quality_score: float
    categories: list[str]
    tools_available: list[str]
    agent_cached: bool


class HistoryMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime | None = None


class ConversationHistoryItem(BaseModel):
    session_id: str
    exchange_count: int
    lead_captured: bool
    lead_name: str | None = None
    lead_phone: str | None = None
    lead_email: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    tools_used_history: list[list[str]]
    messages: list[HistoryMessage]


class ConversationHistoryResponse(BaseModel):
    company_id: str
    total_sessions: int
    limit: int
    offset: int
    sessions: list[ConversationHistoryItem]


# ── POST /chat/{company_id} ───────────────────────────────────────────────────

@router.post(
    "/{company_id}",
    response_model=ChatResponse,
    summary="Chat with company AI",
)
async def chat(company_id: str, payload: ChatRequest):
    """
    Send a message to the AI assistant for a specific company.

    - `company_id` — MongoDB ObjectId of the company (from signup).
    - `session_id` — any string; same ID preserves conversation memory.
    - `message`    — the visitor's question or message.

    The agent uses the company's trained Pinecone knowledge base (if trained),
    DuckDuckGo web search, and Wikipedia as fallbacks.
    """
    t_start = time.monotonic()

    # ── Validate message ──────────────────────────────────────────────────────
    if not payload.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty.",
        )

    logger.info(
        "chat.request.in company_id=%s session_id=%s message_len=%d user_timezone=%r",
        company_id, payload.session_id, len(payload.message), payload.user_timezone,
    )

    # ── Load company context ──────────────────────────────────────────────────
    t_ctx = time.monotonic()
    ctx = await get_company_context(company_id)
    if ctx is None:
        logger.warning(
            "chat.request.rejected company_id=%s reason=company_not_found",
            company_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found.",
        )

    elapsed_ctx = int((time.monotonic() - t_ctx) * 1000)
    logger.info(
        "chat.context.loaded company_id=%s company_name=%r is_trained=%s "
        "entries_stored=%d elapsed_ms=%d",
        company_id, ctx["company_name"], ctx["is_trained"],
        ctx["entries_stored"], elapsed_ctx,
    )

    # ── Get (or build + cache) the agent for this company ────────────────────
    t_agent = time.monotonic()
    agent = await build_company_agent(company_id, company_ctx=ctx)
    elapsed_agent = int((time.monotonic() - t_agent) * 1000)
    logger.debug(
        "chat.agent.ready company_id=%s elapsed_ms=%d", company_id, elapsed_agent
    )

    # ── Session: track for cache TTL + visitor lead restore ──────────────────
    thread_id = f"{company_id}:{payload.session_id}"
    session = get_session(thread_id)
    if session is None:
        db_lead = await _get_session_lead_state(company_id, payload.session_id)

        # Returning visitor: carry forward contact info from a previous session
        # so the LLM is not asking for details it already has in MongoDB.
        if not db_lead.get("lead_captured") and payload.visitor_id:
            visitor_lead = await _get_visitor_lead_state(company_id, payload.visitor_id)
            if visitor_lead and visitor_lead.get("lead_captured"):
                db_lead = visitor_lead
                logger.info(
                    "chat.visitor.lead_restored company_id=%s visitor_id=%s session_id=%s",
                    company_id, payload.visitor_id, payload.session_id,
                )

        session = await create_or_refresh_session(
            thread_id=thread_id,
            company_id=company_id,
            company_ctx=ctx,
            lead_state=db_lead,
            user_timezone=(payload.user_timezone or "").strip() or None,
        )
    else:
        if payload.user_timezone and not session.user_timezone:
            session.user_timezone = (payload.user_timezone or "").strip() or None

    # ── Invoke agent — LLM is in full control ────────────────────────────────
    # Router passes ONLY the user's timezone — the one piece of context the LLM
    # cannot infer from conversation history.  Everything else (lead capture,
    # tool selection, appointment flow) is the LLM's responsibility via its
    # system prompt and the tools available to it.
    config = {"configurable": {"thread_id": thread_id}}
    register_thread_activity(thread_id)

    logger.info(
        "chat.agent.invoke company_id=%s session_id=%s thread_id=%s",
        company_id, payload.session_id, thread_id,
    )

    user_timezone = (payload.user_timezone or "").strip()
    input_messages: list[tuple[str, str]] = []
    if user_timezone:
        input_messages.append((
            "system",
            f"User timezone: {user_timezone} (auto-detected). "
            "Use this for all scheduling and slot display. "
            "Do not ask for timezone unless the user explicitly requests a change.",
        ))
    input_messages.append(("user", payload.message))

    t_invoke = time.monotonic()
    try:
        result = await agent.ainvoke(
            {"messages": input_messages},
            config=config,
        )
    except Exception as exc:
        logger.exception(
            "chat.agent.failed company_id=%s session_id=%s error=%s",
            company_id, payload.session_id, exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {str(exc)}",
        )

    elapsed_invoke = int((time.monotonic() - t_invoke) * 1000)

    # ── Extract the last AI reply ─────────────────────────────────────────────
    messages = result.get("messages", [])
    turn_messages = _slice_current_turn_messages(messages, payload.message)
    reply = ""
    for msg in reversed(turn_messages):
        if hasattr(msg, "content") and getattr(msg, "type", "") == "ai":
            reply = msg.content
            break
    if not reply and turn_messages:
        reply = str(turn_messages[-1].content)

    # ── Collect which tools were actually called this turn ────────────────────
    tools_used: list[str] = []
    for msg in turn_messages:
        if getattr(msg, "type", "") == "tool":
            name = getattr(msg, "name", "")
            if name and name not in tools_used:
                tools_used.append(name)

    elapsed_total = int((time.monotonic() - t_start) * 1000)
    logger.info(
        "chat.response.out company_id=%s session_id=%s reply_len=%d "
        "tools_used=%s invoke_ms=%d total_ms=%d",
        company_id, payload.session_id, len(reply),
        tools_used, elapsed_invoke, elapsed_total,
    )

    # ── Side-effect: extract lead info, update cache, persist ────────────────
    # Happens AFTER the LLM reply — does NOT influence the response content.
    if _looks_like_contact_share_message(payload.message):
        message_lead = await extract_lead_info_async(payload.message)
    else:
        message_lead = LeadInfo()

    if message_lead.has_any:
        update_session_lead(
            thread_id=thread_id,
            name=message_lead.name,
            phone=message_lead.phone,
            email=message_lead.email,
        )

    await _persist_exchange(
        company_id=company_id,
        session_id=payload.session_id,
        user_message=payload.message,
        ai_reply=reply,
        tools_used=tools_used,
        lead_info=message_lead,
        visitor_id=payload.visitor_id,
    )

    return ChatResponse(
        reply=reply,
        session_id=payload.session_id,
        company_id=company_id,
        tools_used=tools_used,
    )


# ── GET /chat/{company_id}/status ─────────────────────────────────────────────

@router.get(
    "/{company_id}/status",
    response_model=StatusResponse,
    summary="Agent + tool health for this company",
)
async def chat_status(company_id: str):
    """
    Returns the current health and readiness of the AI agent for a company:
    - Whether the company KB is trained and how many facts it holds
    - Which tools the agent has available
    - Whether the agent is currently cached (warm) or will need building on next request
    """
    logger.info("chat.status.check company_id=%s", company_id)

    ctx = await get_company_context(company_id)
    if ctx is None:
        logger.warning(
            "chat.status.rejected company_id=%s reason=company_not_found",
            company_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found.",
        )

    agent_cached    = company_id in _get_agent_cache()
    tools_available = get_cached_tool_names(company_id)
    if not tools_available:
        tools_available = [t.name for t in build_tools(company_id=company_id, company_ctx=ctx)]

    logger.info(
        "chat.status.ok company_id=%s company_name=%r is_trained=%s "
        "entries_stored=%d agent_cached=%s tools_available=%s",
        company_id, ctx["company_name"], ctx["is_trained"],
        ctx["entries_stored"], agent_cached, tools_available,
    )

    return StatusResponse(
        company_id=company_id,
        company_name=ctx["company_name"],
        company_type=ctx["company_type"],
        is_trained=ctx["is_trained"],
        entries_stored=ctx["entries_stored"],
        quality_score=ctx["quality_score"],
        categories=ctx["categories"],
        tools_available=tools_available,
        agent_cached=agent_cached,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

@router.get(
    "/{company_id}/history",
    response_model=ConversationHistoryResponse,
    summary="Conversation history by company",
)
async def conversation_history(
    company_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Return paginated conversation history for a company from `chat_sessions`."""
    ctx = await get_company_context(company_id)
    if ctx is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found.",
        )

    db = get_database()
    query = {"company_id": company_id}
    total_sessions = await db["chat_sessions"].count_documents(query)

    docs = await (
        db["chat_sessions"]
        .find(query)
        .sort("updated_at", -1)
        .skip(offset)
        .limit(limit)
        .to_list(length=limit)
    )

    sessions: list[ConversationHistoryItem] = []
    for doc in docs:
        messages = [
            HistoryMessage(
                role=str(msg.get("role", "")),
                content=str(msg.get("content", "")),
                timestamp=msg.get("timestamp"),
            )
            for msg in doc.get("messages", [])
            if isinstance(msg, dict)
        ]
        tools_used_history = [
            [str(tool) for tool in turn_tools if isinstance(tool, str)]
            for turn_tools in doc.get("tools_used_history", [])
            if isinstance(turn_tools, list)
        ]
        sessions.append(
            ConversationHistoryItem(
                session_id=str(doc.get("session_id", "")),
                exchange_count=int(doc.get("exchange_count", 0)),
                lead_captured=bool(doc.get("lead_captured", False)),
                lead_name=doc.get("lead_name"),
                lead_phone=doc.get("lead_phone"),
                lead_email=doc.get("lead_email"),
                created_at=doc.get("created_at"),
                updated_at=doc.get("updated_at"),
                tools_used_history=tools_used_history,
                messages=messages,
            )
        )

    logger.info(
        "chat.history.ok company_id=%s total_sessions=%d returned=%d limit=%d offset=%d",
        company_id, total_sessions, len(sessions), limit, offset,
    )
    return ConversationHistoryResponse(
        company_id=company_id,
        total_sessions=total_sessions,
        limit=limit,
        offset=offset,
        sessions=sessions,
    )


def _get_agent_cache() -> dict:
    """Access the agent cache dict for status checks (avoids circular import)."""
    from services.chatbot import agent as agent_module
    return agent_module._agent_cache


async def _persist_exchange(
    company_id: str,
    session_id: str,
    user_message: str,
    ai_reply: str,
    tools_used: list[str],
    lead_info: LeadInfo | None = None,
    visitor_id: str | None = None,
) -> None:
    """
    Upsert this chat exchange into the `chat_sessions` MongoDB collection.

    Steps performed on every turn:
      1. Append user + AI messages to the session.
      2. Reuse lead info extracted during this turn (single LLM extraction).
      3. If new lead data is found, set `lead_captured = True` and store the
         contact fields (lead_name, lead_phone, lead_email) on the session.
      4. Upsert a document in the `leads` collection so the company can view
         all captured leads in one place.
    """
    try:
        db  = get_database()
        now = datetime.now(timezone.utc)

        # Reuse extracted lead info from chat() to avoid a second LLM call.
        if lead_info is None:
            lead_info = await extract_lead_info_async(user_message)

        # ── 2. Build the $set payload ─────────────────────────────────────────
        set_fields: dict = {"updated_at": now}
        if visitor_id:
            set_fields["visitor_id"] = visitor_id

        if lead_info.has_any:
            if lead_info.name:
                set_fields["lead_name"]  = lead_info.name
            if lead_info.phone:
                set_fields["lead_phone"] = lead_info.phone
            if lead_info.email:
                set_fields["lead_email"] = lead_info.email
            # A name alone is NOT enough — only flag lead_captured once we have
            # a real contact method (phone or email) that lets the team reach out.
            if lead_info.phone or lead_info.email:
                set_fields["lead_captured"] = True

            logger.debug(
                "chat.lead.detected company_id=%s session_id=%s "
                "lead_captured=%s name=%r phone=%r email=%r",
                company_id, session_id,
                bool(lead_info.phone or lead_info.email),
                _mask_value(lead_info.name),
                _mask_value(lead_info.phone),
                _mask_value(lead_info.email),
            )

        # ── 3. Build the messages to append ───────────────────────────────────
        new_messages = [
            {"role": "user",      "content": user_message, "timestamp": now},
            {"role": "assistant", "content": ai_reply,     "timestamp": now},
        ]

        # ── 4. Upsert the chat session document ───────────────────────────────
        result = await db["chat_sessions"].update_one(
            {"company_id": company_id, "session_id": session_id},
            {
                "$push": {
                    "messages":           {"$each": new_messages},
                    "tools_used_history": tools_used,
                },
                "$inc":  {"exchange_count": 1},
                "$set":  set_fields,
                "$setOnInsert": {
                    # IMPORTANT: do NOT duplicate fields that appear in $set.
                    # MongoDB forbids the same field path in both $set and
                    # $setOnInsert — it raises a WriteError and the ENTIRE
                    # write fails. visitor_id is already handled in $set above.
                    "company_id":  company_id,
                    "session_id":  session_id,
                    "created_at":  now,
                    "last_intent": None,
                },
            },
            upsert=True,
        )

        op = "inserted" if result.upserted_id else "updated"
        logger.info(
            "chat.session.%s company_id=%s session_id=%s tools_used=%s",
            op, company_id, session_id, tools_used,
        )

        # ── 5. Upsert into the `leads` collection only when a contact method exists ─
        # A name alone is not actionable — only write to `leads` once we have
        # phone or email so the dashboard doesn't show incomplete lead records.
        if lead_info.has_any and (lead_info.phone or lead_info.email):
            # Merge with session-accumulated lead so a name captured in a prior
            # turn is included even when the current message only has phone/email.
            session = get_session(session_id)
            effective_name  = lead_info.name  or (session.lead_name  if session else None)
            effective_phone = lead_info.phone or (session.lead_phone if session else None)
            effective_email = lead_info.email or (session.lead_email if session else None)

            lead_set: dict = {"updated_at": now}
            if effective_name:
                lead_set["name"]  = effective_name
            if effective_phone:
                lead_set["phone"] = effective_phone
            if effective_email:
                lead_set["email"] = effective_email

            await db["leads"].update_one(
                {"company_id": company_id, "session_id": session_id},
                {
                    "$set": lead_set,
                    "$setOnInsert": {
                        "company_id":   company_id,
                        "session_id":   session_id,
                        "is_contacted": False,
                        "created_at":   now,
                    },
                },
                upsert=True,
            )
            logger.info(
                "chat.lead.upserted company_id=%s session_id=%s",
                company_id, session_id,
            )

    except Exception as exc:
        # Never let a persistence failure break the chat reply
        logger.warning(
            "chat.session.persist_failed company_id=%s session_id=%s error=%s",
            company_id, session_id, exc,
        )


def _slice_current_turn_messages(messages: list, user_message: str) -> list:
    """Return messages from the latest matching human message onward."""
    if not messages:
        return []

    target = (user_message or "").strip()
    idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if getattr(msg, "type", "") != "human":
            continue
        content = str(getattr(msg, "content", "")).strip()
        if target and content == target:
            idx = i
            break
        if not target:
            idx = i
            break

    return messages[idx:] if idx >= 0 else messages


def _mask_value(value: str | None, keep_tail: int = 2) -> str:
    if not value:
        return "none"
    if len(value) <= keep_tail:
        return "*" * len(value)
    return ("*" * (len(value) - keep_tail)) + value[-keep_tail:]


def _build_tool_order_instruction(available_tools: list[str]) -> str:
    """
    Build a per-turn system instruction telling the LLM exactly which tools
    to call and in what order.  Covers both fact-retrieval tools and appointment
    scheduling tools so the agent never skips a mandatory tool call.
    """
    fact_tools = [
        t for t in ("knowledge_base", "web_search", "wikipedia")
        if t in available_tools
    ]
    appt_tools = [
        t for t in ("check_appointment_setup", "get_available_appointment_slots", "get_slot_booking_link")
        if t in available_tools
    ]

    parts: list[str] = []

    if fact_tools:
        steps = " -> ".join(fact_tools)
        parts.append(
            "FACT-RETRIEVAL TOOLS (mandatory for company/legal/factual questions):\n"
            f"Required order: {steps}.\n"
            "- Call knowledge_base FIRST for any company-specific question.\n"
            "- If knowledge_base returns nothing useful, call web_search NEXT.\n"
            "- Only say 'I don't have that verified' after trying ALL available fact tools.\n"
            "- The user's lead-capture status does NOT exempt you from calling tools.\n"
            "- EXCEPTION: Off-topic questions (politics, sports, trivia) — politely refuse."
        )
    else:
        parts.append(
            "No factual retrieval tools available. "
            "Do not invent facts. Ask a focused follow-up question when uncertain."
        )

    if appt_tools:
        parts.append(
            "APPOINTMENT SCHEDULING TOOLS (mandatory when user gives a preferred time):\n"
            "- When the user has chosen to schedule AND given a date/time preference, "
            "you MUST call check_appointment_setup THEN get_available_appointment_slots immediately.\n"
            "- Do NOT say 'I cannot confirm availability' or 'Would you like me to check?' — "
            "just call the tools and show real results.\n"
            "- Pass preferred_time exactly as the user stated (e.g. '3 pm', '14:00', 'morning').\n"
            "- Pass user_timezone from the session context when available.\n"
            "- After user picks a slot and gives name+email, call get_slot_booking_link "
            "with the exact iso_start_time and share the confirmation page URL immediately.\n"
            "- DATE RULE FOR SCHEDULING: When user says 'tomorrow' or a future date for an APPOINTMENT, "
            "treat it as the literal next calendar day. This rule is for SCHEDULING ONLY.\n"
            "  (The 'did you mean yesterday?' date check only applies when asking about the accident/incident date.)"
        )

    return "\n\n".join(parts)


def _tool_outputs_insufficient(tool_outputs: list[str]) -> bool:
    # BUG 2 FIX: Previously, empty strings (returned by knowledge_base when no
    # docs are found) were filtered via `if out`, leaving only the web_search
    # result.  If that web result's body happened to contain a failure-marker
    # substring (e.g. "no longer available" in a quoted news headline), ALL
    # remaining outputs were flagged and the good web_search reply was silently
    # discarded and replaced with the generic fallback string.
    #
    # Correct logic:
    #   - Empty strings = KB found nothing (treat as "no result", not a failure).
    #   - Only flag as insufficient when EVERY non-empty output explicitly
    #     signals a tool-level error, not just a substring coincidence.
    #   - Exclude "missing" and "empty" as standalone markers — too broad.
    if not tool_outputs:
        return False
    non_empty = [out.strip() for out in tool_outputs if out and out.strip()]
    if not non_empty:
        # All outputs were empty (KB returned no docs) — let the reply stand.
        return False
    markers = (
        "no available",
        "no answer",
        "not found",
        "cannot fetch",
        "could not fetch",
        "not configured",
        "no longer available",
    )
    return all(any(marker in out.lower() for marker in markers) for out in non_empty)


def _reply_acknowledges_uncertainty(reply: str) -> bool:
    text = (reply or "").lower()
    markers = (
        "don't have",
        "do not have",
        "not enough",
        "cannot confirm",
        "can't confirm",
        "not sure",
        "uncertain",
    )
    return any(marker in text for marker in markers)


def _is_information_request(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False

    # Skip tiny acknowledgements and pure chit-chat
    tiny = {
        "ok", "okay", "yes", "no", "sure", "hi", "hello", "thanks", "thank you",
        "confirm", "confirmed",
    }
    if text in tiny:
        return False

    keywords = (
        "what", "which", "where", "when", "who", "how",
        "address", "email", "phone", "contact", "office",
        "price", "cost", "fee", "hours", "service", "policy",
        "details", "information", "compensation", "claim", "process",
        "experience", "attorney", "lawyer", "years", "how long",
        "team", "staff", "partner", "founder", "practice",
        "location", "area", "state", "handle", "speciali",
        "win rate", "success", "settlement", "case",
    )
    return any(k in text for k in keywords) or "?" in text


def _is_scheduling_time_response(message: str) -> bool:
    """
    Return True when the user's message looks like a time/date preference
    response to a scheduling question (e.g. '3 pm', 'tomorrow at 10', 'morning').
    Used to trigger a retry that forces the agent to call appointment tools
    instead of answering from memory.
    """
    import re
    text = (message or "").strip().lower()
    if not text or len(text) > 80:  # long messages are unlikely to be bare time responses
        return False

    # Tiny non-scheduling words to skip
    tiny = {"ok", "okay", "yes", "no", "sure", "hi", "hello", "thanks", "thank you"}
    if text in tiny:
        return False

    # Explicit time-of-day markers
    time_keywords = ("am", "pm", "morning", "afternoon", "evening", "noon", "midnight", "o'clock")
    if any(k in text for k in time_keywords):
        return True

    # HH:MM pattern
    if re.search(r"\b\d{1,2}:\d{2}\b", text):
        return True

    # Bare digit that looks like an hour (e.g. "7", "14")
    if re.fullmatch(r"\d{1,2}", text.strip()):
        return True

    # Date/day words in a short message (strongly scheduling context)
    schedule_words = (
        "tomorrow", "today", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday",
        "next week", "this week", "next monday",
    )
    if any(k in text for k in schedule_words):
        return True

    return False


async def _get_session_lead_state(company_id: str, session_id: str) -> dict[str, str | bool | None]:
    """Read lead capture fields for this session, if they exist."""
    db = get_database()
    doc = await db["chat_sessions"].find_one(
        {"company_id": company_id, "session_id": session_id},
        {
            "_id": 0,
            "lead_captured": 1,
            "lead_name": 1,
            "lead_phone": 1,
            "lead_email": 1,
        },
    )
    if not doc:
        return {
            "lead_captured": False,
            "lead_name": None,
            "lead_phone": None,
            "lead_email": None,
        }
    return {
        "lead_captured": bool(doc.get("lead_captured", False)),
        "lead_name": doc.get("lead_name"),
        "lead_phone": doc.get("lead_phone"),
        "lead_email": doc.get("lead_email"),
    }


async def _get_visitor_lead_state(company_id: str, visitor_id: str) -> dict[str, str | bool | None] | None:
    """Return lead info from the visitor's most recent session that captured a contact.
    Used to avoid re-asking returning visitors for their name / phone / email.
    """
    db = get_database()
    doc = await db["chat_sessions"].find_one(
        {"company_id": company_id, "visitor_id": visitor_id, "lead_captured": True},
        {
            "_id": 0,
            "lead_captured": 1,
            "lead_name": 1,
            "lead_phone": 1,
            "lead_email": 1,
        },
        sort=[("updated_at", -1)],
    )
    if not doc:
        return None
    return {
        "lead_captured": bool(doc.get("lead_captured", False)),
        "lead_name":     doc.get("lead_name"),
        "lead_phone":    doc.get("lead_phone"),
        "lead_email":    doc.get("lead_email"),
    }


def _is_callback_timing_question(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False
    if "when" in text and ("call" in text or "callback" in text):
        return True
    if "how soon" in text and ("call" in text or "callback" in text):
        return True
    timing_phrases = (
        "when call your attorney",
        "when will attorney call",
        "when your team call",
        "when will you call me",
        "callback time",
    )
    return any(phrase in text for phrase in timing_phrases)


def _build_callback_timing_reply(lead_name: str | None, lead_phone: str | None) -> str:
    if lead_phone:
        name_part = f", {lead_name}" if lead_name else ""
        return (
            f"Thanks{name_part}. I already have your contact details. "
            "Our intake team will reach out to you shortly to review your case. "
            "Is there anything else you'd like to share about your situation before they call?"
        )
    return (
        "I can arrange that now. Please share your best phone number and preferred call time window, "
        "and our intake team will reach out."
    )


def _build_post_lead_capture_reply_if_needed(
    message: str,
    session_lead_captured: bool,
    message_lead: LeadInfo,
    effective_lead_name: str | None,
    effective_lead_phone: str | None,
    effective_lead_email: str | None,
) -> str | None:
    """
    Deterministic guard: if user shares PHONE contact details, do not let the
    LLM restart the intake flow.

    IMPORTANT — Email is intentionally excluded from triggering this short-circuit.
    Reason: an email address is legitimately required mid-flow for Calendly
    appointment confirmation (the agent asks for name + email to book a slot).
    If we short-circuit on email, we bypass the agent before it can call
    `get_slot_booking_link` and the appointment confirmation URL is never sent.
    Phone numbers are unambiguously lead-capture, so we still guard on those.
    """
    if session_lead_captured:
        return None
    if not getattr(message_lead, "has_any", False):
        return None
    if not _looks_like_contact_share_message(message):
        return None

    # Only short-circuit when the user shares a phone number.
    # Email alone → let the agent decide (could be appointment booking).
    if effective_lead_phone:
        name_part = f", {effective_lead_name}" if effective_lead_name else ""
        return (
            f"Thanks{name_part}. I have your details and our intake team will call you shortly for a free review. "
            "Is there anything else you'd like to share about your situation before we connect?"
        )

    # Email only — do NOT short-circuit. Return None so the agent handles it.
    # This allows the agent to proceed with `get_slot_booking_link` when the
    # user is in the middle of scheduling an appointment.
    if effective_lead_email and not effective_lead_phone:
        return None

    # Name only (no phone, no email) — gently ask for a contact method.
    return (
        "Thanks. I have your name. Can I get your best phone number so our intake team can call you "
        "for your free case review?"
    )


# ── Public widget endpoint ────────────────────────────────────────────────────

class WidgetChatRequest(BaseModel):
    message: str
    history: list = []


class WidgetChatResponse(BaseModel):
    answer: str
    session_id: str
    company_id: str


@widget_router.post("/ask", response_model=WidgetChatResponse, summary="Widget public chat")
async def widget_ask(
    payload: WidgetChatRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_visitor_id: str | None = Header(default=None, alias="X-Visitor-ID"),
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
):
    """
    Public endpoint used by the embedded widget script.
    - `X-API-Key`    — format: org-<company_id>
    - `X-Session-ID` — visitor session string (used for conversation memory)
    """
    if not x_api_key:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="X-API-Key header is required.")

    if x_api_key.startswith("org-"):
        company_id = x_api_key[4:]
    else:
        company_id = x_api_key

    if not ObjectId.is_valid(company_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid API key format.")

    session_id = (x_session_id or "default").strip() or "default"

    # Check human takeover — if active, store user message silently and return empty answer
    db = get_database()
    session_doc = await db["chat_sessions"].find_one(
        {"company_id": company_id, "session_id": session_id},
        {"human_takeover": 1},
    )
    if session_doc and session_doc.get("human_takeover"):
        now = datetime.now(timezone.utc)
        await db["chat_sessions"].update_one(
            {"company_id": company_id, "session_id": session_id},
            {
                "$push": {"messages": {"role": "user", "content": payload.message, "timestamp": now}},
                "$set": {"updated_at": now},
            },
        )
        await ws_manager.notify_dashboard(company_id, {
            "type": "new_message",
            "session_id": session_id,
            "content": payload.message,
            "timestamp": now.isoformat(),
        })
        return WidgetChatResponse(answer="", session_id=session_id, company_id=company_id)

    await ws_manager.notify_dashboard(company_id, {
        "type": "new_message",
        "session_id": session_id,
        "content": payload.message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    await ws_manager.notify_dashboard(company_id, {
        "type": "typing_start",
        "session_id": session_id,
    })

    chat_request = ChatRequest(
        session_id=session_id,
        visitor_id=(x_visitor_id or "").strip() or None,
        message=payload.message,
    )
    chat_response = await chat(company_id, chat_request)

    await ws_manager.notify_dashboard(company_id, {
        "type": "ai_reply",
        "session_id": session_id,
        "content": chat_response.reply,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return WidgetChatResponse(
        answer=chat_response.reply,
        session_id=chat_response.session_id,
        company_id=chat_response.company_id,
    )


@widget_router.get("/history", summary="Widget session history")
async def widget_history(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
):
    """
    Return all messages for the current widget session so the widget
    can restore conversation history on page reload.
    - `X-API-Key`    — format: org-<company_id>
    - `X-Session-ID` — session string
    """
    if not x_api_key:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="X-API-Key header is required.")

    company_id = x_api_key[4:] if x_api_key.startswith("org-") else x_api_key
    if not ObjectId.is_valid(company_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid API key format.")

    session_id = (x_session_id or "").strip()
    if not session_id:
        return {"messages": []}

    db = get_database()
    doc = await db["chat_sessions"].find_one(
        {"company_id": company_id, "session_id": session_id},
        {"_id": 0, "messages": 1},
    )
    if not doc:
        return {"messages": []}

    messages = [
        {"role": msg.get("role", ""), "text": msg.get("content", "")}
        for msg in doc.get("messages", [])
        if isinstance(msg, dict) and msg.get("role") in ("user", "assistant")
    ]
    return {"messages": messages}


def _looks_like_contact_share_message(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False

    # Keyword-based detection (e.g. "my name is...", "my phone is...", "email me at...")
    indicators = (
        "my name",
        "name is",
        "phone",
        "number",
        "call me",
        "email",
        "@",
    )
    if any(token in text for token in indicators):
        return True

    # Raw phone number detection: message is mostly digits (e.g. "01792843207", "+1 555 1234")
    # Strip common phone separators and count digits vs total length.
    stripped = text.replace(" ", "").replace("-", "").replace("+", "").replace("(", "").replace(")", "").replace(".", "")
    if stripped and len(stripped) >= 6:
        digit_ratio = sum(c.isdigit() for c in stripped) / len(stripped)
        if digit_ratio >= 0.8:
            return True

    return False
