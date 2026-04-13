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

from fastapi import APIRouter, HTTPException, Query, status
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

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger(__name__)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = "default"
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

    # ── Load lead state: session cache first, MongoDB fallback on first message ──
    thread_id = f"{company_id}:{payload.session_id}"
    session = get_session(thread_id)
    if session is None:
        # First message (or session expired) — load from MongoDB then cache
        db_lead = await _get_session_lead_state(company_id, payload.session_id)
        session = await create_or_refresh_session(
            thread_id=thread_id,
            company_id=company_id,
            company_ctx=ctx,
            lead_state=db_lead,
            user_timezone=(payload.user_timezone or "").strip() or None,
        )
    else:
        # Session alive — refresh timezone if newly provided
        if payload.user_timezone and not session.user_timezone:
            session.user_timezone = (payload.user_timezone or "").strip() or None

    session_lead = session.to_lead_dict()
    message_lead = await extract_lead_info_async(payload.message)
    effective_lead_name = message_lead.name or session_lead["lead_name"]
    effective_lead_phone = message_lead.phone or session_lead["lead_phone"]
    effective_lead_email = message_lead.email or session_lead["lead_email"]
    # A name alone is NOT enough — we need at least a phone or email to reach the user.
    # Only suppress the intake flow once we have a real contact method.
    lead_captured_effective = bool(
        session_lead["lead_captured"]
        or effective_lead_phone
        or effective_lead_email
    )
    short_circuit_lead_reply = _build_post_lead_capture_reply_if_needed(
        message=payload.message,
        session_lead_captured=bool(session_lead["lead_captured"]),
        message_lead=message_lead,
        effective_lead_name=effective_lead_name,
        effective_lead_phone=effective_lead_phone,
        effective_lead_email=effective_lead_email,
    )

    if short_circuit_lead_reply:
        reply = short_circuit_lead_reply
        tools_used: list[str] = []
        elapsed_total = int((time.monotonic() - t_start) * 1000)
        logger.info(
            "chat.response.short_circuit company_id=%s session_id=%s reason=lead_capture "
            "reply_len=%d total_ms=%d",
            company_id, payload.session_id, len(reply), elapsed_total,
        )
        # Update session cache immediately so next message sees the lead info
        update_session_lead(
            thread_id=thread_id,
            name=effective_lead_name,
            phone=effective_lead_phone,
            email=effective_lead_email,
        )
        await _persist_exchange(
            company_id=company_id,
            session_id=payload.session_id,
            user_message=payload.message,
            ai_reply=reply,
            tools_used=tools_used,
            lead_info=message_lead,
        )
        return ChatResponse(
            reply=reply,
            session_id=payload.session_id,
            company_id=company_id,
            tools_used=tools_used,
        )

    # ── Invoke agent — LangGraph uses thread_id for per-session memory ────────
    # thread_id is already set above (session cache lookup)
    config = {"configurable": {"thread_id": thread_id}}
    register_thread_activity(thread_id)
    available_tools = get_cached_tool_names(company_id)

    logger.info(
        "chat.agent.invoke company_id=%s session_id=%s thread_id=%s",
        company_id, payload.session_id, thread_id,
    )

    t_invoke = time.monotonic()
    try:
        input_messages: list[tuple[str, str]] = []
        if lead_captured_effective:
            # We have a real contact method (phone or email) — suppress re-asking.
            input_messages.append(
                (
                    "system",
                    (
                        "Session lead state: contact details are already captured for this conversation. "
                        f"Known name={effective_lead_name or 'unknown'}, "
                        f"phone={effective_lead_phone or 'missing'}, "
                        f"email={effective_lead_email or 'missing'}. "
                        "STRICT RULES — follow all of these:\n"
                        "1. Do NOT restart the contact intake flow. Do NOT ask for name, phone, or email again.\n"
                        "2. Do NOT present the 'phone call / email / schedule now' options again — that decision was already made.\n"
                        "3. If the user says 'yes' or agrees to 'connect with an attorney' or 'speak with the team' "
                        "at any point after the lead is captured, do NOT ask 'Would you prefer a phone call or to schedule?'. "
                        "Instead say: 'I already have your details — I will make sure our team reaches out to you shortly. "
                        "Is there anything else I can help you with while you wait?'\n"
                        "4. If user asks when the attorney/team will call: state simply that the intake team will reach out shortly to assist them.\n"
                        "   Do NOT state any specific timeframes like 15-30 minutes or next business day.\n"
                        "5. If the user replies 'yes' to a question like 'Is there anything about X you would like to know?', "
                        "do NOT ask 'What specific questions do you have about X?' — answer X directly and helpfully right away.\n"
                        "6. Always end with one specific follow-up question to keep the conversation going."
                    ),
                )
            )
        elif effective_lead_name and not effective_lead_phone and not effective_lead_email:
            # We have the user's name but no contact method yet — remind the agent
            # to acknowledge the name and ask specifically for phone or email.
            input_messages.append(
                (
                    "system",
                    (
                        f"The user has shared their name: {effective_lead_name}. "
                        "You do NOT yet have a phone number or email address for this person. "
                        "STRICT RULE: Acknowledge their name warmly, then ask specifically for "
                        "their best phone number (or email) so the team can follow up. "
                        "Do NOT say 'I already have your details' — you only have a name, not a contact method."
                    ),
                )
            )
        input_messages.append(("system", _build_tool_order_instruction(available_tools)))
        user_timezone = (payload.user_timezone or "").strip()
        if user_timezone:
            input_messages.append(
                (
                    "system",
                    (
                        f"User timezone is {user_timezone}. This timezone is auto-detected. "
                        "Do not ask for timezone unless the user explicitly asks to change it. "
                        "Use this timezone by default for scheduling and slot suggestions."
                    ),
                )
            )
        input_messages.append(("user", payload.message))

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
    tool_outputs: list[str] = []
    for msg in turn_messages:
        if getattr(msg, "type", "") == "tool":
            name = getattr(msg, "name", "")
            if name and name not in tools_used:
                tools_used.append(name)
            tool_outputs.append(str(getattr(msg, "content", "")))

    has_fact_tools = any(
        tool_name in available_tools for tool_name in ("knowledge_base", "web_search", "wikipedia")
    )
    if _is_information_request(payload.message) and not tools_used and has_fact_tools:
        logger.info(
            "chat.agent.retry_tool_enforced company_id=%s session_id=%s reason=no_tools_used_first_pass",
            company_id,
            payload.session_id,
        )
        # Keep same thread so retry has full conversation memory.
        retry_config = {"configurable": {"thread_id": thread_id}}
        retry_messages = [
            (
                "system",
                _build_tool_order_instruction(available_tools),
            ),
            ("user", payload.message),
        ]
        try:
            retry_result = await agent.ainvoke(
                {"messages": retry_messages},
                config=retry_config,
            )
            retry_all = retry_result.get("messages", [])
            retry_turn = _slice_current_turn_messages(retry_all, payload.message)

            retry_reply = ""
            for msg in reversed(retry_turn):
                if hasattr(msg, "content") and getattr(msg, "type", "") == "ai":
                    retry_reply = msg.content
                    break
            if not retry_reply and retry_turn:
                retry_reply = str(retry_turn[-1].content)

            retry_tools_used: list[str] = []
            retry_tool_outputs: list[str] = []
            for msg in retry_turn:
                if getattr(msg, "type", "") == "tool":
                    name = getattr(msg, "name", "")
                    if name and name not in retry_tools_used:
                        retry_tools_used.append(name)
                    retry_tool_outputs.append(str(getattr(msg, "content", "")))

            if retry_reply:
                reply = retry_reply
            if retry_tools_used:
                tools_used = retry_tools_used
            if retry_tool_outputs:
                tool_outputs = retry_tool_outputs
        except Exception as exc:
            logger.warning(
                "chat.agent.retry_tool_enforced_failed company_id=%s session_id=%s error=%s",
                company_id,
                payload.session_id,
                exc,
            )

    if _tool_outputs_insufficient(tool_outputs) and not _reply_acknowledges_uncertainty(reply):
        reply = (
            "I checked my sources but I still don't have enough verified information to answer that confidently right now. "
            "If you want, I can take your details and have our team follow up with a precise answer."
        )

    if _is_callback_timing_question(payload.message):
        reply = _build_callback_timing_reply(
            lead_name=effective_lead_name,
            lead_phone=effective_lead_phone,
        )

    elapsed_total = int((time.monotonic() - t_start) * 1000)
    logger.info(
        "chat.response.out company_id=%s session_id=%s reply_len=%d "
        "tools_used=%s invoke_ms=%d total_ms=%d",
        company_id, payload.session_id, len(reply),
        tools_used, elapsed_invoke, elapsed_total,
    )

    # ── Update session cache immediately (before persist) ────────────────────
    # This ensures the NEXT message sees lead info even if _persist_exchange
    # hasn't finished yet — fixing the race condition that caused re-asking.
    if message_lead.has_any:
        update_session_lead(
            thread_id=thread_id,
            name=effective_lead_name,
            phone=effective_lead_phone,
            email=effective_lead_email,
        )

    # ── Persist exchange to MongoDB chat_sessions ─────────────────────────────
    await _persist_exchange(
        company_id=company_id,
        session_id=payload.session_id,
        user_message=payload.message,
        ai_reply=reply,
        tools_used=tools_used,
        lead_info=message_lead,
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

        if lead_info.has_any:
            set_fields["lead_captured"] = True
            if lead_info.name:
                set_fields["lead_name"]  = lead_info.name
            if lead_info.phone:
                set_fields["lead_phone"] = lead_info.phone
            if lead_info.email:
                set_fields["lead_email"] = lead_info.email

            logger.debug(
                "chat.lead.detected company_id=%s session_id=%s "
                "name=%r phone=%r email=%r",
                company_id, session_id,
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
                    # IMPORTANT: do NOT put lead_captured here.
                    # MongoDB forbids the same field path in both $set and
                    # $setOnInsert. When lead info arrives on the very first
                    # message of a new session, $set sets lead_captured=True
                    # while $setOnInsert had lead_captured=False — MongoDB
                    # raises a silent WriteError and the ENTIRE write fails.
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

        # ── 5. Upsert into the `leads` collection if contact info was found ───
        if lead_info.has_any:
            lead_set: dict = {"updated_at": now}
            if lead_info.name:
                lead_set["name"]  = lead_info.name
            if lead_info.phone:
                lead_set["phone"] = lead_info.phone
            if lead_info.email:
                lead_set["email"] = lead_info.email

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
    ordered_tools = [
        tool_name
        for tool_name in ("knowledge_base", "web_search", "wikipedia")
        if tool_name in available_tools
    ]
    if not ordered_tools:
        return (
            "No factual verification tools are currently available. "
            "Do not invent facts, and clearly ask a focused follow-up question when uncertain."
        )

    steps = " -> ".join(ordered_tools)
    return (
        "You must verify factual/company answers with available tools. "
        f"Required order (only tools currently available): {steps}. "
        "If one source is insufficient, continue to the next available source. "
        "If all are insufficient, explicitly say information is not verified and do not guess. "
        "CRITICAL EXCEPTION: If the question is completely OFF-TOPIC (e.g., politics, sports, trivia), "
        "do not use tools and politely refuse."
    )


def _tool_outputs_insufficient(tool_outputs: list[str]) -> bool:
    if not tool_outputs:
        return False
    markers = (
        "no available",
        "no answer",
        "not found",
        "cannot fetch",
        "could not fetch",
        "missing",
        "not configured",
        "no longer available",
        "empty",
    )
    lowered = [out.lower() for out in tool_outputs if out]
    if not lowered:
        return False
    return all(any(marker in out for marker in markers) for out in lowered)


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
    )
    return any(k in text for k in keywords) or "?" in text


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
    Deterministic guard: if user shares contact details, do not let the LLM restart intake.
    """
    if session_lead_captured:
        return None
    if not getattr(message_lead, "has_any", False):
        return None
    if not _looks_like_contact_share_message(message):
        return None

    if effective_lead_phone:
        name_part = f", {effective_lead_name}" if effective_lead_name else ""
        return (
            f"Thanks{name_part}. I have your details and our intake team will call you shortly for a free review. "
            "Is there anything else you'd like to share about your situation before we connect?"
        )

    if effective_lead_email:
        name_part = f", {effective_lead_name}" if effective_lead_name else ""
        return (
            f"Thanks{name_part}. I have your details and our intake team will follow up for your free review by email. "
            "Would you like a callback instead, or should we continue by email?"
        )

    return (
        "Thanks. I have your name. Please share your best phone number so our intake team can call you "
        "for your free case review."
    )


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
