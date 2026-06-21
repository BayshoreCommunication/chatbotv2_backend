import logging
from datetime import datetime
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
from pymongo import ReturnDocument

from services.chatbot.llm import llm

logger = logging.getLogger(__name__)

CONVERSATION_THRESHOLD = 10


def serialize_lead(lead: dict) -> dict:
    """Convert MongoDB lead document to JSON-serializable dict."""
    return {
        "id": str(lead["_id"]),
        "company_id": lead.get("company_id"),
        "session_id": lead.get("session_id"),
        "name": lead.get("name"),
        "email": lead.get("email"),
        "phone": lead.get("phone"),
        "message": lead.get("message"),
        "is_contacted": lead.get("is_contacted", False),
        "created_at": lead.get("created_at"),
        "updated_at": lead.get("updated_at"),
    }

async def get_leads_by_company(db: AsyncIOMotorDatabase, company_id: str) -> List[dict]:
    """Fetch all leads for a specific company, sorted by newest first."""
    cursor = db["leads"].find({"company_id": company_id}).sort("created_at", -1)
    return [serialize_lead(lead) async for lead in cursor]

async def delete_lead(db: AsyncIOMotorDatabase, lead_id: str, company_id: str) -> bool:
    """Delete a specific lead belonging to a company."""
    if not ObjectId.is_valid(lead_id):
        return False
    result = await db["leads"].delete_one({"_id": ObjectId(lead_id), "company_id": company_id})
    return result.deleted_count > 0

async def set_lead_contacted(
    db: AsyncIOMotorDatabase,
    lead_id: str,
    company_id: str,
    is_contacted: bool,
) -> Optional[dict]:
    """Toggle the `is_contacted` flag on a lead belonging to this company."""
    if not ObjectId.is_valid(lead_id):
        return None

    result = await db["leads"].find_one_and_update(
        {"_id": ObjectId(lead_id), "company_id": company_id},
        {"$set": {"is_contacted": is_contacted, "updated_at": datetime.utcnow()}},
        return_document=ReturnDocument.AFTER,
    )
    if result:
        logger.info(
            "leads.api.contacted_updated company_id=%s lead_id=%s is_contacted=%s",
            company_id, lead_id, is_contacted,
        )
    return serialize_lead(result) if result else None


async def _summarize_conversation(messages: List[str]) -> str:
    """Ask the LLM for a short summary of what problem/service the user discussed."""
    joined = "\n".join(f"- {m}" for m in messages)
    prompt = (
        "Here is a conversation between a chatbot and a visitor. "
        "Write ONE short sentence (max 20 words) summarizing what problem "
        "or service the visitor needs.\n\n"
        f"{joined}"
    )
    try:
        result = await llm.ainvoke(prompt)
        return result.content.strip()
    except Exception as exc:
        logger.error("lead_service.summary_failed error=%s", exc)
        return "Summary unavailable"


async def maybe_generate_lead_summaries(db: AsyncIOMotorDatabase, company_id: str) -> None:
    """
    Call this when leads are fetched (GET /leads).

    For every lead belonging to this company:
      1. Check the lead's `message` key — skip if it already has a value.
      2. If `message` is None, check whether phone or email was collected.
      3. If a contact method exists, look up the linked ChatSessionModel
         (by company_id + session_id) and check how many entries are in
         its `messages` array.
      4. If that count is >= CONVERSATION_THRESHOLD (10), summarize the
         conversation via AI and save the short summary into the lead's
         `message` key.
    """
    leads_cursor = db["leads"].find({"company_id": company_id})
    leads = [doc async for doc in leads_cursor]

    logger.info(
        "leads.api.called company_id=%s total_leads=%s",
        company_id, len(leads),
    )

    summaries_generated = 0

    for lead in leads:
        lead_id = lead["_id"]
        message = lead.get("message")

        logger.info(
            "leads.api.check_message company_id=%s lead_id=%s message=%s",
            company_id, lead_id, message if message is not None else "None",
        )

        if message is not None:
            continue  # already has a message — nothing to do

        has_contact = bool(lead.get("phone") or lead.get("email"))
        logger.info(
            "leads.api.check_contact company_id=%s lead_id=%s collected=%s",
            company_id, lead_id, "yes" if has_contact else "no",
        )

        if not has_contact:
            continue

        session_id = lead.get("session_id")
        chat_session = await db["chat_sessions"].find_one(
            {"company_id": company_id, "session_id": session_id},
            {"messages": 1},
        )
        chat_messages = (chat_session or {}).get("messages", [])
        message_count = len(chat_messages)

        logger.info(
            "leads.api.check_chat_index company_id=%s lead_id=%s session_id=%s message_count=%s",
            company_id, lead_id, session_id, message_count,
        )

        if message_count < CONVERSATION_THRESHOLD:
            logger.info(
                "leads.api.threshold_not_reached company_id=%s lead_id=%s message_count=%s threshold=%s",
                company_id, lead_id, message_count, CONVERSATION_THRESHOLD,
            )
            continue

        texts = [m["content"] for m in chat_messages if m.get("role") == "user" and m.get("content")]
        if not texts:
            logger.info(
                "leads.api.no_user_texts company_id=%s lead_id=%s session_id=%s",
                company_id, lead_id, session_id,
            )
            continue

        logger.info(
            "leads.api.summarizing company_id=%s lead_id=%s session_id=%s user_message_count=%s",
            company_id, lead_id, session_id, len(texts),
        )

        summary = await _summarize_conversation(texts)

        await db["leads"].update_one(
            {"_id": lead_id},
            {"$set": {"message": summary}},
        )
        summaries_generated += 1
        logger.info(
            "leads.api.summary_saved company_id=%s lead_id=%s summary=%s",
            company_id, lead_id, summary,
        )

    logger.info(
        "leads.api.done company_id=%s total_leads=%s summaries_generated=%s",
        company_id, len(leads), summaries_generated,
    )
