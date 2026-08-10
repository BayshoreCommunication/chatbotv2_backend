from typing import Optional
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

UNKNOWN_COMPANY = {"company_name": "Unknown company", "email": ""}


async def _company_lookup(db: AsyncIOMotorDatabase, company_ids: set) -> dict:
    """
    Maps company_id (string) -> {"company_name", "email"} via the matching
    `users` documents. chat_sessions.company_id is stored as a plain string
    (the JWT `sub` used at chat time), while users._id is an ObjectId — cast
    to join the two.
    """
    object_ids = [ObjectId(cid) for cid in company_ids if cid and ObjectId.is_valid(cid)]
    if not object_ids:
        return {}
    cursor = db["users"].find(
        {"_id": {"$in": object_ids}},
        {"company_name": 1, "email": 1},
    )
    return {
        str(doc["_id"]): {
            "company_name": doc.get("company_name") or UNKNOWN_COMPANY["company_name"],
            "email": doc.get("email", ""),
        }
        async for doc in cursor
    }


def _last_message_preview(messages: list) -> Optional[str]:
    for msg in reversed(messages):
        content = (msg or {}).get("content")
        if content:
            return content[:140]
    return None


async def get_all_chat_sessions(
    db: AsyncIOMotorDatabase,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """Cross-company chat session list for the admin panel, newest first —
    each row enriched with which company/organization it belongs to so the
    admin can identify and group conversations by organization."""
    total = await db["chat_sessions"].count_documents({})
    docs = await (
        db["chat_sessions"]
        .find({})
        .sort("updated_at", -1)
        .skip(offset)
        .limit(limit)
        .to_list(length=limit)
    )

    companies = await _company_lookup(db, {doc.get("company_id", "") for doc in docs})

    sessions = []
    for doc in docs:
        company_id = doc.get("company_id", "")
        company = companies.get(company_id, UNKNOWN_COMPANY)
        sessions.append({
            "session_id": doc.get("session_id", ""),
            "company_id": company_id,
            "company_name": company["company_name"],
            "company_email": company["email"],
            "exchange_count": doc.get("exchange_count", 0),
            "lead_captured": doc.get("lead_captured", False),
            "lead_name": doc.get("lead_name"),
            "human_takeover": doc.get("human_takeover", False),
            "last_message": _last_message_preview(doc.get("messages", [])),
            "created_at": doc.get("created_at"),
            "updated_at": doc.get("updated_at"),
        })

    return {"sessions": sessions, "total": total, "limit": limit, "offset": offset}


async def get_chat_session_detail(
    db: AsyncIOMotorDatabase,
    company_id: str,
    session_id: str,
) -> Optional[dict]:
    """Full transcript for one session, enriched with company info."""
    doc = await db["chat_sessions"].find_one(
        {"company_id": company_id, "session_id": session_id},
    )
    if not doc:
        return None

    companies = await _company_lookup(db, {company_id})
    company = companies.get(company_id, UNKNOWN_COMPANY)

    messages = [
        {
            "role": str(msg.get("role", "")),
            "content": str(msg.get("content", "")),
            "timestamp": msg.get("timestamp"),
        }
        for msg in doc.get("messages", [])
        if isinstance(msg, dict)
    ]

    return {
        "session_id": doc.get("session_id", ""),
        "company_id": company_id,
        "company_name": company["company_name"],
        "company_email": company["email"],
        "visitor_id": doc.get("visitor_id"),
        "exchange_count": doc.get("exchange_count", 0),
        "lead_captured": doc.get("lead_captured", False),
        "lead_name": doc.get("lead_name"),
        "lead_phone": doc.get("lead_phone"),
        "lead_email": doc.get("lead_email"),
        "human_takeover": doc.get("human_takeover", False),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "messages": messages,
    }


def _visitor_key(doc: dict) -> str:
    """visitor_id when the widget sent X-Visitor-ID, else fall back to this
    session's own session_id — so older/untracked sessions still show up as
    a (single-session) visitor instead of disappearing from the list."""
    return doc.get("visitor_id") or doc.get("session_id", "")


async def get_company_visitors(db: AsyncIOMotorDatabase, company_id: str) -> dict:
    """Distinct visitors for one company, aggregated from their sessions —
    most recently active first."""
    docs = await (
        db["chat_sessions"]
        .find({"company_id": company_id})
        .sort("updated_at", -1)
        .to_list(length=None)
    )

    visitors: dict = {}
    for doc in docs:
        key = _visitor_key(doc)
        v = visitors.setdefault(key, {
            "visitor_id": key,
            "session_count": 0,
            "total_exchanges": 0,
            "lead_captured": False,
            "lead_name": None,
            "lead_email": None,
            "lead_phone": None,
            "last_message": None,
            "first_seen": None,
            "last_seen": None,
        })

        v["session_count"] += 1
        v["total_exchanges"] += doc.get("exchange_count", 0)
        if doc.get("lead_captured"):
            v["lead_captured"] = True
        v["lead_name"] = v["lead_name"] or doc.get("lead_name")
        v["lead_email"] = v["lead_email"] or doc.get("lead_email")
        v["lead_phone"] = v["lead_phone"] or doc.get("lead_phone")

        updated_at = doc.get("updated_at")
        if updated_at and (v["last_seen"] is None or updated_at > v["last_seen"]):
            v["last_seen"] = updated_at
            v["last_message"] = _last_message_preview(doc.get("messages", []))

        created_at = doc.get("created_at")
        if created_at and (v["first_seen"] is None or created_at < v["first_seen"]):
            v["first_seen"] = created_at

    companies = await _company_lookup(db, {company_id})
    company = companies.get(company_id, UNKNOWN_COMPANY)

    visitor_list = sorted(
        visitors.values(), key=lambda v: v["last_seen"] or "", reverse=True,
    )

    return {
        "company_id": company_id,
        "company_name": company["company_name"],
        "company_email": company["email"],
        "visitors": visitor_list,
        "total": len(visitor_list),
    }


async def get_visitor_sessions(
    db: AsyncIOMotorDatabase,
    company_id: str,
    visitor_id: str,
) -> dict:
    """All chat sessions for one visitor within a company — matches by
    visitor_id, or by session_id itself as a fallback, mirroring the same
    fallback key used to build the visitor list in the first place."""
    query = {
        "company_id": company_id,
        "$or": [{"visitor_id": visitor_id}, {"session_id": visitor_id}],
    }
    docs = await db["chat_sessions"].find(query).sort("updated_at", -1).to_list(length=None)

    companies = await _company_lookup(db, {company_id})
    company = companies.get(company_id, UNKNOWN_COMPANY)

    sessions = [
        {
            "session_id": doc.get("session_id", ""),
            "company_id": company_id,
            "company_name": company["company_name"],
            "company_email": company["email"],
            "exchange_count": doc.get("exchange_count", 0),
            "lead_captured": doc.get("lead_captured", False),
            "lead_name": doc.get("lead_name"),
            "human_takeover": doc.get("human_takeover", False),
            "last_message": _last_message_preview(doc.get("messages", [])),
            "created_at": doc.get("created_at"),
            "updated_at": doc.get("updated_at"),
        }
        for doc in docs
    ]

    return {"sessions": sessions, "total": len(sessions), "limit": len(sessions), "offset": 0}


async def get_visitor_chat_history(
    db: AsyncIOMotorDatabase,
    company_id: str,
    visitor_id: str,
) -> Optional[dict]:
    """One visitor's full chat history within a company — every message from
    every one of their sessions, flattened into a single chronological
    thread (oldest first), for the admin's chat viewer center panel."""
    query = {
        "company_id": company_id,
        "$or": [{"visitor_id": visitor_id}, {"session_id": visitor_id}],
    }
    docs = await db["chat_sessions"].find(query).sort("created_at", 1).to_list(length=None)
    if not docs:
        return None

    companies = await _company_lookup(db, {company_id})
    company = companies.get(company_id, UNKNOWN_COMPANY)

    messages = []
    lead_name = lead_email = lead_phone = None
    lead_captured = False
    total_exchanges = 0
    for doc in docs:
        total_exchanges += doc.get("exchange_count", 0)
        if doc.get("lead_captured"):
            lead_captured = True
        lead_name = lead_name or doc.get("lead_name")
        lead_email = lead_email or doc.get("lead_email")
        lead_phone = lead_phone or doc.get("lead_phone")
        for msg in doc.get("messages", []):
            if isinstance(msg, dict):
                messages.append({
                    "role": str(msg.get("role", "")),
                    "content": str(msg.get("content", "")),
                    "timestamp": msg.get("timestamp"),
                })

    return {
        "visitor_id": visitor_id,
        "company_id": company_id,
        "company_name": company["company_name"],
        "company_email": company["email"],
        "lead_name": lead_name,
        "lead_email": lead_email,
        "lead_phone": lead_phone,
        "lead_captured": lead_captured,
        "session_count": len(docs),
        "total_exchanges": total_exchanges,
        "messages": messages,
    }
