import logging
import re
from datetime import datetime
from typing import Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

logger = logging.getLogger(__name__)

UNKNOWN_COMPANY = {"company_name": "Unknown company", "email": ""}


async def _company_lookup(db: AsyncIOMotorDatabase, company_ids: set) -> dict:
    """
    Maps company_id (string) -> {"company_name", "email"} via the matching
    `users` documents — same join used by admin_chat_service, since leads
    also store company_id as a plain string while users._id is an ObjectId.
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


def _serialize(lead: dict, company: dict) -> dict:
    return {
        "id": str(lead["_id"]),
        "company_id": lead.get("company_id"),
        "company_name": company["company_name"],
        "company_email": company["email"],
        "session_id": lead.get("session_id"),
        "name": lead.get("name"),
        "email": lead.get("email"),
        "phone": lead.get("phone"),
        "message": lead.get("message"),
        "is_contacted": lead.get("is_contacted", False),
        "appointment_time": lead.get("appointment_time"),
        "created_at": lead.get("created_at"),
        "updated_at": lead.get("updated_at"),
    }


async def get_all_leads(db: AsyncIOMotorDatabase, limit: int = 20, offset: int = 0) -> dict:
    """Cross-company leads list for the admin panel, newest first — each row
    enriched with which company/organization captured it."""
    total = await db["leads"].count_documents({})
    docs = await (
        db["leads"]
        .find({})
        .sort("created_at", -1)
        .skip(offset)
        .limit(limit)
        .to_list(length=limit)
    )

    companies = await _company_lookup(db, {doc.get("company_id", "") for doc in docs})

    leads = [
        _serialize(doc, companies.get(doc.get("company_id", ""), UNKNOWN_COMPANY))
        for doc in docs
    ]

    return {"leads": leads, "total": total, "limit": limit, "offset": offset}


async def search_leads(db: AsyncIOMotorDatabase, query: str, limit: int = 10) -> dict:
    """Search leads across every company by name, email, or phone — for the
    admin topbar's global search bar."""
    query = query.strip()
    if not query:
        return {"leads": [], "total": 0, "limit": limit, "offset": 0}

    pattern = re.escape(query)
    docs = await (
        db["leads"]
        .find({
            "$or": [
                {"name":  {"$regex": pattern, "$options": "i"}},
                {"email": {"$regex": pattern, "$options": "i"}},
                {"phone": {"$regex": pattern, "$options": "i"}},
            ],
        })
        .sort("created_at", -1)
        .limit(limit)
        .to_list(length=limit)
    )

    companies = await _company_lookup(db, {doc.get("company_id", "") for doc in docs})
    leads = [
        _serialize(doc, companies.get(doc.get("company_id", ""), UNKNOWN_COMPANY))
        for doc in docs
    ]
    return {"leads": leads, "total": len(leads), "limit": limit, "offset": 0}


async def set_lead_contacted(
    db: AsyncIOMotorDatabase, lead_id: str, is_contacted: bool,
) -> Optional[dict]:
    """Toggle the `is_contacted` flag on any lead, regardless of company."""
    if not ObjectId.is_valid(lead_id):
        return None

    result = await db["leads"].find_one_and_update(
        {"_id": ObjectId(lead_id)},
        {"$set": {"is_contacted": is_contacted, "updated_at": datetime.utcnow()}},
        return_document=ReturnDocument.AFTER,
    )
    if not result:
        return None

    companies = await _company_lookup(db, {result.get("company_id", "")})
    company = companies.get(result.get("company_id", ""), UNKNOWN_COMPANY)
    logger.info(
        "admin.leads.contacted_updated lead_id=%s is_contacted=%s", lead_id, is_contacted,
    )
    return _serialize(result, company)


async def delete_lead(db: AsyncIOMotorDatabase, lead_id: str) -> bool:
    """Delete any lead, regardless of company."""
    if not ObjectId.is_valid(lead_id):
        return False
    result = await db["leads"].delete_one({"_id": ObjectId(lead_id)})
    return result.deleted_count > 0
