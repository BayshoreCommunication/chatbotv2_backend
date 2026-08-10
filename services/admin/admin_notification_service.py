from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

UNKNOWN_COMPANY = {"company_name": "Unknown company", "email": ""}


async def _company_lookup(db: AsyncIOMotorDatabase, company_ids: set) -> dict:
    """
    Maps company_id (string) -> {"company_name", "email"} via the matching
    `users` documents — same join used by admin_chat_service/admin_lead_service.
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


def _serialize(doc: dict, company: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "company_id": doc.get("company_id"),
        "company_name": company["company_name"],
        "company_email": company["email"],
        "type": doc.get("type"),
        "title": doc.get("title"),
        "message": doc.get("message"),
        "lead_id": doc.get("lead_id"),
        "session_id": doc.get("session_id"),
        "is_read": doc.get("is_read", False),
        "created_at": doc.get("created_at"),
    }


async def get_all_notifications(db: AsyncIOMotorDatabase, limit: int = 20, offset: int = 0) -> dict:
    """Cross-company notification feed for the admin panel, newest first —
    every notification type (lead captured, chat started, subscription
    events) across every organization, enriched with which one raised it."""
    total = await db["notifications"].count_documents({})
    docs = await (
        db["notifications"]
        .find({})
        .sort("created_at", -1)
        .skip(offset)
        .limit(limit)
        .to_list(length=limit)
    )

    companies = await _company_lookup(db, {doc.get("company_id", "") for doc in docs})

    notifications = [
        _serialize(doc, companies.get(doc.get("company_id", ""), UNKNOWN_COMPANY))
        for doc in docs
    ]

    return {"notifications": notifications, "total": total, "limit": limit, "offset": offset}


async def get_unread_count(db: AsyncIOMotorDatabase) -> int:
    return await db["notifications"].count_documents({"is_read": False})


async def mark_all_read(db: AsyncIOMotorDatabase) -> int:
    """Marks every unread notification, across every company, as read — this
    is a shared admin-panel view, not one staff member's personal inbox."""
    result = await db["notifications"].update_many(
        {"is_read": False},
        {"$set": {"is_read": True}},
    )
    return result.modified_count
