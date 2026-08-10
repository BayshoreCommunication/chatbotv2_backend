import logging
from datetime import datetime
from typing import Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

from model.sales_lead_model import SalesLeadModel
from schemas.sales_lead_schema import SalesLeadCreate

logger = logging.getLogger(__name__)


def serialize_sales_lead(lead: dict) -> dict:
    """Convert a MongoDB sales_lead document to a JSON-serializable dict."""
    return {
        "id": str(lead["_id"]),
        "full_name": lead.get("full_name"),
        "email": lead.get("email"),
        "company_name": lead.get("company_name"),
        "company_size": lead.get("company_size"),
        "phone": lead.get("phone"),
        "message": lead.get("message"),
        "is_connected": lead.get("is_connected", False),
        "is_confirmed": lead.get("is_confirmed", False),
        "created_at": lead.get("created_at"),
        "updated_at": lead.get("updated_at"),
    }


async def create_sales_lead(db: AsyncIOMotorDatabase, payload: SalesLeadCreate) -> dict:
    """Insert a new Enterprise inquiry from the pricing page's Contact Sales modal."""
    lead_doc = SalesLeadModel(**payload.model_dump()).model_dump()
    result = await db["sales_leads"].insert_one(lead_doc)
    lead_doc["_id"] = result.inserted_id
    logger.info(
        "sales_leads.created lead_id=%s company_name=%s email=%s",
        result.inserted_id, payload.company_name, payload.email,
    )
    return serialize_sales_lead(lead_doc)


async def get_sales_leads(db: AsyncIOMotorDatabase, limit: int = 20, offset: int = 0) -> dict:
    """Fetch Enterprise inquiries for the admin panel, newest first, paginated."""
    total = await db["sales_leads"].count_documents({})
    cursor = (
        db["sales_leads"]
        .find()
        .sort("created_at", -1)
        .skip(offset)
        .limit(limit)
    )
    leads = [serialize_sales_lead(lead) async for lead in cursor]
    return {"leads": leads, "total": total, "limit": limit, "offset": offset}


async def update_sales_lead_status(
    db: AsyncIOMotorDatabase,
    lead_id: str,
    is_connected: Optional[bool],
    is_confirmed: Optional[bool],
) -> Optional[dict]:
    """
    Toggle the connected/confirmed flags on a sales lead. Confirming a lead
    implies the team already connected with them, so is_confirmed=True
    force-sets is_connected=True too (but not vice versa — you can connect
    without confirming yet).
    """
    if not ObjectId.is_valid(lead_id):
        return None

    updates: dict = {"updated_at": datetime.utcnow()}
    if is_connected is not None:
        updates["is_connected"] = is_connected
    if is_confirmed is not None:
        updates["is_confirmed"] = is_confirmed
        if is_confirmed:
            updates["is_connected"] = True

    result = await db["sales_leads"].find_one_and_update(
        {"_id": ObjectId(lead_id)},
        {"$set": updates},
        return_document=ReturnDocument.AFTER,
    )
    if result:
        logger.info(
            "sales_leads.status_updated lead_id=%s is_connected=%s is_confirmed=%s",
            lead_id, result.get("is_connected"), result.get("is_confirmed"),
        )
    return serialize_sales_lead(result) if result else None
