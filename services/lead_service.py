from datetime import datetime
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId

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
