from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorDatabase

from model.widget_settings import WidgetSettingsModel

COLLECTION = "widget_settings"


def _serialize(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "company_id": doc.get("company_id"),
        "bot_name": doc.get("bot_name"),
        "theme": doc.get("theme"),
        "behavior": doc.get("behavior"),
        "content": doc.get("content"),
        "launcher": doc.get("launcher"),
    }


async def get_settings(db: AsyncIOMotorDatabase, company_id: str) -> dict | None:
    doc = await db[COLLECTION].find_one({"company_id": company_id})
    if doc:
        return _serialize(doc)
    return None


async def upsert_settings(
    db: AsyncIOMotorDatabase, company_id: str, data: WidgetSettingsModel
) -> dict:
    payload = {
        **data.model_dump(),
        "company_id": company_id,
        "updated_at": datetime.now(timezone.utc),
    }
    result = await db[COLLECTION].find_one_and_update(
        {"company_id": company_id},
        {"$set": payload},
        upsert=True,
        return_document=True,
    )
    return _serialize(result)


async def delete_settings(db: AsyncIOMotorDatabase, company_id: str) -> bool:
    result = await db[COLLECTION].delete_one({"company_id": company_id})
    return result.deleted_count > 0
