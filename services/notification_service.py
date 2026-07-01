import logging
from datetime import datetime
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
from pymongo import ReturnDocument

from model.notification_model import NotificationModel, NotificationType

logger = logging.getLogger(__name__)


def serialize_notification(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "company_id": doc.get("company_id"),
        "type": doc.get("type"),
        "title": doc.get("title"),
        "message": doc.get("message"),
        "lead_id": doc.get("lead_id"),
        "session_id": doc.get("session_id"),
        "is_read": doc.get("is_read", False),
        "created_at": doc.get("created_at"),
    }


async def _insert(
    db: AsyncIOMotorDatabase,
    company_id: str,
    type_: NotificationType,
    title: str,
    message: str,
    lead_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """
    Validate against NotificationModel before writing — every notification
    in the `notifications` collection goes through this single path, so a
    bad field/type fails loudly here instead of producing a malformed doc.
    """
    notification = NotificationModel(
        company_id=company_id,
        type=type_,
        title=title,
        message=message,
        lead_id=lead_id,
        session_id=session_id,
    )
    result = await db["notifications"].insert_one(notification.model_dump())
    logger.info(
        "notifications.created company_id=%s type=%s notification_id=%s lead_id=%s session_id=%s",
        company_id, type_, result.inserted_id, lead_id, session_id,
    )


async def create_lead_notification(
    db: AsyncIOMotorDatabase,
    company_id: str,
    lead_id: str,
    lead_name: Optional[str],
    lead_contact: Optional[str],
) -> None:
    """Insert a notification when a brand-new lead is captured by the chatbot."""
    who = lead_name or lead_contact or "A visitor"
    await _insert(
        db, company_id, "lead_captured",
        title="New lead captured",
        message=f"{who} just left their contact info on your chatbot.",
        lead_id=lead_id,
    )


async def create_chat_notification(
    db: AsyncIOMotorDatabase,
    company_id: str,
    session_id: str,
) -> None:
    """Insert a notification when a brand-new chat session/visitor starts."""
    await _insert(
        db, company_id, "chat_started",
        title="New visitor started a chat",
        message="A new visitor just opened a conversation with your chatbot.",
        session_id=session_id,
    )


async def create_subscription_ending_notification(
    db: AsyncIOMotorDatabase,
    company_id: str,
    ends_at: Optional[datetime],
) -> None:
    """Insert a notification when a subscription is set to cancel at period end."""
    when = ends_at.strftime("%B %d, %Y") if ends_at else "the end of the billing period"
    await _insert(
        db, company_id, "subscription_ending",
        title="Subscription ending soon",
        message=f"Your subscription is set to cancel and will end on {when}.",
    )


async def create_subscription_canceled_notification(
    db: AsyncIOMotorDatabase, company_id: str,
) -> None:
    """Insert a notification when a subscription has fully ended/canceled."""
    await _insert(
        db, company_id, "subscription_canceled",
        title="Subscription canceled",
        message="Your subscription has ended. You're now on the free plan.",
    )


async def create_payment_failed_notification(
    db: AsyncIOMotorDatabase, company_id: str,
) -> None:
    """Insert a notification when a subscription payment attempt fails."""
    await _insert(
        db, company_id, "payment_failed",
        title="Payment failed",
        message="We couldn't process your subscription payment. Please update your payment method.",
    )


async def create_trial_ending_notification(
    db: AsyncIOMotorDatabase,
    company_id: str,
    trial_end: Optional[datetime],
) -> None:
    """Insert a notification 3 days before the free trial ends (Stripe trial_will_end event)."""
    when = trial_end.strftime("%B %d, %Y") if trial_end else "soon"
    await _insert(
        db, company_id, "trial_ending",
        title="Free trial ending in 3 days",
        message=f"Your 14-day free trial ends on {when}. Your saved card will be charged automatically — cancel anytime before then to avoid a charge.",
    )


async def get_notifications(
    db: AsyncIOMotorDatabase, company_id: str, limit: int = 20,
) -> List[dict]:
    cursor = (
        db["notifications"]
        .find({"company_id": company_id})
        .sort("created_at", -1)
        .limit(limit)
    )
    return [serialize_notification(doc) async for doc in cursor]


async def get_unread_count(db: AsyncIOMotorDatabase, company_id: str) -> int:
    return await db["notifications"].count_documents(
        {"company_id": company_id, "is_read": False},
    )


async def mark_all_read(db: AsyncIOMotorDatabase, company_id: str) -> int:
    result = await db["notifications"].update_many(
        {"company_id": company_id, "is_read": False},
        {"$set": {"is_read": True}},
    )
    return result.modified_count


async def mark_one_read(
    db: AsyncIOMotorDatabase, company_id: str, notification_id: str,
) -> Optional[dict]:
    if not ObjectId.is_valid(notification_id):
        return None
    doc = await db["notifications"].find_one_and_update(
        {"_id": ObjectId(notification_id), "company_id": company_id},
        {"$set": {"is_read": True}},
        return_document=ReturnDocument.AFTER,
    )
    return serialize_notification(doc) if doc else None
