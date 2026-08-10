from datetime import datetime, timedelta, timezone
from typing import Optional
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from model.admin_model import AdminModel
from schemas.admin import AdminCreate, AdminUpdate
from services.admin.admin_auth import hash_password


def serialize_admin(admin: dict) -> dict:
    return {
        "id": str(admin["_id"]),
        "name": admin["name"],
        "email": admin["email"],
        "role": admin.get("role", "admin"),
        "is_active": admin.get("is_active", True),
        "created_at": admin["created_at"],
        "updated_at": admin["updated_at"],
    }


async def create_admin(db: AsyncIOMotorDatabase, payload: AdminCreate) -> Optional[dict]:
    """Creates a regular admin or manager. Only callable by a super admin (enforced in the router).

    Role is always "admin" or "manager" — the single super_admin account is
    fixed and seeded at startup, it is never created through this endpoint.
    """
    if await db["admins"].find_one({"email": payload.email}):
        return None

    now = datetime.utcnow()
    admin_doc = AdminModel(
        name=payload.name,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        role=payload.role,
        created_at=now,
        updated_at=now,
    ).model_dump()

    result = await db["admins"].insert_one(admin_doc)
    admin_doc["_id"] = result.inserted_id
    return serialize_admin(admin_doc)


async def get_all_admins(db: AsyncIOMotorDatabase, page: int = 1, page_size: int = 10) -> dict:
    skip = (page - 1) * page_size
    total = await db["admins"].count_documents({})
    admins = [serialize_admin(a) async for a in db["admins"].find({}).skip(skip).limit(page_size)]
    return {"admins": admins, "total": total, "page": page, "page_size": page_size}


async def get_admin_by_id(db: AsyncIOMotorDatabase, admin_id: str) -> Optional[dict]:
    if not ObjectId.is_valid(admin_id):
        return None
    admin = await db["admins"].find_one({"_id": ObjectId(admin_id)})
    return serialize_admin(admin) if admin else None


async def update_admin(db: AsyncIOMotorDatabase, admin_id: str, payload: AdminUpdate) -> Optional[dict]:
    """Updates a regular admin. The fixed super admin can never be edited via this path."""
    if not ObjectId.is_valid(admin_id):
        return None

    existing = await db["admins"].find_one({"_id": ObjectId(admin_id)})
    if not existing or existing.get("role") == "super_admin":
        return None

    fields = payload.model_dump(exclude_none=True)
    if not fields:
        return serialize_admin(existing)

    if "password" in fields:
        fields["hashed_password"] = hash_password(fields.pop("password"))

    fields["updated_at"] = datetime.utcnow()

    result = await db["admins"].find_one_and_update(
        {"_id": ObjectId(admin_id)},
        {"$set": fields},
        return_document=True,
    )
    return serialize_admin(result) if result else None


async def get_platform_stats(db: AsyncIOMotorDatabase) -> dict:
    """Cross-company totals + breakdowns for the admin dashboard.

    Everything derived from `users` is computed server-side in a single
    $facet aggregation, so it stays correct and cheap regardless of how many
    companies exist — no need to ship full user documents to the client just
    to count them.
    """
    facet_cursor = db["users"].aggregate([
        {"$facet": {
            "verified": [
                {"$match": {"is_verified": True}},
                {"$count": "count"},
            ],
            "trained": [
                {"$match": {"train_data.is_trained": True}},
                {"$count": "count"},
            ],
            "avg_score": [
                {"$group": {"_id": None, "avg": {"$avg": "$train_data.score"}}},
            ],
            "plan_distribution": [
                {"$group": {"_id": "$subscription_type", "count": {"$sum": 1}}},
            ],
        }},
    ])
    facets = (await facet_cursor.to_list(length=1))[0]

    total_companies = await db["users"].count_documents({})

    active_subscriptions = await db["subscriptions"].count_documents({
        "subscription_status": {"$in": ["active", "trialing"]},
        "subscription_tier": {"$ne": "free"},
    })

    active_widgets = await db["widget_settings"].count_documents({})

    # Normalize annual subscriptions to a monthly figure so they're
    # comparable to monthly ones in a single "this month" revenue number.
    revenue_cursor = db["subscriptions"].aggregate([
        {"$match": {"subscription_status": {"$in": ["active", "trialing"]}}},
        {"$group": {
            "_id": None,
            "monthly_total": {
                "$sum": {
                    "$cond": [
                        {"$eq": ["$billing_cycle", "annual"]},
                        {"$divide": ["$payment_amount", 12]},
                        "$payment_amount",
                    ],
                },
            },
        }},
    ])
    revenue_docs = await revenue_cursor.to_list(length=1)
    monthly_revenue = revenue_docs[0]["monthly_total"] if revenue_docs else 0.0

    verified_companies = facets["verified"][0]["count"] if facets["verified"] else 0
    trained_companies = facets["trained"][0]["count"] if facets["trained"] else 0
    avg_kb_score = facets["avg_score"][0]["avg"] if facets["avg_score"] and facets["avg_score"][0]["avg"] else 0.0
    plan_distribution = {row["_id"] or "free": row["count"] for row in facets["plan_distribution"]}

    return {
        "total_companies": total_companies,
        "active_subscriptions": active_subscriptions,
        "monthly_revenue": round(monthly_revenue, 2),
        "active_widgets": active_widgets,
        "verified_companies": verified_companies,
        "trained_companies": trained_companies,
        "avg_kb_score": round(avg_kb_score, 1),
        "plan_distribution": plan_distribution,
    }


def _trend_bucket_key(period: str, doc_id: dict) -> tuple:
    if period == "yearly":
        return (doc_id["year"],)
    return (doc_id["year"], doc_id["month"])


async def get_signup_trends(db: AsyncIOMotorDatabase, period: str = "monthly") -> dict:
    """
    Monthly (trailing 12 months) or yearly (trailing 5 years) breakdown of
    sign-ups, leads captured, and trial→paid conversions — all cross-company,
    for the admin dashboard's overview chart. Every bucket is zero-filled so
    the chart never has gaps.
    """
    now = datetime.now(timezone.utc)
    num_buckets = 5 if period == "yearly" else 12

    if period == "yearly":
        start = datetime(now.year - num_buckets + 1, 1, 1, tzinfo=timezone.utc)
        group_key = {"year": {"$year": "$created_at"}}
        conv_group_key = {"year": {"$year": "$subscription_start_date"}}
    else:
        start = now - timedelta(days=365)
        group_key = {"year": {"$year": "$created_at"}, "month": {"$month": "$created_at"}}
        conv_group_key = {
            "year": {"$year": "$subscription_start_date"},
            "month": {"$month": "$subscription_start_date"},
        }

    signups_cursor = db["users"].aggregate([
        {"$match": {"created_at": {"$gte": start}}},
        {"$group": {"_id": group_key, "count": {"$sum": 1}}},
    ])
    signups_by_key = {
        _trend_bucket_key(period, row["_id"]): row["count"] async for row in signups_cursor
    }

    leads_cursor = db["leads"].aggregate([
        {"$match": {"created_at": {"$gte": start}}},
        {"$group": {"_id": group_key, "count": {"$sum": 1}}},
    ])
    leads_by_key = {
        _trend_bucket_key(period, row["_id"]): row["count"] async for row in leads_cursor
    }

    # A "conversion" = a company whose paid subscription started in this
    # bucket — has_paid_subscription is only ever set True once Stripe
    # confirms an active (non-trial, non-free) subscription.
    conversions_cursor = db["users"].aggregate([
        {"$match": {"has_paid_subscription": True, "subscription_start_date": {"$gte": start}}},
        {"$group": {"_id": conv_group_key, "count": {"$sum": 1}}},
    ])
    conversions_by_key = {
        _trend_bucket_key(period, row["_id"]): row["count"] async for row in conversions_cursor
    }

    points = []
    for i in range(num_buckets - 1, -1, -1):
        if period == "yearly":
            year = now.year - i
            key = (year,)
            label = str(year)
        else:
            year = now.year
            month = now.month - i
            while month <= 0:
                month += 12
                year -= 1
            key = (year, month)
            label = datetime(year, month, 1).strftime("%b %Y")

        points.append({
            "label": label,
            "signups": signups_by_key.get(key, 0),
            "leads": leads_by_key.get(key, 0),
            "conversions": conversions_by_key.get(key, 0),
        })

    return {"period": period, "points": points}


async def delete_admin(db: AsyncIOMotorDatabase, admin_id: str) -> bool:
    """Deletes a regular admin. The fixed super admin can never be deleted."""
    if not ObjectId.is_valid(admin_id):
        return False

    existing = await db["admins"].find_one({"_id": ObjectId(admin_id)})
    if not existing or existing.get("role") == "super_admin":
        return False

    result = await db["admins"].delete_one({"_id": ObjectId(admin_id)})
    return result.deleted_count > 0
