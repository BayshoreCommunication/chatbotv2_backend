"""
services/dashboard_service.py
──────────────────────────────
All MongoDB aggregation logic for the dashboard analytics endpoints.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

import bson
from database import get_database


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _week_start(dt: datetime) -> datetime:
    return (dt - timedelta(days=dt.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def _month_start(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


# ── summary ───────────────────────────────────────────────────────────────────

async def get_summary(company_id: str) -> dict:
    db = get_database()
    now = _now()
    period_start = now - timedelta(days=30)
    prev_start   = period_start - timedelta(days=30)

    # sessions
    total_sessions = await db["chat_sessions"].count_documents({"company_id": company_id})
    sessions_this  = await db["chat_sessions"].count_documents({
        "company_id": company_id, "created_at": {"$gte": period_start},
    })
    sessions_prev  = await db["chat_sessions"].count_documents({
        "company_id": company_id, "created_at": {"$gte": prev_start, "$lt": period_start},
    })

    # messages
    msg_result = await db["chat_sessions"].aggregate([
        {"$match": {"company_id": company_id}},
        {"$project": {"n": {"$size": {"$ifNull": ["$messages", []]}}}},
        {"$group": {"_id": None, "total": {"$sum": "$n"}}},
    ]).to_list(1)
    total_messages = msg_result[0]["total"] if msg_result else 0

    # leads
    total_leads = await db["leads"].count_documents({"company_id": company_id})
    leads_this  = await db["leads"].count_documents({
        "company_id": company_id, "created_at": {"$gte": period_start},
    })
    leads_prev  = await db["leads"].count_documents({
        "company_id": company_id, "created_at": {"$gte": prev_start, "$lt": period_start},
    })

    # knowledge base / training info from users document
    user_doc = await db["users"].find_one(
        {"_id": bson.ObjectId(company_id)},
        {"train_data": 1},
    )
    train_data    = (user_doc or {}).get("train_data", {})
    total_trains  = int(train_data.get("update_count", 0))
    last_trained  = train_data.get("last_updated")
    kb_score      = float(train_data.get("score", 0.0))
    entries_stored = int(train_data.get("entries_stored", 0))
    pages_crawled  = int(train_data.get("pages_crawled", 0))

    def _delta(this: int, prev: int) -> float:
        if prev == 0:
            return 100.0 if this > 0 else 0.0
        return round((this - prev) / prev * 100, 1)

    return {
        "total_sessions":   total_sessions,
        "total_messages":   total_messages,
        "total_leads":      total_leads,
        "total_train_runs": total_trains,
        "entries_stored":   entries_stored,
        "pages_crawled":    pages_crawled,
        "kb_score":         kb_score,
        "last_trained":     last_trained.isoformat() if last_trained else None,
        "deltas": {
            "sessions_pct": _delta(sessions_this, sessions_prev),
            "leads_pct":    _delta(leads_this,    leads_prev),
        },
    }


# ── chart ─────────────────────────────────────────────────────────────────────

async def get_chat_chart(
    company_id: str,
    granularity: Literal["weekly", "monthly"],
    periods: int = 12,
    year_offset: int = 0,
) -> list[dict]:
    """
    Time-series chart data.  year_offset=0 → most recent N periods,
    year_offset=1 → same N periods shifted back 1 year.

    Each bucket: { label, sessions, messages, leads, visitors }
    """
    db  = get_database()
    now = _now() - timedelta(days=365 * year_offset)

    buckets = _weekly_buckets(now, periods) if granularity == "weekly" else _monthly_buckets(now, periods)
    cutoff  = buckets[0]["start"]

    if granularity == "weekly":
        group_id = {
            "year": {"$isoWeekYear": "$created_at"},
            "week": {"$isoWeek":     "$created_at"},
        }
    else:
        group_id = {
            "year":  {"$year":  "$created_at"},
            "month": {"$month": "$created_at"},
        }

    rows = await db["chat_sessions"].aggregate([
        {"$match": {"company_id": company_id, "created_at": {"$gte": cutoff}}},
        {"$group": {
            "_id":         group_id,
            "sessions":    {"$sum": 1},
            "messages":    {"$sum": {"$size": {"$ifNull": ["$messages", []]}}},
            "leads":       {"$sum": {"$cond": [{"$eq": ["$lead_captured", True]}, 1, 0]}},
            "visitor_ids": {"$addToSet": "$visitor_id"},
        }},
        {"$addFields": {
            "visitors": {
                "$size": {
                    "$filter": {
                        "input": "$visitor_ids",
                        "as":    "v",
                        "cond":  {"$ne": ["$$v", None]},
                    }
                }
            }
        }},
    ]).to_list(None)

    row_index = {_bucket_key(r["_id"], granularity): r for r in rows}

    return [
        {
            "label":    b["label"],
            "sessions": row_index.get(_bucket_key_from_bucket(b, granularity), {}).get("sessions", 0),
            "messages": row_index.get(_bucket_key_from_bucket(b, granularity), {}).get("messages", 0),
            "leads":    row_index.get(_bucket_key_from_bucket(b, granularity), {}).get("leads",    0),
            "visitors": row_index.get(_bucket_key_from_bucket(b, granularity), {}).get("visitors", 0),
        }
        for b in buckets
    ]


# ── visitors ──────────────────────────────────────────────────────────────────

async def get_visitor_stats(company_id: str) -> dict:
    db  = get_database()
    now = _now()
    period_start = now - timedelta(days=30)

    total_res = await db["chat_sessions"].aggregate([
        {"$match": {"company_id": company_id, "visitor_id": {"$exists": True, "$ne": None}}},
        {"$group": {"_id": "$visitor_id"}},
        {"$count": "total"},
    ]).to_list(1)
    total_visitors = total_res[0]["total"] if total_res else 0

    new_res = await db["chat_sessions"].aggregate([
        {"$match": {"company_id": company_id, "visitor_id": {"$exists": True, "$ne": None}}},
        {"$group": {"_id": "$visitor_id", "first_seen": {"$min": "$created_at"}}},
        {"$match": {"first_seen": {"$gte": period_start}}},
        {"$count": "total"},
    ]).to_list(1)
    new_visitors = new_res[0]["total"] if new_res else 0

    return {
        "total_visitors":     total_visitors,
        "new_visitors_30d":   new_visitors,
        "returning_visitors": max(0, total_visitors - new_visitors),
    }


# ── recent sessions ───────────────────────────────────────────────────────────

async def get_recent_sessions(company_id: str, limit: int = 5) -> list[dict]:
    db   = get_database()
    docs = await (
        db["chat_sessions"]
        .find({"company_id": company_id})
        .sort("updated_at", -1)
        .limit(limit)
        .to_list(limit)
    )
    result = []
    for doc in docs:
        last_msg = (doc.get("messages") or [{}])[-1]
        result.append({
            "session_id":     doc.get("session_id", ""),
            "lead_name":      doc.get("lead_name"),
            "lead_captured":  doc.get("lead_captured", False),
            "exchange_count": doc.get("exchange_count", 0),
            "last_message":   last_msg.get("content", ""),
            "updated_at":     doc["updated_at"].isoformat() if doc.get("updated_at") else None,
        })
    return result


# ── bucket helpers ────────────────────────────────────────────────────────────

def _weekly_buckets(now: datetime, count: int) -> list[dict]:
    buckets, week = [], _week_start(now)
    for _ in range(count):
        iso = week.isocalendar()
        buckets.insert(0, {
            "start": week,
            "end":   week + timedelta(days=7),
            "label": week.strftime("W%V %Y"),
            "year":  iso[0],
            "week":  iso[1],
        })
        week -= timedelta(days=7)
    return buckets


def _monthly_buckets(now: datetime, count: int) -> list[dict]:
    buckets, month = [], _month_start(now)
    for _ in range(count):
        next_m = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
        buckets.insert(0, {
            "start": month,
            "end":   next_m,
            "label": month.strftime("%b %Y"),
            "year":  month.year,
            "month": month.month,
        })
        month = (month - timedelta(days=1)).replace(day=1)
    return buckets


def _bucket_key(group_id: dict, granularity: str) -> str:
    if granularity == "weekly":
        return f"{group_id['year']}-{group_id['week']}"
    return f"{group_id['year']}-{group_id['month']}"


def _bucket_key_from_bucket(bucket: dict, granularity: str) -> str:
    if granularity == "weekly":
        return f"{bucket['year']}-{bucket['week']}"
    return f"{bucket['year']}-{bucket['month']}"
