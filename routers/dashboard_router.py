"""
routers/dashboard_router.py
────────────────────────────
Analytics endpoints for the owner dashboard.

Endpoints:
  GET /dashboard/{company_id}/summary          — KPI cards (sessions, messages, leads, training)
  GET /dashboard/{company_id}/chart            — Chat/lead time-series for bar/line chart
  GET /dashboard/{company_id}/visitors         — Unique visitor stats
  GET /dashboard/{company_id}/recent-sessions  — Last N session previews
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, status

from services.dashboard_service import (
    get_chat_chart,
    get_recent_sessions,
    get_summary,
    get_visitor_stats,
)

router = APIRouter(prefix="/dashboard", tags=["Dashboard Analytics"])


# ── GET /dashboard/{company_id}/summary ──────────────────────────────────────

@router.get("/{company_id}/summary", summary="KPI summary cards")
async def dashboard_summary(company_id: str):
    """
    Returns top-level KPI cards:
    - total_sessions, total_messages, total_leads, total_train_runs
    - kb_score, last_trained
    - deltas: % change vs previous 30-day window for sessions and leads
    """
    try:
        return await get_summary(company_id)
    except Exception as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


# ── GET /dashboard/{company_id}/chart ─────────────────────────────────────────

@router.get("/{company_id}/chart", summary="Sessions/messages/leads time-series")
async def dashboard_chart(
    company_id: str,
    granularity: Literal["weekly", "monthly"] = Query(default="monthly"),
    periods: int = Query(default=12, ge=4, le=52),
    year_offset: int = Query(default=0, ge=0, le=5),
):
    """
    Time-series data for a bar or line chart.
    - granularity: "weekly" (last N ISO weeks) or "monthly" (last N calendar months)
    - periods: number of buckets to return (4–52)

    Each bucket: { label, sessions, messages, leads }
    """
    try:
        return await get_chat_chart(company_id, granularity, periods, year_offset)
    except Exception as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


# ── GET /dashboard/{company_id}/visitors ──────────────────────────────────────

@router.get("/{company_id}/visitors", summary="Unique visitor statistics")
async def dashboard_visitors(company_id: str):
    """
    Returns:
    - total_visitors: all-time unique visitor_id count
    - new_visitors_30d: first-time visitors in the last 30 days
    - returning_visitors: total - new
    """
    try:
        return await get_visitor_stats(company_id)
    except Exception as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


# ── GET /dashboard/{company_id}/recent-sessions ───────────────────────────────

@router.get("/{company_id}/recent-sessions", summary="Recent session previews")
async def dashboard_recent_sessions(
    company_id: str,
    limit: int = Query(default=5, ge=1, le=20),
):
    """
    Returns the most recently active sessions with:
    - session_id, lead_name, lead_captured, exchange_count
    - last_message (content of latest message)
    - updated_at
    """
    try:
        return await get_recent_sessions(company_id, limit)
    except Exception as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
