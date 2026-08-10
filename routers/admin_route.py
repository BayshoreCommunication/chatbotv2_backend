from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from schemas.admin import (
    AdminSigninRequest,
    AdminTokenResponse,
    AdminCreate,
    AdminUpdate,
    AdminResponse,
    AdminListResponse,
    AdminPlatformStats,
    SignupTrendPeriod,
    SignupTrendsResponse,
)
from schemas.admin_chat_schema import (
    AdminChatSessionsResponse,
    AdminChatSessionDetail,
    AdminCompanyVisitorsResponse,
    AdminVisitorChatHistory,
)
from schemas.admin_lead_schema import (
    AdminLeadsResponse,
    AdminLeadResponse,
    AdminLeadContactedUpdate,
)
from schemas.admin_notification_schema import AdminNotificationsResponse
from services.admin import admin_service, admin_chat_service, admin_lead_service, admin_notification_service
from services.admin.admin_auth import signin, get_current_admin, require_super_admin
from services.chatbot.ws_manager import ws_manager

router = APIRouter(prefix="/admin", tags=["Admin"])

ERROR_MAP = {
    "invalid_credentials": (status.HTTP_401_UNAUTHORIZED, "Invalid email or password."),
    "account_disabled":    (status.HTTP_403_FORBIDDEN,     "Your account has been disabled."),
}


def raise_if_error(result: dict):
    error = result.get("error")
    if error:
        code, detail = ERROR_MAP.get(error, (status.HTTP_400_BAD_REQUEST, error))
        raise HTTPException(status_code=code, detail=detail)


# ── Auth ──────────────────────────────────────────────────────────────────────

@router.post("/signin", response_model=AdminTokenResponse, summary="Admin/Super admin sign in")
async def admin_signin(payload: AdminSigninRequest, db: AsyncIOMotorDatabase = Depends(get_database)):
    """Sign in with email + password. Works for both `admin` and `super_admin` roles."""
    result = await signin(db, payload)
    raise_if_error(result)
    return result


@router.get("/me", response_model=AdminResponse, summary="Get the current admin's profile")
async def get_me(
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_admin: dict[str, Any] = Depends(get_current_admin),
):
    admin = await admin_service.get_admin_by_id(db, current_admin["id"])
    if not admin:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Admin not found.")
    return admin


# ── Platform stats ────────────────────────────────────────────────────────────

@router.get(
    "/platform-stats",
    response_model=AdminPlatformStats,
    summary="Cross-company totals for the admin dashboard (any staff role)",
)
async def platform_stats(
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(get_current_admin),
):
    return await admin_service.get_platform_stats(db)


@router.get(
    "/platform-stats/trends",
    response_model=SignupTrendsResponse,
    summary="Sign-ups, leads, and conversions per month/year (any staff role)",
)
async def platform_stats_trends(
    period: SignupTrendPeriod = Query(default="monthly"),
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(get_current_admin),
):
    return await admin_service.get_signup_trends(db, period=period)


# ── Chat sessions (cross-company, any staff role) ─────────────────────────────

@router.get(
    "/chats",
    response_model=AdminChatSessionsResponse,
    summary="List chat sessions across all companies, newest first",
)
async def list_chat_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(get_current_admin),
):
    """Each session is enriched with company_name/company_email so the admin
    can tell which organization a conversation belongs to and group by it."""
    return await admin_chat_service.get_all_chat_sessions(db, limit=limit, offset=offset)


# NOTE: both of these must be registered before the generic
# "/chats/{company_id}/{session_id}" route below — FastAPI/Starlette matches
# routes in registration order, and "visitors" would otherwise be swallowed
# as a {session_id} value by that catch-all.

@router.get(
    "/chats/{company_id}/visitors",
    response_model=AdminCompanyVisitorsResponse,
    summary="List distinct visitors for one company, most recently active first",
)
async def list_company_visitors(
    company_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(get_current_admin),
):
    return await admin_chat_service.get_company_visitors(db, company_id)


@router.get(
    "/chats/{company_id}/visitors/{visitor_id}/sessions",
    response_model=AdminChatSessionsResponse,
    summary="All chat sessions for one visitor within a company",
)
async def list_visitor_sessions(
    company_id: str,
    visitor_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(get_current_admin),
):
    return await admin_chat_service.get_visitor_sessions(db, company_id, visitor_id)


@router.get(
    "/chats/{company_id}/visitors/{visitor_id}",
    response_model=AdminVisitorChatHistory,
    summary="One visitor's full chat history, flattened across all their sessions",
)
async def get_visitor_chat_history(
    company_id: str,
    visitor_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(get_current_admin),
):
    history = await admin_chat_service.get_visitor_chat_history(db, company_id, visitor_id)
    if not history:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Visitor not found.")
    return history


@router.get(
    "/chats/{company_id}/{session_id}",
    response_model=AdminChatSessionDetail,
    summary="Full transcript for one chat session",
)
async def get_chat_session(
    company_id: str,
    session_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(get_current_admin),
):
    session = await admin_chat_service.get_chat_session_detail(db, company_id, session_id)
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Chat session not found.")
    return session


# ── Leads (cross-company, any staff role) ─────────────────────────────────────
# The company-scoped /api/v1/leads/... endpoints (lead_router.py) require a
# `users`-collection token and always 401 for an admin session — these mirror
# them without the company_id scoping, enriched with company_name/email.

@router.get(
    "/leads",
    response_model=AdminLeadsResponse,
    summary="List leads across all companies, newest first",
)
async def list_all_leads(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(get_current_admin),
):
    return await admin_lead_service.get_all_leads(db, limit=limit, offset=offset)


@router.get(
    "/leads/search",
    response_model=AdminLeadsResponse,
    summary="Search leads across all companies by name, email, or phone",
)
async def search_all_leads(
    q: str = Query(default=""),
    limit: int = Query(default=10, ge=1, le=50),
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(get_current_admin),
):
    return await admin_lead_service.search_leads(db, q, limit=limit)


@router.post(
    "/leads/{lead_id}/contacted",
    response_model=AdminLeadResponse,
    summary="Mark a lead as contacted/uncontacted",
)
async def update_lead_contacted(
    lead_id: str,
    payload: AdminLeadContactedUpdate,
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(get_current_admin),
):
    updated = await admin_lead_service.set_lead_contacted(db, lead_id, payload.is_contacted)
    if not updated:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Lead not found.")
    return updated


@router.delete(
    "/leads/{lead_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a lead",
)
async def delete_lead(
    lead_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(get_current_admin),
):
    if not await admin_lead_service.delete_lead(db, lead_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Lead not found.")


# ── Notifications (cross-company, any staff role) ─────────────────────────────
# The company-scoped /api/v1/notifications/... endpoints require a
# `users`-collection token and always 401 for an admin session — these mirror
# them without the company_id scoping, enriched with company_name.

@router.get(
    "/notifications",
    response_model=AdminNotificationsResponse,
    summary="List notifications across all companies, newest first",
)
async def list_all_notifications(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(get_current_admin),
):
    return await admin_notification_service.get_all_notifications(db, limit=limit, offset=offset)


@router.get(
    "/notifications/unread-count",
    summary="Unread notification count across all companies — poll for the topbar bell badge",
)
async def admin_unread_count(
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(get_current_admin),
):
    count = await admin_notification_service.get_unread_count(db)
    return {"count": count}


@router.post(
    "/notifications/mark-read",
    summary="Mark every notification as read, across all companies",
)
async def admin_mark_all_read(
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(get_current_admin),
):
    modified = await admin_notification_service.mark_all_read(db)
    return {"modified": modified}


@router.websocket("/notifications/ws")
async def admin_notifications_ws(websocket: WebSocket):
    """
    Real-time push for the admin topbar bell — every notification created
    anywhere in the platform (services/notification_service.py's single
    `_insert()` write path) gets broadcast here the instant it's written, to
    every connected admin browser tab. Unauthenticated, matching the existing
    dashboard chat WS (chat_router.py's `dashboard_ws`) — same posture, not a
    new relaxation.
    """
    await ws_manager.connect_admin(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive pings from the client
    except WebSocketDisconnect:
        ws_manager.disconnect_admin(websocket)


# ── Admin management (super admin only) ───────────────────────────────────────

@router.post(
    "/admins",
    response_model=AdminResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new admin or manager (super admin only)",
)
async def create_admin(
    payload: AdminCreate,
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(require_super_admin),
):
    admin = await admin_service.create_admin(db, payload)
    if admin is None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=f"Email '{payload.email}' is already registered.")
    return admin


@router.get("/admins", response_model=AdminListResponse, summary="List all admins (super admin only)")
async def list_admins(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(require_super_admin),
):
    return await admin_service.get_all_admins(db, page=page, page_size=page_size)


@router.get("/admins/{admin_id}", response_model=AdminResponse, summary="Get an admin by id (super admin only)")
async def get_admin(
    admin_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(require_super_admin),
):
    admin = await admin_service.get_admin_by_id(db, admin_id)
    if not admin:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Admin '{admin_id}' not found.")
    return admin


@router.patch("/admins/{admin_id}", response_model=AdminResponse, summary="Update an admin (super admin only)")
async def update_admin(
    admin_id: str,
    payload: AdminUpdate,
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(require_super_admin),
):
    admin = await admin_service.update_admin(db, admin_id, payload)
    if not admin:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"Admin '{admin_id}' not found, or it is the fixed super admin and cannot be edited.",
        )
    return admin


@router.delete("/admins/{admin_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete an admin (super admin only)")
async def delete_admin(
    admin_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(require_super_admin),
):
    if not await admin_service.delete_admin(db, admin_id):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"Admin '{admin_id}' not found, or it is the fixed super admin and cannot be deleted.",
        )
