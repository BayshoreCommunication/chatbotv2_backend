from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
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
)
from services.admin import admin_service
from services.admin.admin_auth import signin, get_current_admin, require_super_admin

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
