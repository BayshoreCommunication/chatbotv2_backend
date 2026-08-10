from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from services import sales_lead_service
from services.admin.admin_auth import get_current_admin
from schemas.sales_lead_schema import (
    SalesLeadCreate,
    SalesLeadResponse,
    SalesLeadsResponse,
    SalesLeadStatusUpdate,
)

router = APIRouter(prefix="/sales-leads", tags=["Sales Leads"])


@router.post("/", response_model=SalesLeadResponse, status_code=status.HTTP_201_CREATED)
async def create_sales_lead(
    payload: SalesLeadCreate,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Public — submitted from the pricing page's Enterprise "Contact Sales"
    modal. No auth: visitors filling this out may not be signed in yet.
    """
    return await sales_lead_service.create_sales_lead(db, payload)


@router.get("/", response_model=SalesLeadsResponse)
async def get_sales_leads(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(get_current_admin),
):
    """Admin-only — list Enterprise inquiries, newest first."""
    return await sales_lead_service.get_sales_leads(db, limit=limit, offset=offset)


@router.patch("/{lead_id}/status", response_model=SalesLeadResponse)
async def update_sales_lead_status(
    lead_id: str,
    payload: SalesLeadStatusUpdate,
    db: AsyncIOMotorDatabase = Depends(get_database),
    _: dict[str, Any] = Depends(get_current_admin),
):
    """Admin-only — toggle a lead's connected/confirmed status."""
    updated = await sales_lead_service.update_sales_lead_status(
        db, lead_id, payload.is_connected, payload.is_confirmed,
    )
    if not updated:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Sales lead not found")
    return updated
