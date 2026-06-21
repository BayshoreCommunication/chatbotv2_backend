"""
routers/billing_router.py
────────────────────────────
Payment-method and invoice-history endpoints for the billing dashboard.
Plan/checkout/cancel endpoints live in routers/subscription_router.py.

  GET    /billing/payment-methods/{company_id}                      — list saved cards
  POST   /billing/payment-methods/{company_id}/default               — set default card
  DELETE /billing/payment-methods/{company_id}/{payment_method_id}   — remove a card
  POST   /billing/setup-intent/{company_id}                          — start adding a new card (Stripe Elements)
  GET    /billing/invoices/{company_id}                              — payment history
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel

from database import get_database
from services import billing_service

router = APIRouter(prefix="/billing", tags=["Billing"])


class SetDefaultPaymentMethodRequest(BaseModel):
    payment_method_id: str


def _raise_service_error(result: dict) -> None:
    """Convert service-layer error dicts into HTTP exceptions."""
    err = result.get("error")
    if not err:
        return
    detail = result.get("detail", err)
    if err == "no_subscription":
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail)
    if err == "default_payment_method":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail)
    raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail)


# ── Payment methods ───────────────────────────────────────────────────────────

@router.get(
    "/payment-methods/{company_id}",
    summary="List saved payment methods (cards) for a company",
)
async def get_payment_methods(
    company_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    result = await billing_service.list_payment_methods(db, company_id)
    _raise_service_error(result)
    return result   # {"cards": [...]}


@router.post(
    "/payment-methods/{company_id}/default",
    summary="Set a card as the default payment method",
)
async def set_default_payment_method(
    company_id: str,
    payload: SetDefaultPaymentMethodRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    result = await billing_service.set_default_payment_method(
        db, company_id, payload.payment_method_id,
    )
    _raise_service_error(result)
    return {"ok": True}


@router.delete(
    "/payment-methods/{company_id}/{payment_method_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a saved card",
)
async def remove_payment_method(
    company_id: str,
    payment_method_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    result = await billing_service.remove_payment_method(db, company_id, payment_method_id)
    _raise_service_error(result)


@router.post(
    "/setup-intent/{company_id}",
    summary="Create a SetupIntent so the frontend can collect a new card via Stripe Elements",
)
async def create_setup_intent(
    company_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    result = await billing_service.create_setup_intent(db, company_id)
    _raise_service_error(result)
    return result   # {"client_secret": "seti_..._secret_..."}


# ── Invoice history ───────────────────────────────────────────────────────────

@router.get(
    "/invoices/{company_id}",
    summary="Get payment/invoice history for a company",
)
async def get_invoices(
    company_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    result = await billing_service.list_invoices(db, company_id)
    _raise_service_error(result)
    return result   # {"invoices": [...]}
