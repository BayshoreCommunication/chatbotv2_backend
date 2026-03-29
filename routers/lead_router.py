from typing import List, Any
from fastapi import APIRouter, Depends, HTTPException, status, Header
from motor.motor_asyncio import AsyncIOMotorDatabase
from jose import JWTError, jwt
from bson import ObjectId

from database import get_database
from config import settings
from services import lead_service
from schemas.lead_schema import LeadResponse

router = APIRouter(prefix="/leads", tags=["Leads"])

async def get_current_user(
    db: AsyncIOMotorDatabase = Depends(get_database),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> dict[str, Any]:
    if not authorization:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    token = authorization.strip()
    if token.lower().startswith("bearer "):
        token = token.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not ObjectId.is_valid(user_id):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject")

    user = await db["users"].find_one({"_id": ObjectId(user_id), "is_active": True})
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    return {
        "id": str(user["_id"]),
        "email": user.get("email", ""),
        "company_name": user.get("company_name", ""),
    }

@router.get("/", response_model=List[LeadResponse])
async def get_leads(
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Fetch all leads for the current authenticated user's company."""
    return await lead_service.get_leads_by_company(db, current_user["id"])

@router.delete("/{lead_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_lead(
    lead_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Delete a lead belonging to the current user."""
    if not await lead_service.delete_lead(db, lead_id, current_user["id"]):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Lead not found")
