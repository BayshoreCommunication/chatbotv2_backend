"""
routers/company_auth.py
───────────────────────
Dependency for endpoints that take a `company_id` in the URL (billing and
subscription). The company id is just the owner's user id, and team members
log in with a token whose `sub` is also their owner's id — so the rule is
simply: the token's `sub` must equal the company_id in the path.

Without this, anyone who knew a company_id could list its cards, change its
plan or cancel it.
"""

from __future__ import annotations

from bson import ObjectId
from fastapi import Depends, Header, HTTPException, status
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorDatabase

from config import settings
from database import get_database


async def require_company_access(
    company_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> str:
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

    user = await db["users"].find_one({"_id": ObjectId(user_id), "is_active": True}, {"_id": 1})
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    if user_id != company_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Not allowed for this company")

    return company_id
