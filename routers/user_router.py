from fastapi import APIRouter, HTTPException, status, Depends, Query
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from schemas.user import UserCreate, UserUpdate, UserResponse, UserListResponse
from services import user_service

router = APIRouter(prefix="/users", tags=["Users"])


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(payload: UserCreate, db: AsyncIOMotorDatabase = Depends(get_database)):
    user = await user_service.create_user(db, payload)
    if user is None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=f"Email '{payload.email}' is already registered.")
    return user


@router.get("/", response_model=UserListResponse)
async def list_users(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    return await user_service.get_all_users(db, page=page, page_size=page_size)


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, db: AsyncIOMotorDatabase = Depends(get_database)):
    user = await user_service.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"User '{user_id}' not found.")
    return user


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, payload: UserUpdate, db: AsyncIOMotorDatabase = Depends(get_database)):
    user = await user_service.update_user(db, user_id, payload)
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"User '{user_id}' not found.")
    return user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str, db: AsyncIOMotorDatabase = Depends(get_database)):
    if not await user_service.delete_user(db, user_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"User '{user_id}' not found.")
