import bcrypt
from datetime import datetime
from bson import ObjectId
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorDatabase

from schemas.user import UserCreate, UserUpdate
from model.user_model import UserModel, TrainData


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def serialize_user(user: dict) -> dict:
    return {
        "id": str(user["_id"]),
        "company_name": user["company_name"],
        "company_type": user.get("company_type", "other"),
        "company_website": user.get("company_website"),
        "email": user["email"],
        "role": user.get("role", "organization"),
        "is_active": user.get("is_active", True),
        "is_verified": user.get("is_verified", False),
        "is_subscribed": user.get("is_subscribed", False),
        "has_paid_subscription": user.get("has_paid_subscription", False),
        "subscription_type": user.get("subscription_type", "free"),
        "subscription_start_date": user.get("subscription_start_date"),
        "subscription_end_date": user.get("subscription_end_date"),
        "vector_store_id": user.get("vector_store_id"),
        "train_data": user.get("train_data", TrainData().model_dump()),
        "created_at": user["created_at"],
        "updated_at": user["updated_at"],
    }


async def create_user(db: AsyncIOMotorDatabase, user_data: UserCreate) -> Optional[dict]:
    if await db["users"].find_one({"email": user_data.email}):
        return None

    now = datetime.utcnow()
    user_doc = UserModel(
        company_name=user_data.company_name,
        company_type=user_data.company_type,
        company_website=user_data.company_website,
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        role=user_data.role,
        created_at=now,
        updated_at=now,
    ).model_dump()

    result = await db["users"].insert_one(user_doc)
    user_doc["_id"] = result.inserted_id
    return serialize_user(user_doc)


async def get_all_users(db: AsyncIOMotorDatabase, page: int = 1, page_size: int = 10) -> dict:
    skip = (page - 1) * page_size
    total = await db["users"].count_documents({})
    users = [serialize_user(u) async for u in db["users"].find({}).skip(skip).limit(page_size)]
    return {"users": users, "total": total, "page": page, "page_size": page_size}


async def get_user_by_id(db: AsyncIOMotorDatabase, user_id: str) -> Optional[dict]:
    if not ObjectId.is_valid(user_id):
        return None
    user = await db["users"].find_one({"_id": ObjectId(user_id)})
    return serialize_user(user) if user else None


async def update_user(db: AsyncIOMotorDatabase, user_id: str, user_data: UserUpdate) -> Optional[dict]:
    if not ObjectId.is_valid(user_id):
        return None

    fields = user_data.model_dump(exclude_none=True)
    if not fields:
        return await get_user_by_id(db, user_id)

    if "password" in fields:
        fields["hashed_password"] = hash_password(fields.pop("password"))

    fields["updated_at"] = datetime.utcnow()

    result = await db["users"].find_one_and_update(
        {"_id": ObjectId(user_id)},
        {"$set": fields},
        return_document=True,
    )
    return serialize_user(result) if result else None


async def delete_user(db: AsyncIOMotorDatabase, user_id: str) -> bool:
    if not ObjectId.is_valid(user_id):
        return False
    result = await db["users"].delete_one({"_id": ObjectId(user_id)})
    return result.deleted_count > 0
