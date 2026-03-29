from motor.motor_asyncio import AsyncIOMotorClient
from config import settings

client: AsyncIOMotorClient = None


def get_database():
    return client[settings.DATABASE_NAME]


async def connect_to_mongo():
    global client
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    print(f"✅ Connected to MongoDB: {settings.MONGODB_URL}")


async def close_mongo_connection():
    global client
    if client:
        client.close()
        print("❌ Disconnected from MongoDB")
