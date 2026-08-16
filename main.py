import asyncio
import logging
from contextlib import asynccontextmanager

from database import close_mongo_connection, connect_to_mongo, get_database
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import (
    admin_route,
    appointments_router,
    apps_integration_router,
    auth_router,
    billing_router,
    chat_router,
    dashboard_router,
    invite_router,
    knowledge_router,
    lead_router,
    meta_webhook_router,
    notification_router,
    sales_lead_router,
    subscription_router,
    team_access_router,
    upload_router,
    user_profile_router,
    user_router,
    widget_settings,
)
from routers.chat_router import widget_router
from services.admin.admin_auth import seed_super_admin
from services.subscription.subscription_service import send_ending_soon_reminders

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
# Keep noisy third-party libs at WARNING so our own logs stand out
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("pinecone").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# How often to check for canceled subscriptions entering their "ends
# tomorrow" reminder window. No Stripe webhook fires for this (unlike trial
# endings), so this is the only thing driving that email.
SUBSCRIPTION_REMINDER_INTERVAL_SECONDS = 60 * 60


async def _subscription_reminder_loop() -> None:
    while True:
        try:
            await send_ending_soon_reminders(get_database())
        except Exception:
            logger.exception("subscription_reminder_loop.failed")
        await asyncio.sleep(SUBSCRIPTION_REMINDER_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    await seed_super_admin(get_database())
    reminder_task = asyncio.create_task(_subscription_reminder_loop())
    yield
    reminder_task.cancel()
    await close_mongo_connection()


app = FastAPI(
    title="AI Chatbot SaaS API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth_router.router, prefix="/api/v1")
app.include_router(user_router.router, prefix="/api/v1")
app.include_router(chat_router.router, prefix="/api/v1")
app.include_router(knowledge_router.router, prefix="/api/v1")
app.include_router(appointments_router.router, prefix="/api")
app.include_router(user_profile_router.router, prefix="/api")
app.include_router(lead_router.router, prefix="/api/v1")
app.include_router(sales_lead_router.router, prefix="/api/v1")
app.include_router(widget_settings.router, prefix="/api/v1")
app.include_router(upload_router.router, prefix="/api/v1")
app.include_router(widget_router, prefix="/api")
app.include_router(subscription_router.router, prefix="/api/v1")
app.include_router(billing_router.router, prefix="/api/v1")
app.include_router(dashboard_router.router, prefix="/api/v1")
app.include_router(notification_router.router, prefix="/api/v1")
app.include_router(admin_route.router, prefix="/api/v1")
app.include_router(invite_router.router, prefix="/api/v1")
app.include_router(team_access_router.router, prefix="/api/v1")
app.include_router(apps_integration_router.router, prefix="/api")
app.include_router(meta_webhook_router.router, prefix="/api")


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "AI Chatbot SaaS API is running 🚀"}


if __name__ == "__main__":
    import uvicorn

    # host="0.0.0.0" = accept connections from any device on the network
    # (phone, emulator, etc.), not just this machine
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
