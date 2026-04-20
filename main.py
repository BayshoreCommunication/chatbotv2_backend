import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import connect_to_mongo, close_mongo_connection
from routers import (
    user_router,
    auth_router,
    chat_router,
    knowledge_router,
    appointments_router,
    user_profile_router,
    lead_router,
    widget_settings,
    upload_router,
    subscription_router,
    dashboard_router,
)
from routers.chat_router import widget_router

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    yield
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
app.include_router(auth_router.router,      prefix="/api/v1")
app.include_router(user_router.router,      prefix="/api/v1")
app.include_router(chat_router.router,      prefix="/api/v1")
app.include_router(knowledge_router.router, prefix="/api/v1")
app.include_router(appointments_router.router, prefix="/api")
app.include_router(user_profile_router.router, prefix="/api")
app.include_router(lead_router.router, prefix="/api/v1")
app.include_router(widget_settings.router, prefix="/api/v1")
app.include_router(upload_router.router,   prefix="/api/v1")
app.include_router(widget_router,                   prefix="/api")
app.include_router(subscription_router.router,      prefix="/api/v1")
app.include_router(dashboard_router.router,         prefix="/api/v1")


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "AI Chatbot SaaS API is running 🚀"}
