"""
routers/knowledge_router.py
─────────────────────────────
Endpoints for training a company's AI knowledge base.

  POST /knowledge/train/{company_id}  — crawl website, embed, upsert to Pinecone
  GET  /knowledge/status/{company_id} — return training metadata from MongoDB
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, HTTPException, status, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, HttpUrl

from database import get_database
from services.knowledgebase import train_company
from services.chatbot import invalidate_company_agent
from services.chatbot.company_context import invalidate_context
from model.knowledge_model import TrainResult as TrainResultModel, KnowledgeBaseDocument, TrainRunHistory

router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])
logger = logging.getLogger(__name__)


# ── Schemas ───────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    website_url: str
    company_name: str
    company_type: str = "other"


class TrainResponse(BaseModel):
    message: str
    company_id: str
    pages_crawled: int
    search_results: int          # web search snippets evaluated by LLM
    entries_stored: int          # LLM-approved facts stored in Pinecone
    quality_score: float         # 0–100
    categories: list[str]        # knowledge categories extracted
    vector_store_id: str
    namespace: str
    last_updated: datetime


class TrainStatusResponse(BaseModel):
    company_id: str
    company_name: str
    is_trained: bool
    quality_score: float
    entries_stored: int          # LLM-approved facts stored in Pinecone
    pages_crawled: int
    categories: list[str]
    last_updated: datetime | None
    update_count: int
    update_limit: int
    vector_store_id: str | None
    namespace: str | None


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_company(db: AsyncIOMotorDatabase, company_id: str) -> dict:
    if not ObjectId.is_valid(company_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid company ID.")
    company = await db["users"].find_one({"_id": ObjectId(company_id)})
    if not company:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Company not found.")
    return company


# ── POST /knowledge/train/{company_id} ───────────────────────────────────────

@router.post(
    "/train/{company_id}",
    response_model=TrainResponse,
    summary="Crawl website and train company knowledge base",
)
async def train(
    company_id: str,
    payload: TrainRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    1. Crawls all pages reachable from `website_url` (same domain, max 25 pages).
    2. Splits content into chunks and embeds with OpenAI.
    3. Upserts vectors into Pinecone under `namespace = company_id`.
    4. Calculates a quality score (0–100) and saves metadata to MongoDB.
    """
    company = await _get_company(db, company_id)

    train_data = company.get("train_data", {})
    update_count = train_data.get("update_count", 0)
    update_limit = train_data.get("update_limit", 10)

    if update_count >= update_limit:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Training update limit reached ({update_limit}). Upgrade your plan.",
        )

    logger.info(
        "knowledge.train.start company_id=%s website=%s",
        company_id, payload.website_url,
    )

    # Run the training pipeline
    logger.info("Starting pipeline for company_id=%s, URL=%s", company_id, payload.website_url)
    try:
        raw_result = await train_company(
            company_id=company_id,
            website_url=payload.website_url,
            company_name=payload.company_name,
            company_type=payload.company_type,
        )
    except Exception as e:
        logger.exception("Pipeline failed unexpectedly for company_id=%s: %s", company_id, e)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Training failed with an internal error: {e}"
        )

    if "error" in raw_result and raw_result["error"]:
        logger.error("Pipeline returned logic error for company_id=%s: %s", company_id, raw_result["error"])
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, raw_result["error"])

    # Validate the result through the Pydantic model — catches missing/bad fields early
    try:
        result = TrainResultModel(**raw_result)
    except Exception as e:
        logger.exception("TrainResult validation failed for company_id=%s: %s", company_id, e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Training result invalid: {e}")

    # Build history entry (shared by both collections)
    history_entry = TrainRunHistory(
        updated_at=result.last_updated.isoformat(),
        website_url=payload.website_url,
        pages_crawled=result.pages_crawled,
        search_results=result.search_results,
        entries_stored=result.entries_stored,
        quality_score=result.quality_score,
        categories=result.categories,
    ).model_dump()

    # ── 1. Update users document (training stats embedded on the account) ──────
    await db["users"].update_one(
        {"_id": ObjectId(company_id)},
        {
            "$set": {
                "vector_store_id":              result.vector_store_id,
                "train_data.is_trained":        True,
                "train_data.score":             result.quality_score,
                "train_data.last_updated":      result.last_updated,
                "train_data.entries_stored":    result.entries_stored,
                "train_data.pages_crawled":     result.pages_crawled,
                "train_data.categories":        result.categories,
                "train_data.namespace":         company_id,
                "updated_at":                   datetime.now(timezone.utc),
            },
            "$inc": {"train_data.update_count": 1},
            "$push": {
                "train_data.history": {
                    "$each":     [history_entry],
                    "$slice":    -20,
                    "$position": 0,
                }
            },
        },
    )
    logger.info("knowledge.users_doc.updated company_id=%s", company_id)

    # ── 2. Upsert full KnowledgeBaseDocument to knowledge_base collection ──────
    # One document per company — contains every LLM-extracted fact that was
    # indexed in Pinecone, so you can audit the knowledge base from MongoDB.
    kb_doc = KnowledgeBaseDocument(
        company_id=company_id,
        company_name=payload.company_name,
        company_type=payload.company_type,
        website_url=payload.website_url,
        entries_stored=result.entries_stored,
        pages_crawled=result.pages_crawled,
        search_results=result.search_results,
        quality_score=result.quality_score,
        categories=result.categories,
        vector_store_id=result.vector_store_id,
        namespace=result.namespace,
        last_updated=result.last_updated,
        entries=[e for e in result.knowledge_entries],
    )
    await db["knowledge_base"].update_one(
        {"company_id": company_id},
        {
            "$set": {
                **{k: v for k, v in kb_doc.model_dump(exclude={"history"}).items()},
                "last_updated": result.last_updated,
            },
            "$push": {
                "history": {
                    "$each":     [history_entry],
                    "$slice":    -20,
                    "$position": 0,
                }
            },
        },
        upsert=True,   # create the doc on first training, update on re-training
    )
    logger.info(
        "knowledge.knowledge_base_doc.upserted company_id=%s entries=%d",
        company_id, result.entries_stored,
    )

    logger.info(
        "knowledge.train.done company_id=%s score=%.1f entries=%d pages=%d",
        company_id,
        result.quality_score,
        result.entries_stored,
        result.pages_crawled,
    )

    # ── Evict caches so next chat uses fresh KB data ──────────────────────────
    invalidate_context(company_id)          # clear 5-min company context cache
    invalidate_company_agent(company_id)    # force agent rebuild with new tools/prompt
    logger.info(
        "knowledge.train.caches_evicted company_id=%s",
        company_id,
    )

    return TrainResponse(
        message="Training complete — knowledge base is ready.",
        company_id=company_id,
        pages_crawled=result.pages_crawled,
        search_results=result.search_results,
        entries_stored=result.entries_stored,
        quality_score=result.quality_score,
        categories=result.categories,
        vector_store_id=result.vector_store_id,
        namespace=result.namespace,
        last_updated=result.last_updated,
    )


# ── GET /knowledge/status/{company_id} ───────────────────────────────────────

@router.get(
    "/status/{company_id}",
    response_model=TrainStatusResponse,
    summary="Get training status for a company",
)
async def get_status(
    company_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Returns training metadata stored on the company's MongoDB document."""
    company = await _get_company(db, company_id)
    td = company.get("train_data", {})

    return TrainStatusResponse(
        company_id=company_id,
        company_name=company.get("company_name", ""),
        is_trained=td.get("is_trained", False),
        quality_score=td.get("score", 0.0),
        entries_stored=td.get("entries_stored", 0),
        pages_crawled=td.get("pages_crawled", 0),
        categories=td.get("categories", []),
        last_updated=td.get("last_updated"),
        update_count=td.get("update_count", 0),
        update_limit=td.get("update_limit", 10),
        vector_store_id=company.get("vector_store_id"),
        namespace=td.get("namespace"),
    )
