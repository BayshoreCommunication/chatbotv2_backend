"""
model/knowledge_model.py
──────────────────────────
Pydantic models for the knowledge base training pipeline.

MongoDB collection: `knowledge_base`
──────────────────────────────────────
One document per company (upserted on each training run).
Stores the full training result including every extracted knowledge entry,
so you can inspect exactly what was indexed in Pinecone without querying
the vector store itself.

Document shape in MongoDB:
{
  "_id":            ObjectId,
  "company_id":     "699d34ecf9838ca515c89bb4",   ← references users._id
  "company_name":   "Carter Injury Law",
  "company_type":   "law-firm",
  "website_url":    "https://www.carterinjurylaw.com",

  # ── Training stats ──
  "entries_stored":  18,
  "pages_crawled":   30,
  "search_results":  16,
  "quality_score":   84.0,
  "categories":      ["overview", "services", ...],
  "vector_store_id": "bayai",
  "namespace":       "699d34ecf9838ca515c89bb4",
  "last_updated":    ISODate(...),

  # ── All LLM-extracted facts that went into Pinecone ──
  "entries": [
    {
      "topic":      "Free Consultation Policy",
      "content":    "Carter Injury Law offers a free initial consultation...",
      "category":   "pricing",
      "source_url": "https://www.carterinjurylaw.com/contact"
    },
    ...
  ],

  # ── Run history (last 20 training runs) ──
  "history": [ { "updated_at": "...", "pages_crawled": 30, ... }, ... ]
}
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Single LLM-extracted knowledge entry ─────────────────────────────────────

class KnowledgeEntry(BaseModel):
    """One fact extracted by the LLM and indexed in Pinecone."""
    topic:      str             # e.g. "Free Consultation Policy"
    content:    str             # Clean factual sentence(s)
    category:   str = "overview"   # overview|services|team|process|
                                   # pricing|contact|faq|testimonial|coverage
    source_url: str = ""        # Original page URL or "web_search"


# ── One training run snapshot (stored in history[]) ───────────────────────────

class TrainRunHistory(BaseModel):
    """One entry in knowledge_base.history[] — one snapshot per training run."""
    updated_at:     str         # ISO datetime string
    website_url:    str
    pages_crawled:  int   = 0
    search_results: int   = 0   # web search snippets evaluated
    entries_stored: int   = 0   # LLM-approved facts stored in Pinecone
    quality_score:  float = 0.0
    categories:     list[str] = Field(default_factory=list)


# ── Full training result (returned by trainer.py) ────────────────────────────

class TrainResult(BaseModel):
    """
    Returned by `services.knowledgebase.trainer.train_company()`.
    Validated by knowledge_router before writing to MongoDB.
    """
    entries_stored:    int
    pages_crawled:     int
    search_results:    int
    quality_score:     float
    categories:        list[str]
    vector_store_id:   str
    namespace:         str
    last_updated:      datetime
    knowledge_entries: list[dict] = Field(default_factory=list)  # raw LLM entries
    error:             Optional[str] = None


# ── MongoDB collection document: `knowledge_base` ─────────────────────────────

class KnowledgeBaseDocument(BaseModel):
    """
    One document per company in the `knowledge_base` MongoDB collection.

    Upserted on every successful training run so you can always inspect
    exactly which facts are stored in Pinecone — without querying the
    vector store itself.

    Use `company_id` as the upsert key (one doc per company, overwritten
    each time). History is appended via $push/$slice.
    """
    company_id:     str         # references users._id (as string)
    company_name:   str
    company_type:   str
    website_url:    str

    # ── Training stats ────────────────────────────────────────────────────────
    entries_stored:  int   = 0
    pages_crawled:   int   = 0
    search_results:  int   = 0
    quality_score:   float = 0.0
    categories:      list[str] = Field(default_factory=list)
    vector_store_id: str   = ""
    namespace:       str   = ""
    last_updated:    Optional[datetime] = None

    # ── All LLM-extracted facts stored in Pinecone ────────────────────────────
    entries: list[KnowledgeEntry] = Field(
        default_factory=list,
        description="Every fact the LLM extracted and indexed in Pinecone."
    )

    # ── Run history (populated via $push in router) ───────────────────────────
    history: list[TrainRunHistory] = Field(default_factory=list)
