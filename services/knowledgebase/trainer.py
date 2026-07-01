"""
services/knowledgebase/trainer.py
────────────────────────────────────
Main orchestrator for the knowledge base training pipeline.

Pipeline:
  1. CRAWL    — BFS-crawl the company website (crawler.py)
  2. SEARCH   — Enrich with DuckDuckGo web search results (web_search.py)
  3. EXTRACT  — LLM reads ALL content and decides what facts to keep (extractor.py)
  4. STORE    — Embed + upsert LLM-approved facts into Pinecone (store.py)
  5. SCORE    — Calculate data quality score (extractor.py)

The LLM is the intelligent gatekeeper: only structured, factual, useful knowledge
reaches Pinecone. Raw noise, boilerplate, and duplicates are discarded.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

# Called as on_progress(percent, stage, message, found) at each stage
# boundary — `found` is an optional list of {category, label, source_url}
# dicts for newly-discovered facts. Purely an observer hook for live
# progress reporting; does not affect the pipeline's actual output.
OnProgress = Callable[[int, str, str, Optional[list[dict]]], Awaitable[None]]

from services.chatbot.llm import llm
from config import settings

from .crawler    import crawl_website
from .web_search import enrich_with_web_search
from .extractor  import extract_knowledge, calculate_quality_score
from .store      import entries_to_documents, upsert_to_pinecone

logger = logging.getLogger(__name__)

# ── Required info checks ──────────────────────────────────────────────────────

_REQUIRED_CHECKS = [
    {
        "key": "company_overview",
        "label": "Company Overview (Home Page)",
        "categories": {"overview"},
        "keywords": [],
    },
    {
        "key": "about_details",
        "label": "About / Company Details",
        "categories": {"team", "about"},
        "keywords": ["founded", "mission", "vision", "history", "about us", "our story"],
    },
    {
        "key": "services",
        "label": "Services / Products",
        "categories": {"services", "products"},
        "keywords": ["service", "product", "offer", "provide", "solution"],
    },
    {
        "key": "contact_phone",
        "label": "Contact Phone Number",
        "categories": {"contact"},
        "keywords": ["phone", "mobile", "call", "tel", "+", "("],
    },
    {
        "key": "contact_email",
        "label": "Contact Email Address",
        "categories": {"contact"},
        "keywords": ["@", "email", "mail"],
    },
    {
        "key": "office_address",
        "label": "Office Address",
        "categories": {"contact", "location"},
        "keywords": ["address", "street", "ave", "road", "blvd", "suite", "floor", "city", "state", "zip"],
    },
    {
        "key": "office_hours",
        "label": "Office / Business Hours",
        "categories": {"contact", "hours"},
        "keywords": ["hours", "monday", "tuesday", "open", "closed", "am", "pm", "9-5", "9 am", "5 pm"],
    },
]


def check_required_info(entries: list[dict]) -> list[dict]:
    """
    Inspect extracted knowledge entries and return a list of important
    info items that appear to be missing.

    Each missing item: {"key": str, "label": str}
    """
    missing = []
    for check in _REQUIRED_CHECKS:
        found = False
        for entry in entries:
            cat = (entry.get("category") or "").lower()
            content = (entry.get("content") or "").lower()
            topic = (entry.get("topic") or "").lower()
            combined = content + " " + topic

            # Match by category
            if cat in check["categories"]:
                # If no keywords required, category match is enough
                if not check["keywords"]:
                    found = True
                    break
                # Otherwise require at least one keyword in content/topic
                if any(kw in combined for kw in check["keywords"]):
                    found = True
                    break

        if not found:
            missing.append({"key": check["key"], "label": check["label"]})

    return missing


async def train_company(
    company_id: str,
    website_url: str,
    company_name: str,
    company_type: str,
    on_progress: Optional[OnProgress] = None,
) -> dict[str, Any]:
    """
    Full LLM-powered knowledge base training pipeline.

    Steps:
      1. Crawl the company website (up to 50 pages)
      2. Run DuckDuckGo web searches for company enrichment
      3. LLM extracts ONLY useful, structured facts from all raw content
      4. Embed facts + upsert into Pinecone (namespace = company_id)
      5. Calculate quality score based on entry count, diversity & coverage

    Returns metadata dict — the caller (knowledge_router) saves this to MongoDB.

    `on_progress`, if given, is called at each stage boundary for live
    progress reporting (see OnProgress above). Purely observational — it
    never changes what any stage computes.

    Return shape:
    {
        "entries_stored":  int,      # LLM-approved facts stored in Pinecone
        "pages_crawled":   int,
        "search_results":  int,      # web search snippets evaluated
        "quality_score":   float,    # 0–100
        "vector_store_id": str,      # Pinecone index name
        "namespace":       str,      # = company_id
        "last_updated":    datetime,
        "categories":      list[str] # knowledge categories found
    }
    """
    async def _report(percent: int, stage: str, message: str, found: Optional[list[dict]] = None) -> None:
        if on_progress:
            await on_progress(percent, stage, message, found)

    logger.info(
        "knowledgebase.train.start company_id=%s website=%s",
        company_id, website_url,
    )
    await _report(2, "starting", "Starting training…")

    # ── Step 1: Crawl website ─────────────────────────────────────────────────
    logger.info("knowledgebase.step=1 crawling website=%s", website_url)
    try:
        crawled_pages = await crawl_website(website_url)
        logger.info("knowledgebase.step=1 completed. crawled %d pages.", len(crawled_pages))
    except Exception as e:
        logger.exception("knowledgebase.step=1 failed during crawl_website: %s", e)
        crawled_pages = []

    if not crawled_pages:
        logger.warning(
            "knowledgebase.no_crawlable_pages company_id=%s website=%s "
            "continuing_with_web_search=true",
            company_id, website_url,
        )
    await _report(20, "crawl", f"Crawled {len(crawled_pages)} page(s) from your website")

    # ── Step 2: Enrich with web search ────────────────────────────────────────
    logger.info("knowledgebase.step=2 web_search company=%s", company_name)
    try:
        search_results = await enrich_with_web_search(
            company_name=company_name,
            company_type=company_type,
            website_url=website_url,
        )
        logger.info("knowledgebase.step=2 completed. found %d search results.", len(search_results))
    except Exception as e:
        logger.exception("knowledgebase.step=2 failed during enrich_with_web_search: %s", e)
        search_results = []

    await _report(35, "search", f"Found {len(search_results)} supporting result(s) from web search")

    if not crawled_pages and not search_results:
        return {
            "entries_stored":  0,
            "pages_crawled":   0,
            "search_results":  0,
            "quality_score":   0.0,
            "vector_store_id": settings.PINECONE_INDEX,
            "namespace":       company_id,
            "last_updated":    datetime.now(timezone.utc),
            "categories":      [],
            "error": "Could not find enough website or web search content to build knowledge.",
        }

    # ── Step 3: LLM extraction — decides what to keep ─────────────────────────
    logger.info(
        "knowledgebase.step=3 llm_extraction pages=%d search_snippets=%d",
        len(crawled_pages), len(search_results),
    )

    # Reports per-batch as extraction progresses — each batch covers a
    # handful of crawled pages (or the web-search snippets), so this is
    # what powers the live "found X on this page" feed during the slowest
    # stage of the pipeline. Capped below 75 — the final jump to 75% is
    # reported once extraction is fully done, below.
    _extract_percent = {"value": 35}

    async def _on_batch_done(label: str, entries: list[dict]) -> None:
        _extract_percent["value"] = min(74, _extract_percent["value"] + 5)
        found = [
            {
                "category":   e.get("category", "overview"),
                "label":      e.get("topic", ""),
                "source_url": label,
            }
            for e in entries
        ]
        message = (
            f"Found {len(entries)} fact(s) on {label}"
            if entries else f"Checked {label} — nothing new found"
        )
        await _report(_extract_percent["value"], "extract", message, found=found or None)

    try:
        knowledge_entries = await extract_knowledge(
            llm=llm,
            company_name=company_name,
            company_type=company_type,
            crawled_pages=crawled_pages,
            search_results=search_results,
            on_batch_done=_on_batch_done if on_progress else None,
        )
        logger.info("knowledgebase.step=3 completed. extracted %d entries.", len(knowledge_entries))
    except Exception as e:
        logger.exception("knowledgebase.step=3 failed during extract_knowledge: %s", e)
        knowledge_entries = []

    await _report(75, "extract", f"Extracted {len(knowledge_entries)} fact(s) total")

    if not knowledge_entries:
        return {
            "entries_stored":  0,
            "pages_crawled":   len(crawled_pages),
            "search_results":  len(search_results),
            "quality_score":   0.0,
            "vector_store_id": settings.PINECONE_INDEX,
            "namespace":       company_id,
            "last_updated":    datetime.now(timezone.utc),
            "categories":      [],
            "error": "LLM could not extract useful knowledge from the content.",
        }

    # ── Step 4: Embed + upsert to Pinecone ────────────────────────────────────
    logger.info(
        "knowledgebase.step=4 pinecone_upsert entries=%d namespace=%s",
        len(knowledge_entries), company_id,
    )
    entries_stored = 0
    if knowledge_entries:
        try:
            documents = entries_to_documents(
                entries=knowledge_entries,
                company_id=company_id,
                company_name=company_name,
                company_type=company_type,
            )
            entries_stored = await upsert_to_pinecone(
                documents=documents,
                company_id=company_id,
            )
            logger.info("knowledgebase.step=4 completed. stored %d entries in pinecone.", entries_stored)
        except Exception as e:
            logger.exception("knowledgebase.step=4 failed during pinecone upsert: %s", e)
    else:
        logger.info("knowledgebase.step=4 skipped because knowledge_entries is empty.")

    await _report(90, "store", f"Stored {entries_stored} fact(s) in your knowledge base")

    # ── Step 5: Quality score ─────────────────────────────────────────────────
    categories = list({e.get("category", "") for e in knowledge_entries if e.get("category")})

    missing_info = check_required_info(knowledge_entries)

    quality_score = calculate_quality_score(
        entries=knowledge_entries,
        pages_crawled=len(crawled_pages),
        missing_info=missing_info,
    )

    await _report(95, "score", f"Quality score: {quality_score:.0f}/100")

    result = {
        "entries_stored":  entries_stored,
        "pages_crawled":   len(crawled_pages),
        "search_results":  len(search_results),
        "quality_score":   quality_score,
        "vector_store_id": settings.PINECONE_INDEX,
        "namespace":       company_id,
        "last_updated":    datetime.now(timezone.utc),
        "categories":      sorted(categories),
        "knowledge_entries": knowledge_entries,
        "missing_info":    missing_info,
    }
    logger.info(
        "knowledgebase.train.complete company_id=%s score=%.1f entries=%d categories=%s",
        company_id, quality_score, entries_stored, categories,
    )
    return result
