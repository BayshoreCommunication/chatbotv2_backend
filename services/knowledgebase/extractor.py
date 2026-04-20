"""
services/knowledgebase/extractor.py
──────────────────────────────────────
The LLM reads ALL raw content (website pages + web search snippets) and
decides what is factually useful, structured, and worth storing as
knowledge base entries.

Only the LLM-approved, cleaned, structured facts go into Pinecone.
Noise, boilerplate, ads, navigation, and duplicate info are discarded.

Output: List of KnowledgeEntry dicts ready to be embedded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
# Each batch groups PAGES_PER_BATCH pages into ONE LLM call.
# 30 pages / 5 per batch = 6 concurrent LLM calls instead of 30 sequential.
MAX_CHARS_PER_PAGE  = 6_000   # chars per page before truncation
PAGES_PER_BATCH     = 5       # pages grouped into one LLM call
MAX_CONCURRENT      = 4       # max simultaneous LLM calls

_EXTRACTOR_SYSTEM_PROMPT = """\
You are a Knowledge Base Extraction AI. Your job is to read raw content from a company's website and web search results and extract ONLY the factual, useful, structured information that would help a chatbot answer questions about this company.

**What to EXTRACT (keep):**
- Company overview (who they are, what they do, location, founding)
- Services and practice areas offered
- Attorneys / team members and their expertise
- Process / how the company works (step-by-step)
- Fees, pricing, payment (e.g. "no win no fee", "free consultation")
- Contact info (phone, email, hours, address)
- FAQs and common client questions with answers
- Client testimonials and reviews (with key phrases)
- Awards, certifications, accreditations
- Important legal/industry terms explained
- Geographic coverage areas

**What to DISCARD (ignore):**
- Navigation menus, headers, footers
- Cookie notices, privacy policy boilerplate
- Repeated text from multiple pages
- Generic legal disclaimers
- SEO filler text
- Ads or promotional banners
- JavaScript/CSS artifacts

**Output format — return ONLY a valid JSON array:**
```json
[
  {
    "topic": "Short descriptive label e.g. 'Free Consultation Policy'",
    "content": "Clean, complete factual sentence(s) about this topic. Written as if explaining to a potential client.",
    "category": "one of: overview|services|team|process|pricing|contact|faq|testimonial|coverage",
    "source_url": "URL where this was found, or 'web_search'"
  }
]
```

Return ONLY the JSON array. No markdown, no explanation. If no useful data found, return [].
"""

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
ADDRESS_HINT_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.\- ]+\s(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Plaza|Pkwy|Parkway)\b",
    re.I,
)
SERVICE_LINE_RE = re.compile(
    r"\b(services|practice areas|what we do|our services)\b[:\-]?\s*(.+)?",
    re.I,
)


def _build_extraction_prompt(
    company_name: str,
    company_type: str,
    raw_content: str,
    source_url: str,
) -> str:
    return (
        f"Company: {company_name}\n"
        f"Type: {company_type}\n"
        f"Source URL: {source_url}\n\n"
        f"RAW CONTENT:\n{raw_content}"
    )


async def _extract_from_batch(
    llm: Any,
    company_name: str,
    company_type: str,
    raw_content: str,
    source_url: str,
) -> list[dict]:
    """Send one batch of raw content to the LLM and return parsed entries."""
    prompt_text = _build_extraction_prompt(
        company_name, company_type, raw_content, source_url
    )
    messages = [
        SystemMessage(content=_EXTRACTOR_SYSTEM_PROMPT),
        HumanMessage(content=prompt_text),
    ]

    try:
        response = await llm.ainvoke(messages)
        text = response.content.strip()

        # Strip markdown fences if the LLM added them
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        entries = json.loads(text)
        if not isinstance(entries, list):
            return []
        return entries

    except json.JSONDecodeError:
        logger.warning(
            "extractor.json_error source=%s — skipping batch", source_url
        )
        return []
    except Exception as exc:
        logger.warning(
            "extractor.llm_error source=%s reason=%s", source_url, exc
        )
        return []


async def extract_knowledge(
    llm: Any,
    company_name: str,
    company_type: str,
    crawled_pages: list[dict],
    search_results: list[dict],
) -> list[dict]:
    """
    Batched + concurrent LLM extraction.

    Instead of one LLM call per page (slow, times out at 30 pages),
    pages are grouped into batches of PAGES_PER_BATCH and all batches
    run concurrently with asyncio.gather (capped at MAX_CONCURRENT).

    This reduces 30 sequential calls (~5 min) to ~6 concurrent batches (~30s).
    """
    all_entries: list[dict] = []
    seen_keys: set[str] = set()

    def _entry_key(entry: dict) -> str:
        topic = str(entry.get("topic", "")).strip().lower()
        category = str(entry.get("category", "")).strip().lower()
        content = str(entry.get("content", "")).strip().lower()
        if topic:
            return f"{category}|{topic}"
        return f"{category}|{content[:120]}"

    def _add_entries(entries: list[dict]) -> None:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            content = str(entry.get("content", "")).strip()
            if not content:
                continue
            key = _entry_key(entry)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_entries.append(entry)

    # ── 1. Build page batches ─────────────────────────────────────────────────
    batches: list[tuple[str, str]] = []   # (combined_text, label)
    for i in range(0, len(crawled_pages), PAGES_PER_BATCH):
        batch_pages = crawled_pages[i : i + PAGES_PER_BATCH]
        combined = "\n\n---\n\n".join(
            f"[Page {i+j+1}: {p['url']}]\n{p['raw_text'][:MAX_CHARS_PER_PAGE]}"
            for j, p in enumerate(batch_pages)
        )
        label = batch_pages[0]["url"]
        batches.append((combined, label))

    total_batches = len(batches)
    logger.info(
        "extractor.batches company=%s pages=%d batches=%d concurrency=%d",
        company_name, len(crawled_pages), total_batches, MAX_CONCURRENT,
    )

    # ── 2. Run page batches concurrently (semaphore to cap concurrency) ────────
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def _run_batch(idx: int, text: str, label: str) -> list[dict]:
        async with sem:
            logger.info(
                "extractor.batch_start company=%s batch=%d/%d",
                company_name, idx + 1, total_batches,
            )
            result = await _extract_from_batch(llm, company_name, company_type, text, label)
            logger.info(
                "extractor.batch_done company=%s batch=%d/%d entries=%d",
                company_name, idx + 1, total_batches, len(result),
            )
            return result

    batch_results = await asyncio.gather(
        *[_run_batch(i, text, label) for i, (text, label) in enumerate(batches)]
    )
    for entries in batch_results:
        _add_entries(entries)

    # ── 3. Extract from web search snippets (single call) ─────────────────────
    if search_results:
        search_text = "\n\n".join(
            f"[{r['query']}] {r['title']}: {r['snippet']}"
            for r in search_results
        )[:MAX_CHARS_PER_PAGE * 2]

        logger.info(
            "extractor.web_search company=%s snippets=%d",
            company_name, len(search_results),
        )
        entries = await _extract_from_batch(
            llm, company_name, company_type, search_text, "web_search"
        )
        _add_entries(entries)

    fallback_entries = _extract_critical_fallback_entries(
        company_name=company_name,
        crawled_pages=crawled_pages,
        search_results=search_results,
        existing_entries=all_entries,
    )
    if fallback_entries:
        logger.info(
            "extractor.fallback_entries company=%s entries=%d",
            company_name,
            len(fallback_entries),
        )
        _add_entries(fallback_entries)

    logger.info(
        "extractor.done company=%s total_entries=%d",
        company_name, len(all_entries),
    )
    return all_entries


def _extract_critical_fallback_entries(
    company_name: str,
    crawled_pages: list[dict],
    search_results: list[dict],
    existing_entries: list[dict],
) -> list[dict]:
    categories = {str(e.get("category", "")).strip().lower() for e in existing_entries}
    need_contact = "contact" not in categories
    need_services = "services" not in categories
    if not need_contact and not need_services:
        return []

    pages_text = "\n".join(str(p.get("raw_text", "")) for p in crawled_pages)
    search_text = "\n".join(
        f"{r.get('title', '')}. {r.get('snippet', '')}" for r in search_results
    )
    source_text = f"{pages_text}\n{search_text}"

    entries: list[dict] = []

    if need_contact:
        email_match = EMAIL_RE.search(source_text)
        phone_match = PHONE_RE.search(source_text)
        address_match = ADDRESS_HINT_RE.search(source_text)
        if email_match or phone_match or address_match:
            parts: list[str] = []
            if email_match:
                parts.append(f"Email: {email_match.group(0)}")
            if phone_match:
                parts.append(f"Phone: {phone_match.group(0).strip()}")
            if address_match:
                parts.append(f"Address: {address_match.group(0).strip()}")
            entries.append(
                {
                    "topic": "Contact Details",
                    "content": f"{company_name} contact information found in source content. " + " ".join(parts),
                    "category": "contact",
                    "source_url": "web_search",
                }
            )

    if need_services:
        service_bits: list[str] = []
        for text in (pages_text, search_text):
            for line in text.splitlines():
                m = SERVICE_LINE_RE.search(line)
                if not m:
                    continue
                cleaned = re.sub(r"\s+", " ", line).strip(" -:|")
                if cleaned and cleaned not in service_bits:
                    service_bits.append(cleaned)
                if len(service_bits) >= 4:
                    break
            if len(service_bits) >= 4:
                break

        if service_bits:
            entries.append(
                {
                    "topic": "Services Overview",
                    "content": f"{company_name} services mentioned in source content: " + " | ".join(service_bits[:4]),
                    "category": "services",
                    "source_url": "web_search",
                }
            )

    return entries


def calculate_quality_score(
    entries: list[dict],
    pages_crawled: int,
    missing_info: list[dict] | None = None,
) -> float:
    """
    Quality score based on:
      - Number of distinct entries extracted (35%)
      - Category diversity                   (25%)
      - Source / page coverage               (20%)
      - Required info completeness           (20%)

    Each missing required info item carries a weighted penalty so the score
    accurately reflects gaps in the knowledge base.

    Returns 0–100.
    """
    if not entries:
        return 0.0

    # ── Axis 1: entry count  (35 pts) — 30 entries = full score ──────────────
    entry_score = min(1.0, len(entries) / 30) * 35

    # ── Axis 2: category diversity  (25 pts) — 9 categories = full score ─────
    categories = {e.get("category", "") for e in entries}
    diversity_score = min(1.0, len(categories) / 9) * 25

    # ── Axis 3: page coverage  (20 pts) — 20 pages = full score ──────────────
    coverage_score = min(1.0, pages_crawled / 20) * 20

    # ── Axis 4: required info completeness  (20 pts) ─────────────────────────
    # Penalty per missing key — weights reflect business importance.
    # Total possible penalty = 20 pts (sum of all weights below).
    _MISSING_WEIGHTS: dict[str, float] = {
        "company_overview": 4.0,   # home page — most critical
        "services":         4.0,   # what do you offer?
        "contact_phone":    3.0,   # lead capture
        "contact_email":    3.0,   # lead capture
        "office_address":   2.5,   # location
        "office_hours":     2.0,   # availability
        "about_details":    1.5,   # nice-to-have
    }
    _TOTAL_WEIGHT = sum(_MISSING_WEIGHTS.values())  # 20.0

    missing_keys = {item["key"] for item in (missing_info or [])}
    penalty = sum(_MISSING_WEIGHTS.get(k, 0.0) for k in missing_keys)
    completeness_score = max(0.0, _TOTAL_WEIGHT - penalty)

    total = entry_score + diversity_score + coverage_score + completeness_score
    return round(min(total, 100.0), 2)
