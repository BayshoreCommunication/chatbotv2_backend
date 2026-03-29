from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _run_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Blocking DuckDuckGo search — must be called via asyncio.to_thread.
    Returns [] gracefully if ddgs is not installed.
    """
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url":     r.get("href", ""),
                })
        return results
    except Exception as exc:
        logger.warning("web_search.skip query=%r reason=%s", query, exc)
        return []


async def enrich_with_web_search(
    company_name: str,
    company_type: str,
    website_url: str,
) -> list[dict]:
    """
    Run focused DuckDuckGo searches about the company.
    Each search is off-loaded to a thread so the event loop is not blocked.
    """
    parsed = urlparse(website_url)
    domain = parsed.netloc or website_url.replace("https://", "").replace("http://", "").split("/")[0]

    queries = [
        f"{company_name} {company_type} services",
        f"{company_name} practice areas services",
        f"{company_name} contact email phone address office hours",
        f"{company_name} reviews client testimonials",
        f"{company_name} {company_type} FAQ frequently asked questions",
        f"site:{domain} contact",
        f"site:{domain} services",
        f"site:{domain} about",
    ]

    all_results: list[dict] = []
    seen: set[str] = set()
    for query in queries:
        # Run blocking DDGS call in a thread pool
        hits = await asyncio.to_thread(_run_search, query, 4)
        for hit in hits:
            dedupe_key = f"{hit['url']}|{hit['title']}".strip().lower()
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            all_results.append({
                "source":  "web_search",
                "query":   query,
                "title":   hit["title"],
                "snippet": hit["snippet"],
                "url":     hit["url"],
            })

    logger.info(
        "web_search.done company=%s queries=%d results=%d",
        company_name, len(queries), len(all_results),
    )
    return all_results
