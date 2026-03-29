"""
services/knowledgebase/crawler.py
───────────────────────────────────
BFS website crawler.

Fetches all pages reachable from a root URL (same domain only).
Returns a list of dicts: {url, title, raw_text, char_count}.
The LLM extractor downstream decides what's worth keeping.
"""

from __future__ import annotations

import logging
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

MAX_PAGES       = 30
MAX_DEPTH       = 3
REQUEST_TIMEOUT = 15


def _same_domain(base: str, url: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc


def _parse_page(html: str) -> dict:
    """Extract title + clean body text from HTML."""
    from bs4 import BeautifulSoup   # lazy import
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return {"title": title, "text": text}


async def crawl_website(root_url: str) -> list[dict]:
    """
    BFS crawl from root_url.

    Returns:
        List of page dicts:
        {
            "url":       str,
            "title":     str,
            "raw_text":  str,
            "char_count": int,
        }
    """
    import httpx                         # lazy import
    from bs4 import BeautifulSoup        # lazy import

    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(root_url, 0)]
    pages: list[dict] = []

    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0 (compatible; KnowledgeBot/1.0)"},
        follow_redirects=True,
    ) as client:
        while queue and len(visited) < MAX_PAGES:
            url, depth = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            try:
                resp = await client.get(url, timeout=REQUEST_TIMEOUT)
                if resp.status_code != 200:
                    continue
                ct = resp.headers.get("content-type", "")
                if "text/html" not in ct:
                    continue
                html = resp.text
            except Exception as exc:
                logger.debug("crawler.skip url=%s reason=%s", url, exc)
                continue

            parsed = _parse_page(html)
            if len(parsed["text"]) > 150:
                pages.append({
                    "url":       url,
                    "title":     parsed["title"],
                    "raw_text":  parsed["text"],
                    "char_count": len(parsed["text"]),
                })
                logger.debug("crawler.page url=%s chars=%d", url, len(parsed["text"]))

            if depth < MAX_DEPTH:
                from bs4 import BeautifulSoup   # already imported above but safe
                soup = BeautifulSoup(html, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = urljoin(url, a["href"]).split("#")[0]
                    if (
                        href not in visited
                        and _same_domain(root_url, href)
                        and href.startswith("http")
                    ):
                        queue.append((href, depth + 1))

    logger.info("crawler.done root=%s pages=%d", root_url, len(pages))
    return pages
