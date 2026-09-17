"""
services/knowledgebase/crawler.py
───────────────────────────────────
BFS website crawler.

Fetches all pages reachable from a root URL (same domain only), plus any
pages listed in the site's sitemap.xml. Returns a list of dicts:
{url, title, raw_text, char_count}, and a merged dict of structured
(JSON-LD / schema.org) data found across pages.
The LLM extractor downstream decides what's worth keeping.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

MAX_PAGES       = 50
MAX_DEPTH       = 3
REQUEST_TIMEOUT = 15
MAX_SITEMAP_URLS   = 200   # cap raw URLs pulled from sitemap(s) before BFS budget applies
MAX_NESTED_SITEMAPS = 5    # cap recursion into <sitemapindex> entries

# Called with a short human-readable message at crawl-phase milestones
# (sitemap check, sitemap result). Purely an observer hook for live
# progress reporting; does not affect what gets crawled.
OnCrawlProgress = Callable[[str], Awaitable[None]]

# Common high-value routes to attempt first regardless of site link structure
PRIORITY_ROUTES = [
    "/",
    "/about",
    "/about-us",
    "/our-story",
    "/company",
    "/services",
    "/service",
    "/what-we-do",
    "/solutions",
    "/products",
    "/portfolio",
    "/work",
    "/case-studies",
    "/contact",
    "/contact-us",
    "/get-in-touch",
    "/reach-us",
    "/team",
    "/our-team",
    "/people",
    "/pricing",
    "/plans",
    "/faq",
    "/faqs",
    "/blog",
]


def _same_domain(base: str, url: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc


# ── Structured data (JSON-LD / schema.org) ───────────────────────────────────

_ORG_TYPES = {"organization", "localbusiness", "corporation"}


def _flatten_ld_nodes(data: Any) -> list[dict]:
    """A JSON-LD payload can be a single object, a list, or use @graph."""
    nodes: list[dict] = []
    if isinstance(data, list):
        for item in data:
            nodes.extend(_flatten_ld_nodes(item))
    elif isinstance(data, dict):
        if "@graph" in data and isinstance(data["@graph"], list):
            for item in data["@graph"]:
                nodes.extend(_flatten_ld_nodes(item))
        else:
            nodes.append(data)
    return nodes


def _extract_json_ld(html: str) -> dict:
    """
    Parse <script type="application/ld+json"> blocks and pull out
    Organization / LocalBusiness fields — a far more reliable source for
    name/contact/description than free text, when the site provides it.
    """
    from bs4 import BeautifulSoup   # lazy import
    soup = BeautifulSoup(html, "html.parser")

    result: dict[str, Any] = {}
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text() or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        for node in _flatten_ld_nodes(data):
            node_type = str(node.get("@type", "")).strip().lower()
            if node_type and node_type not in _ORG_TYPES:
                continue

            if node.get("name") and "name" not in result:
                result["name"] = str(node["name"]).strip()
            if node.get("description") and "description" not in result:
                result["description"] = str(node["description"]).strip()
            if node.get("telephone") and "telephone" not in result:
                result["telephone"] = str(node["telephone"]).strip()
            if node.get("email") and "email" not in result:
                result["email"] = str(node["email"]).strip()

            address = node.get("address")
            if address and "address" not in result:
                if isinstance(address, dict):
                    parts = [
                        address.get("streetAddress"),
                        address.get("addressLocality"),
                        address.get("addressRegion"),
                        address.get("postalCode"),
                    ]
                    joined = ", ".join(str(p).strip() for p in parts if p)
                    if joined:
                        result["address"] = joined
                elif isinstance(address, str) and address.strip():
                    result["address"] = address.strip()

            hours = node.get("openingHours") or node.get("openingHoursSpecification")
            if hours and "opening_hours" not in result:
                if isinstance(hours, list):
                    result["opening_hours"] = "; ".join(str(h) for h in hours if h)
                else:
                    result["opening_hours"] = str(hours).strip()

            same_as = node.get("sameAs")
            if same_as and "same_as" not in result:
                result["same_as"] = (
                    [str(u).strip() for u in same_as]
                    if isinstance(same_as, list)
                    else [str(same_as).strip()]
                )

    return result


def _parse_page(html: str) -> dict:
    """Extract title + clean body text from HTML."""
    from bs4 import BeautifulSoup   # lazy import
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Strip menu links and script/style noise, but keep header/footer text —
    # contact info (phone, address, hours) very commonly lives there.
    for tag in soup(["script", "style", "nav", "aside", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return {"title": title, "text": text}


# ── Sitemap discovery ─────────────────────────────────────────────────────────

_SITEMAP_PATHS = [
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/sitemap-index.xml",
    "/page-sitemap.xml",
]


def _parse_sitemap_xml(xml_text: str) -> tuple[list[str], list[str]]:
    """Returns (page_urls, nested_sitemap_urls) from a sitemap XML document."""
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return [], []

    # Namespace-agnostic: match any tag ending in "loc"/"sitemap"/"url"
    tag = lambda el: el.tag.rsplit("}", 1)[-1]

    if tag(root) == "sitemapindex":
        nested = [
            loc.text.strip()
            for sm in root if tag(sm) == "sitemap"
            for loc in sm if tag(loc) == "loc" and loc.text
        ]
        return [], nested

    pages = [
        loc.text.strip()
        for url_el in root if tag(url_el) == "url"
        for loc in url_el if tag(loc) == "loc" and loc.text
    ]
    return pages, []


async def _discover_sitemap_urls(client: Any, root_url: str) -> list[str]:
    """
    Try robots.txt's `Sitemap:` directive, then common sitemap paths.
    Fully best-effort — any failure (timeout, 404, malformed XML) just
    results in an empty list, never raises.
    """
    base = root_url.rstrip("/")
    candidate_sitemaps: list[str] = []

    try:
        resp = await client.get(f"{base}/robots.txt", timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            for line in resp.text.splitlines():
                if line.strip().lower().startswith("sitemap:"):
                    candidate_sitemaps.append(line.split(":", 1)[1].strip())
    except Exception as exc:
        logger.debug("crawler.robots_txt.skip root=%s reason=%s", root_url, exc)

    if not candidate_sitemaps:
        candidate_sitemaps = [base + p for p in _SITEMAP_PATHS]

    discovered: list[str] = []
    seen: set[str] = set()
    queue = list(candidate_sitemaps)
    nested_fetched = 0

    while queue and len(discovered) < MAX_SITEMAP_URLS:
        sitemap_url = queue.pop(0)
        if sitemap_url in seen:
            continue
        seen.add(sitemap_url)

        try:
            resp = await client.get(sitemap_url, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                continue
        except Exception as exc:
            logger.debug("crawler.sitemap.skip url=%s reason=%s", sitemap_url, exc)
            continue

        pages, nested = _parse_sitemap_xml(resp.text)
        for p in pages:
            if _same_domain(root_url, p) and p not in discovered:
                discovered.append(p)
            if len(discovered) >= MAX_SITEMAP_URLS:
                break

        if nested and nested_fetched < MAX_NESTED_SITEMAPS:
            queue.extend(nested)
            nested_fetched += 1

    return discovered


async def crawl_website(
    root_url: str,
    on_progress: Optional[OnCrawlProgress] = None,
) -> tuple[list[dict], dict]:
    """
    BFS crawl from root_url, seeded with guessed priority routes and any
    URLs discovered via sitemap.xml / robots.txt.

    Returns:
        (pages, structured_data)

        pages: list of page dicts:
        {
            "url":       str,
            "title":     str,
            "raw_text":  str,
            "char_count": int,
        }

        structured_data: dict merged from every page's JSON-LD
        (Organization/LocalBusiness) blocks — first non-empty value per
        field wins. Empty dict if none found.
    """
    import httpx                         # lazy import
    from bs4 import BeautifulSoup        # lazy import

    async def _notify(message: str) -> None:
        if on_progress:
            await on_progress(message)

    visited: set[str] = set()
    structured_data: dict[str, Any] = {}

    # A user may paste a specific page ("…/about-us") instead of the site
    # root — priority routes, sitemap.xml and robots.txt only make sense
    # relative to the domain root, so that's always what they're built
    # from. Nothing is lost: if the given URL had its own path, it's kept
    # as an extra guaranteed seed page below (in addition to, not instead
    # of, the root).
    parsed_input = urlparse(root_url)
    origin = f"{parsed_input.scheme}://{parsed_input.netloc}"
    has_own_path = parsed_input.path not in ("", "/")

    # Seed priority routes first so critical pages are always attempted
    priority_urls = [origin + route for route in PRIORITY_ROUTES]
    seen_seed: set[str] = set()
    seeded: list[tuple[str, int]] = []
    for u in priority_urls:
        if u not in seen_seed:
            seen_seed.add(u)
            seeded.append((u, 0))
    if has_own_path and root_url not in seen_seed:
        seen_seed.add(root_url)
        seeded.append((root_url, 0))

    pages: list[dict] = []

    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0 (compatible; KnowledgeBot/1.0)"},
        follow_redirects=True,
    ) as client:
        await _notify("Checking for a sitemap…")
        sitemap_urls = await _discover_sitemap_urls(client, origin)
        if sitemap_urls:
            logger.info("crawler.sitemap.found root=%s urls=%d", root_url, len(sitemap_urls))
            await _notify(f"Found {len(sitemap_urls)} page(s) via sitemap")
            for u in sitemap_urls:
                if u not in seen_seed:
                    seen_seed.add(u)
                    seeded.append((u, 0))
        else:
            await _notify("No sitemap found — crawling site directly")

        queue: list[tuple[str, int]] = seeded

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

            for key, value in _extract_json_ld(html).items():
                if value and key not in structured_data:
                    structured_data[key] = value

            if depth < MAX_DEPTH:
                soup = BeautifulSoup(html, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = urljoin(url, a["href"]).split("#")[0]
                    if (
                        href not in visited
                        and _same_domain(origin, href)
                        and href.startswith("http")
                    ):
                        queue.append((href, depth + 1))

    logger.info(
        "crawler.done root=%s pages=%d structured_fields=%d",
        root_url, len(pages), len(structured_data),
    )
    return pages, structured_data
