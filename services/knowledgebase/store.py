"""
services/knowledgebase/store.py
──────────────────────────────────
Converts LLM-extracted knowledge entries into LangChain Documents,
embeds them, and upserts into Pinecone under namespace = company_id.

Each vector in Pinecone stores rich metadata:
  - company_id, company_name, company_type
  - topic, category, source_url
  - entry_id (deterministic — allows re-training to overwrite stale vectors)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from langchain_core.documents import Document
from config import settings
from services.chatbot.llm import embeddings

logger = logging.getLogger(__name__)


def _make_entry_id(company_id: str, topic: str) -> str:
    """
    Deterministic vector ID: same topic for the same company always gets
    the same ID, so re-training overwrites the old vector instead of
    creating duplicates.
    """
    key = f"{company_id}:{topic.lower().strip()}"
    return hashlib.md5(key.encode()).hexdigest()


def entries_to_documents(
    entries: list[dict],
    company_id: str,
    company_name: str,
    company_type: str,
) -> list[Document]:
    """
    Convert LLM-extracted entries into LangChain Documents.

    Each document's page_content is a clean, readable string:
      Topic: <topic>
      <content>

    Metadata is stored alongside for filtering / attribution.
    """
    docs = []
    for e in entries:
        topic   = e.get("topic", "")
        content = e.get("content", "").strip()
        if not content:
            continue

        page_content = f"Topic: {topic}\n{content}"
        metadata = {
            "entry_id":     _make_entry_id(company_id, topic),
            "company_id":   company_id,
            "company_name": company_name,
            "company_type": company_type,
            "topic":        topic,
            "category":     e.get("category", "overview"),
            "source_url":   e.get("source_url", ""),
        }
        docs.append(Document(page_content=page_content, metadata=metadata))

    return docs


async def upsert_to_pinecone(
    documents: list[Document],
    company_id: str,
) -> int:
    """
    Embed documents and upsert into Pinecone.
    Uses namespace = company_id to isolate each company's knowledge.
    Returns the number of documents upserted.
    """
    from langchain_pinecone import PineconeVectorStore   # lazy import
    if not documents:
        logger.warning("store.upsert.skip reason=no_documents company_id=%s", company_id)
        return 0

    logger.info(
        "store.upsert.start company_id=%s docs=%d namespace=%s index=%s",
        company_id, len(documents), company_id, settings.PINECONE_INDEX,
    )

    import os
    os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY

    await asyncio.to_thread(
        PineconeVectorStore.from_documents,
        documents,
        embeddings,
        index_name=settings.PINECONE_INDEX,
        namespace=company_id,           # each company is fully isolated
    )

    logger.info(
        "store.upsert.done company_id=%s docs=%d",
        company_id, len(documents),
    )
    return len(documents)
