"""
services/chatbot/llm.py
────────────────────────
Central LLM and embeddings instance shared by the agent and any other
service that needs them.

Token usage is logged at INFO level through a custom callback so every
request leaves an audit trail in the logs.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import settings

logger = logging.getLogger(__name__)


# ── Token-usage logging callback ──────────────────────────────────────────────

class _TokenLoggerCallback(BaseCallbackHandler):
    """Logs prompt/completion/total token counts after every LLM call."""

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            usage = response.llm_output.get("token_usage", {})
            prompt_tokens     = usage.get("prompt_tokens", "?")
            completion_tokens = usage.get("completion_tokens", "?")
            total_tokens      = usage.get("total_tokens", "?")
            logger.info(
                "chatbot.llm.token_usage prompt_tokens=%s "
                "completion_tokens=%s total_tokens=%s",
                prompt_tokens, completion_tokens, total_tokens,
            )
        except Exception:
            pass  # never crash the agent over a logging failure


_token_logger = _TokenLoggerCallback()


# ── Chat model (used by the ReAct agent) ─────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",       # swap to "gpt-4o" for higher quality
    temperature=0.7,            # lower = more consistent / factual
    openai_api_key=settings.OPENAI_API_KEY,
    max_retries=3,              # auto-retry on transient 5xx / network errors
    callbacks=[_token_logger],
)

# ── Embeddings (used by the Pinecone knowledge-base tool) ────────────────────
# IMPORTANT: dimensions=1024 must match your Pinecone index dimension.
# text-embedding-3-small natively outputs 1536 but supports truncation via `dimensions`.
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1024,
    openai_api_key=settings.OPENAI_API_KEY,
)

logger.info(
    "chatbot.llm.ready model=gpt-4o-mini temperature=0.7 "
    "embedding_model=text-embedding-3-small embedding_dimensions=1024"
)
