from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field
from services.chatbot.llm import llm

logger = logging.getLogger(__name__)


@dataclass
class LeadInfo:
    name: str | None = None
    phone: str | None = None
    email: str | None = None

    @property
    def has_any(self) -> bool:
        return any([self.name, self.phone, self.email])


class _ExtractedLead(BaseModel):
    """Information extracted from the user's message."""

    name: Optional[str] = Field(
        None,
        description="The user's actual name (first, last, or both). Strictly omit filler words like 'Name', 'is', 'and', 'my', etc.",
    )
    phone: Optional[str] = Field(
        None,
        description="The user's phone number. Cleaned of text, retaining plus signs if international.",
    )
    email: Optional[str] = Field(
        None,
        description="The user's email address, standardized and in lowercase.",
    )


# Configure the LLM to strictly output the Pydantic schema.
_llm_extractor = llm.with_structured_output(_ExtractedLead)


def _build_prompt(text: str) -> str:
    return (
        "Extract any contact information (name, phone, email) from the following user message. "
        "If a field is not present, leave it null/None. "
        "Do not invent information. If they say 'Name Sahak and email arsahak@gmail.com', extract name='Sahak'.\n\n"
        f"Message (User): {text}"
    )


def extract_lead_info(text: str) -> LeadInfo:
    """
    Sync extractor kept for compatibility with existing tests and scripts.
    """
    if not text or len(text.strip()) < 2:
        return LeadInfo()

    try:
        result = _llm_extractor.invoke(_build_prompt(text))
        return LeadInfo(name=result.name, phone=result.phone, email=result.email)
    except Exception as exc:
        logger.error("lead_extractor.sync_failed error=%s", exc)
        return LeadInfo()


async def extract_lead_info_async(text: str) -> LeadInfo:
    """
    Async extractor for request handlers so we do not block the event loop.
    """
    if not text or len(text.strip()) < 2:
        return LeadInfo()

    try:
        result = await _llm_extractor.ainvoke(_build_prompt(text))
        return LeadInfo(name=result.name, phone=result.phone, email=result.email)
    except Exception as exc:
        logger.error("lead_extractor.async_failed error=%s", exc)
        return LeadInfo()
