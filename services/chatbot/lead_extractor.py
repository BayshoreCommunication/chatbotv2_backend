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
    inquiry: str | None = None  # what service/issue the user is asking about

    @property
    def has_any(self) -> bool:
        return any([self.name, self.phone, self.email])


class _ExtractedLead(BaseModel):
    """Information extracted from the user's message."""

    name: Optional[str] = Field(
        None,
        description="The user's actual name (first, last, or both). Omit filler words like 'my name is', 'I am', 'call me'.",
    )
    phone: Optional[str] = Field(
        None,
        description="The user's phone number. Cleaned of text, retaining plus signs if international.",
    )
    email: Optional[str] = Field(
        None,
        description="The user's email address, standardized and in lowercase.",
    )
    inquiry: Optional[str] = Field(
        None,
        description=(
            "A short label (2-6 words) for the service or issue the user is asking about. "
            "Detect this from any message that reveals intent or problem type. "
            "Examples: 'car accident', 'slip and fall', 'work injury', 'medical negligence', "
            "'schedule appointment', 'pricing inquiry', 'product demo', 'general inquiry'. "
            "Return null if the message is just a greeting, acknowledgement, or gives no service/issue context."
        ),
    )


# Configure the LLM to strictly output the Pydantic schema.
_llm_extractor = llm.with_structured_output(_ExtractedLead)


def _build_prompt(text: str) -> str:
    return (
        "Extract contact info and inquiry type from this user message. "
        "Return null for any field not present. Never invent information.\n\n"
        "CONTACT EXTRACTION examples:\n"
        "- 'arsahak, 01792873207' → name='arsahak', phone='01792873207'\n"
        "- 'my name is John' → name='John'\n"
        "- '01792873207' → phone='01792873207'\n"
        "- 'john@gmail.com' → email='john@gmail.com'\n"
        "- 'Sarah, sarah@gmail.com, 555-1234' → name='Sarah', phone='555-1234', email='sarah@gmail.com'\n"
        "- 'hello how are you' → all null\n"
        "- 'yes' or 'okay' or 'no' → all null\n\n"
        "INQUIRY EXTRACTION examples:\n"
        "- 'I was in a car accident yesterday' → inquiry='car accident'\n"
        "- 'I slipped and fell at a store' → inquiry='slip and fall'\n"
        "- 'I need help with a work injury' → inquiry='work injury'\n"
        "- 'how much does it cost' → inquiry='pricing inquiry'\n"
        "- 'I want to schedule an appointment' → inquiry='schedule appointment'\n"
        "- 'yes' or 'okay' or 'no' or 'yesterday' → inquiry=null\n\n"
        f"Message: {text}"
    )


def extract_lead_info(text: str) -> LeadInfo:
    if not text or len(text.strip()) < 2:
        return LeadInfo()

    try:
        result = _llm_extractor.invoke(_build_prompt(text))
        return LeadInfo(name=result.name, phone=result.phone, email=result.email, inquiry=result.inquiry)
    except Exception as exc:
        logger.error("lead_extractor.sync_failed error=%s", exc)
        return LeadInfo()


async def extract_lead_info_async(text: str) -> LeadInfo:
    if not text or len(text.strip()) < 2:
        return LeadInfo()

    try:
        result = await _llm_extractor.ainvoke(_build_prompt(text))
        return LeadInfo(name=result.name, phone=result.phone, email=result.email, inquiry=result.inquiry)
    except Exception as exc:
        logger.error("lead_extractor.async_failed error=%s", exc)
        return LeadInfo()
