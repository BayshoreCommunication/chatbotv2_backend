"""
services/chatbot/lead_extractor.py
────────────────────────────────────
LLM-powered lead data extractor.

Uses structured output (with_structured_output) from the pre-configured
LLM to gracefully extract name, phone, and email from messy human inputs.
Returns a LeadInfo dataclass. All fields are Optional[str].
None means "not found in this message".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field
from services.chatbot.llm import llm

logger = logging.getLogger(__name__)


@dataclass
class LeadInfo:
    name:  str | None = None
    phone: str | None = None
    email: str | None = None

    @property
    def has_any(self) -> bool:
        return any([self.name, self.phone, self.email])


class _ExtractedLead(BaseModel):
    """Information extracted from the user's message."""
    name: Optional[str] = Field(
        None, 
        description="The user's actual name (first, last, or both). Strictly omit filler words like 'Name', 'is', 'and', 'my', etc."
    )
    phone: Optional[str] = Field(
        None, 
        description="The user's phone number. Cleaned of text, retaining plus signs if international."
    )
    email: Optional[str] = Field(
        None, 
        description="The user's email address, standardized and in lowercase."
    )


# Configure the LLM to strictly output the Pydantic schema
_llm_extractor = llm.with_structured_output(_ExtractedLead)


def extract_lead_info(text: str) -> LeadInfo:
    """
    Extract name, phone, and email from a single user message using an LLM.

    Args:
        text: The raw user message string.

    Returns:
        LeadInfo with whichever fields were found (others remain None).
    """
    if not text or len(text.strip()) < 2:
        return LeadInfo()

    prompt = (
        "Extract any contact information (name, phone, email) from the following user message. "
        "If a field is not present, leave it null/None. "
        "Do not invent information. If they say 'Name Sahak and email arsahak@gmail.com', extract name='Sahak'.\n\n"
        f"Message (User): {text}"
    )

    try:
        result = _llm_extractor.invoke(prompt)
        return LeadInfo(
            name=result.name,
            phone=result.phone,
            email=result.email
        )
    except Exception as e:
        logger.error("LLM lead extraction failed: %s", e)
        return LeadInfo()
