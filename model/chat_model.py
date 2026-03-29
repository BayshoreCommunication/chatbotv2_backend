from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str       # "user" | "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatSessionModel(BaseModel):
    """Conversation session stored in MongoDB `chat_sessions` collection."""
    company_id: str
    session_id: str
    messages: list[ChatMessage] = []
    exchange_count: int = 0                      # number of user↔AI turn pairs
    tools_used_history: list[list[str]] = []     # tools used per turn
    last_intent: Optional[str] = None            # last classified intent
    lead_captured: bool = False
    # ── Lead contact data (populated once user shares their info) ──────────────
    lead_name: Optional[str] = None
    lead_phone: Optional[str] = None
    lead_email: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
