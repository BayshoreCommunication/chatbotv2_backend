from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class AdminChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None


class AdminChatSessionSummary(BaseModel):
    """One row in the cross-company chat list — enriched with which
    organization the session belongs to, since chat_sessions on its own only
    stores company_id, not a display name."""
    session_id: str
    company_id: str
    company_name: str
    company_email: str
    exchange_count: int
    lead_captured: bool
    lead_name: Optional[str] = None
    human_takeover: bool = False
    last_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AdminChatSessionsResponse(BaseModel):
    sessions: list[AdminChatSessionSummary]
    total: int
    limit: int
    offset: int


class AdminChatSessionDetail(BaseModel):
    session_id: str
    company_id: str
    company_name: str
    company_email: str
    visitor_id: Optional[str] = None
    exchange_count: int
    lead_captured: bool
    lead_name: Optional[str] = None
    lead_phone: Optional[str] = None
    lead_email: Optional[str] = None
    human_takeover: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    messages: list[AdminChatMessage]


class AdminChatVisitorSummary(BaseModel):
    """One distinct visitor within a company, aggregated across all of their
    sessions — visitor_id falls back to session_id for sessions that predate
    visitor tracking (X-Visitor-ID header), so every session still surfaces
    as browsable even without a real cross-session visitor identity."""
    visitor_id: str
    session_count: int
    total_exchanges: int
    lead_captured: bool
    lead_name: Optional[str] = None
    lead_email: Optional[str] = None
    lead_phone: Optional[str] = None
    last_message: Optional[str] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None


class AdminCompanyVisitorsResponse(BaseModel):
    company_id: str
    company_name: str
    company_email: str
    visitors: list[AdminChatVisitorSummary]
    total: int


class AdminVisitorChatHistory(BaseModel):
    """One visitor's full chat history within a company — every message from
    every one of their sessions, flattened into a single chronological
    thread, for the admin's 3-column chat viewer (mirrors the dashboard's
    own chat history layout: visitor list / message thread / info panel)."""
    visitor_id: str
    company_id: str
    company_name: str
    company_email: str
    lead_name: Optional[str] = None
    lead_email: Optional[str] = None
    lead_phone: Optional[str] = None
    lead_captured: bool = False
    session_count: int
    total_exchanges: int
    messages: list[AdminChatMessage]
