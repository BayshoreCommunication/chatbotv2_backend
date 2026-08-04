"""
model/apps_integration_model.py
──────────────────────────────────
Settings models for third-party channel integrations (Messenger, and later
Instagram) shown on the dashboard's "Apps & Integrations" page. WhatsApp was
removed — not needed for this product. One file for all channels — each
platform gets its own settings classes below, following the same shape
(settings + connect request + login-for-business connect request +
test-connection response + Mongo doc), so Instagram can be added as a
sibling here without a new file per platform.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ── Messenger ─────────────────────────────────────────────────────────────────

class MessengerSettings(BaseModel):
    page_id: str = ""
    access_token: str = ""
    page_name: str = ""


class MessengerSettingsResponse(BaseModel):
    settings: MessengerSettings
    connected: bool


class MessengerConnectRequest(BaseModel):
    page_id: str = Field(..., min_length=1)
    access_token: str = Field(..., min_length=1)


# Facebook Login for Business ("Continue with Facebook") — the frontend gets a
# short-lived authorization `code` from FB.login(); Meta doesn't hand back a
# picked Page here, so `page_id` is optional — the backend auto-selects it
# when the login only grants access to one Page, and only needs page_id to
# disambiguate when there's more than one. We never see a manually-copied
# Page Access Token here; the backend exchanges `code` for a long-lived user
# token itself, then reads the matching Page Access Token off GET /me/accounts.
class MessengerEmbeddedConnectRequest(BaseModel):
    code: str = Field(..., min_length=1)
    page_id: str = ""


class MessengerTestConnectionResponse(BaseModel):
    valid: bool
    page_name: str = ""


class MessengerSettingsDoc(BaseModel):
    company_id: str
    page_id: str = ""
    access_token: str = ""
    page_name: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Instagram (not built yet — placeholder for the next channel) ──────────────
# class InstagramSettings(BaseModel): ...


# ── WhatsApp (Embedded Signup) ─────────────────────────────────────────────
# The frontend's FB.login() WhatsApp Embedded Signup flow hands back a short-
# lived authorization `code` (exchanged server-side, same helpers Messenger's
# embedded flow uses) plus `phone_number_id`/`waba_id` read directly off the
# WA_EMBEDDED_SIGNUP postMessage event Meta fires during the flow — unlike
# Messenger, there's no page to resolve afterward, Meta hands us the exact
# phone number/account the user picked in its own signup wizard.
class WhatsAppEmbeddedConnectRequest(BaseModel):
    code: str = Field(..., min_length=1)
    phone_number_id: str = Field(..., min_length=1)
    waba_id: str = Field(..., min_length=1)


# ── Unified channel connections (Messenger + Instagram) ────────────────────────
# The multi-connection model: one row per (company, channel, external
# Page/IG account), letting a company connect more than one Page. Wired into
# services/apps_integration/channel_connections_service.py and the
# /apps-integration/oauth/* + /api/webhooks/meta routes. Coexists with
# MessengerSettingsDoc above (the older single-connection flow) rather than
# replacing it — see the router docstring for which routes use which.

class ChannelType(str, Enum):
    messenger = "messenger"
    instagram = "instagram"
    whatsapp = "whatsapp"


class ConnectionStatus(str, Enum):
    active = "active"
    disconnected = "disconnected"
    token_expired = "token_expired"


class ChannelConnectionDoc(BaseModel):
    """Mongo document shape for the `channel_connections` collection.
    `encrypted_access_token` is ciphertext, never a raw token — write it with
    utils.token_encryption.encrypt_token() and read it back with
    decrypt_token(), never store or compare the plaintext value directly."""
    id: str | None = None
    company_id: str
    channel: ChannelType
    external_id: str
    encrypted_access_token: str
    page_name: str = ""
    connected_at: datetime = Field(default_factory=datetime.utcnow)
    status: ConnectionStatus = ConnectionStatus.active


# ── OAuth connect flow (server-redirect, not the JS-SDK popup Messenger
# above uses) ───────────────────────────────────────────────────────────────
# initiate -> Facebook's own OAuth dialog -> callback (Meta redirects the
# raw browser here, no Authorization header) -> pending selection stashed
# server-side -> confirm (dashboard, authenticated) picks which Page(s) to
# actually connect.

class OAuthInitiateResponse(BaseModel):
    authorize_url: str


class OAuthCandidatePage(BaseModel):
    external_id: str
    page_name: str
    # An Instagram Business Account linked to this Page, if any — selecting
    # the Page in `confirm` connects both together, since IG messaging
    # authenticates with the same Page Access Token.
    linked_instagram_id: str | None = None
    linked_instagram_name: str | None = None


class OAuthPendingSelectionResponse(BaseModel):
    selection_id: str
    pages: list[OAuthCandidatePage]


class OAuthConfirmRequest(BaseModel):
    selection_id: str = Field(..., min_length=1)
    # external_id values from OAuthCandidatePage.external_id (Page ids).
    external_ids: list[str] = Field(..., min_length=1)


class ChannelConnectionSummary(BaseModel):
    id: str
    channel: ChannelType
    external_id: str
    page_name: str
    status: ConnectionStatus
    connected_at: datetime


class ChannelConnectionsListResponse(BaseModel):
    connections: list[ChannelConnectionSummary]
