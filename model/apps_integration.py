"""
model/apps_integration.py
──────────────────────────
Settings models for third-party channel integrations (WhatsApp, Messenger,
and later Instagram) shown on the dashboard's "Apps & Integrations" page.
One file for all channels — each platform gets its own settings classes
below, following the same shape (settings + connect request + embedded/
login-for-business connect request + test-connection response + Mongo doc),
so Instagram can be added as a sibling here without a new file per platform.
"""

from datetime import datetime
from pydantic import BaseModel, Field


# ── WhatsApp ──────────────────────────────────────────────────────────────────

class WhatsAppSettings(BaseModel):
    phone_number_id: str = ""
    access_token: str = ""
    business_account_id: str = ""
    verified_name: str = ""
    display_phone_number: str = ""


class WhatsAppSettingsResponse(BaseModel):
    settings: WhatsAppSettings
    connected: bool


class WhatsAppConnectRequest(BaseModel):
    phone_number_id: str = Field(..., min_length=1)
    access_token: str = Field(..., min_length=1)
    business_account_id: str = Field(..., min_length=1)


# Embedded Signup ("Continue with Facebook") — the frontend gets a short-lived
# `code` from FB.login() plus the phone_number_id/waba_id Meta posts via
# window.postMessage during the flow. We never see a manually-copied access
# token here; the backend exchanges `code` for a long-lived token itself.
class WhatsAppEmbeddedConnectRequest(BaseModel):
    code: str = Field(..., min_length=1)
    phone_number_id: str = Field(..., min_length=1)
    business_account_id: str = Field(..., min_length=1)


class WhatsAppTestConnectionResponse(BaseModel):
    valid: bool
    verified_name: str = ""
    display_phone_number: str = ""


class WhatsAppSettingsDoc(BaseModel):
    company_id: str
    phone_number_id: str = ""
    access_token: str = ""
    business_account_id: str = ""
    verified_name: str = ""
    display_phone_number: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


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
# picked Page here the way it does phone_number_id/waba_id for WhatsApp, so
# `page_id` is optional — the backend auto-selects it when the login only
# grants access to one Page, and only needs page_id to disambiguate when
# there's more than one. We never see a manually-copied Page Access Token
# here; the backend exchanges `code` for a long-lived user token itself, then
# reads the matching Page Access Token off GET /me/accounts.
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
