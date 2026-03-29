from datetime import datetime
from pydantic import BaseModel, Field


class CalendlySettings(BaseModel):
    calendly_url: str = ""
    calendly_access_token: str = ""
    event_type_uri: str = ""
    auto_embed: bool = True


class CalendlySettingsResponse(BaseModel):
    settings: CalendlySettings


class CalendlyTestConnectionRequest(BaseModel):
    access_token: str = Field(..., min_length=1)


class CalendlyTestConnectionResponse(BaseModel):
    valid: bool


class CalendlyEvent(BaseModel):
    uri: str
    name: str
    duration: int
    status: str
    booking_url: str


class CalendlyEventsResponse(BaseModel):
    events: list[CalendlyEvent]


class CalendlySlot(BaseModel):
    start_time: str
    scheduling_url: str


class CalendlyAvailabilityResponse(BaseModel):
    slots: list[CalendlySlot]


class CalendlyStats(BaseModel):
    total_events: int = 0
    active_events: int = 0
    upcoming_bookings: int = 0


class CalendlyStatsResponse(BaseModel):
    stats: CalendlyStats


class AppointmentSettingsDoc(BaseModel):
    user_id: str
    calendly_url: str = ""
    calendly_access_token: str = ""
    event_type_uri: str = ""
    auto_embed: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
