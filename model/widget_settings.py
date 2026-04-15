from typing import Optional
from pydantic import BaseModel, Field


class ThemeSettings(BaseModel):
    primary_color: str = "#807045"
    font_family: str = "Inter"


class BehaviorSettings(BaseModel):
    auto_open: bool = False
    open_delay: int = 2000
    show_welcome_message: bool = True


class ContentSettings(BaseModel):
    welcome_message: str = "Hello! 👋 Welcome to Bayshore Communication. How can I assist you today?"
    welcome_video: str = "https://www.youtube.com/embed/dQw4w9WgXcQ"
    welcome_video_autoplay: bool = True
    input_placeholder: str = "Type your question here..."


class LauncherSettings(BaseModel):
    position: str = "bottom-right"
    icon_style: str = "default"
    show_bubbles: bool = True
    brand_image_url: str = "https://i.ibb.co.com/pvcGHgy9/ocrun0jnwtssa3pbky9y.webp"


class WidgetSettingsModel(BaseModel):
    bot_name: str = "BayAI Assistant"
    theme: ThemeSettings = Field(default_factory=ThemeSettings)
    behavior: BehaviorSettings = Field(default_factory=BehaviorSettings)
    content: ContentSettings = Field(default_factory=ContentSettings)
    launcher: LauncherSettings = Field(default_factory=LauncherSettings)


class WidgetSettingsResponse(WidgetSettingsModel):
    id: Optional[str] = None
    company_id: str
