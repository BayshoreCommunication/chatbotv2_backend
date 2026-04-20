from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── Database ──────────────────────────────────────────────────────────────
    MONGODB_URL: str
    DATABASE_NAME: str

    # ── JWT ───────────────────────────────────────────────────────────────────
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    # ── SMTP Email ────────────────────────────────────────────────────────────
    SMTP_HOST: str
    SMTP_PORT: int
    SMTP_MAIL: str
    SMTP_PASSWORD: str
    FRONTEND_URL: str

    # ── OpenAI ────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str

    # ── Pinecone ──────────────────────────────────────────────────────────────
    PINECONE_API_KEY: str
    PINECONE_ENV: str
    PINECONE_INDEX: str

    # ── Stripe ────────────────────────────────────────────────────────────────
    PUBLISHABLE_KEY:        str = ""
    STRIPE_SECRET_KEY:      str = ""
    STRIPE_WEBHOOK_SECRET:  str = ""

    # ── Calendly ──────────────────────────────────────────────────────────────
    CALENDLY_API_KEY: str

    # ── DigitalOcean Spaces ───────────────────────────────────────────────────
    DO_SPACES_KEY: str = ""
    DO_SPACES_SECRET: str = ""
    DO_SPACES_ENDPOINT: str = ""
    DO_SPACES_BUCKET: str = ""
    DO_SPACES_REGION: str = ""
    DO_SPACES_CDN_URL: str = ""
    DO_FOLDER_NAME: str = "uploads"


settings = Settings()
