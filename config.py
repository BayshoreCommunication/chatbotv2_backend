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

    # ── Calendly ──────────────────────────────────────────────────────────────
    CALENDLY_API_KEY: str


settings = Settings()
