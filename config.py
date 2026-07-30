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

    # ── Email (Resend) ────────────────────────────────────────────────────────
    RESEND_API_KEY: str
    RESEND_FROM_EMAIL: str = "onboarding@resend.dev"
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
    # Publicly reachable base URL for THIS backend (e.g. https://api.example.com)
    # — used to register the Calendly webhook callback. Left blank in local dev,
    # where Calendly can't reach localhost; webhook registration is skipped
    # (logged, not fatal) until this is set.
    BACKEND_PUBLIC_URL: str = ""

    # ── Meta (WhatsApp / Messenger / Instagram) ──────────────────────────────
    # Shared across all companies — one Meta Developer App serves every
    # connected number/page. Per-company credentials (phone_number_id,
    # access_token, business_account_id) live in Mongo instead, see
    # model/apps_integration.py.
    META_APP_ID: str = ""
    META_APP_SECRET: str = ""
    META_WEBHOOK_VERIFY_TOKEN: str = ""
    # Base URL this backend calls itself on to reach POST /chat/{company_id}
    # from the WhatsApp webhook — always localhost, not BACKEND_PUBLIC_URL,
    # since this is a same-process/same-machine call, not Meta calling us.
    INTERNAL_API_BASE_URL: str = "http://127.0.0.1:8000"

    # ── DigitalOcean Spaces ───────────────────────────────────────────────────
    DO_SPACES_KEY: str = ""
    DO_SPACES_SECRET: str = ""
    DO_SPACES_ENDPOINT: str = ""
    DO_SPACES_BUCKET: str = ""
    DO_SPACES_REGION: str = ""
    DO_SPACES_CDN_URL: str = ""
    DO_FOLDER_NAME: str = "uploads"

    # ── Super Admin (fixed account, seeded at startup) ───────────────────────
    SUPER_ADMIN_EMAIL: str = "superadmin@gmail.com"
    SUPER_ADMIN_PASSWORD: str = "superadmin@123"


settings = Settings()
