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

    # ── Meta (Messenger, later Instagram) ─────────────────────────────────────
    # Shared across all companies — one Meta Developer App serves every
    # connected page. Per-company credentials (page_id, access_token) live
    # in Mongo instead, see model/apps_integration_model.py.
    META_APP_ID: str = ""
    META_APP_SECRET: str = ""
    META_WEBHOOK_VERIFY_TOKEN: str = ""
    META_GRAPH_API_VERSION: str = "v21.0"
    # Not currently read anywhere in the backend — reserved for the OAuth
    # connect flow being built against ChannelConnectionDoc.
    META_OAUTH_REDIRECT_URI: str = ""
    # Base URL this backend calls itself on to reach POST /chat/{company_id}
    # from a channel webhook — always localhost, not BACKEND_PUBLIC_URL,
    # since this is a same-process/same-machine call, not Meta calling us.
    INTERNAL_API_BASE_URL: str = "http://127.0.0.1:8000"
    # 64 hex chars (32 raw bytes) — AES-256-GCM key for encrypting connected
    # channels' access tokens at rest (see utils/token_encryption.py).
    # Generate with: openssl rand -hex 32. Losing/rotating this makes every
    # already-stored token undecryptable, so back it up somewhere durable.
    TOKEN_ENCRYPTION_KEY: str = ""

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
