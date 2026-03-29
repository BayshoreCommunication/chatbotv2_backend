# FastAPI + MongoDB

## 🚀 Project Setup

**Stack:** FastAPI · Motor (async MongoDB) · Pydantic v2 · uv

---

## 📁 Project Structure

```
fastapi-mongo-app/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app entry point
│   ├── config.py          # Settings from .env
│   ├── database.py        # MongoDB connection (Motor)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── user.py        # Pydantic request/response schemas
│   ├── services/
│   │   ├── __init__.py
│   │   └── user_service.py  # Business logic (CRUD operations)
│   └── routers/
│       ├── __init__.py
│       └── user_router.py   # FastAPI router (HTTP endpoints)
├── .env                   # Environment variables
├── .env.example           # Template for .env
├── pyproject.toml         # uv/pip dependencies
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Install uv (if not already installed)
pip install uv

# 2. Create virtual environment & install dependencies
uv venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
uv pip install -e .

# 3. Copy env file and update your values
copy .env.example .env
```

---

## 🏃 Running the Server

```bash
uvicorn app.main:app --reload
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/api/v1/users/` | Create user |
| `GET` | `/api/v1/users/` | List all users (paginated) |
| `GET` | `/api/v1/users/{id}` | Get user by ID |
| `PATCH` | `/api/v1/users/{id}` | Update user (partial) |
| `DELETE` | `/api/v1/users/{id}` | Delete user |

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_URL` | `mongodb://localhost:27017` | MongoDB connection string |
| `DATABASE_NAME` | `fastapi_mongo_db` | MongoDB database name |
| `SECRET_KEY` | `change-this` | JWT secret (for future auth) |
| `ALGORITHM` | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Token expiry |
