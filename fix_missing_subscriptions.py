"""
fix_missing_subscriptions.py
────────────────────────────
Creates a free-tier subscription record for any user that has no subscription
document in MongoDB. Run once on the server after fixing the Stripe webhook.

Usage:
    cd /var/www/chatbot_backend
    source .venv/bin/activate
    python3 fix_missing_subscriptions.py
"""
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Load .env from the same directory as this script
load_dotenv(Path(__file__).parent / ".env")

MONGODB_URL   = os.environ["MONGODB_URL"]
DATABASE_NAME = os.environ["DATABASE_NAME"]

CONVERSATION_LIMITS = {
    "free":         1000,
    "professional": 5000,
    "advanced":     None,
    "enterprise":   None,
}


async def main() -> None:
    client = AsyncIOMotorClient(MONGODB_URL)
    db     = client[DATABASE_NAME]

    # Fetch all user IDs that already have a subscription record
    existing_subs = await db["subscriptions"].distinct("company_id")
    existing_set  = set(existing_subs)

    # Find all non-admin users with no subscription
    cursor   = db["users"].find({"role": {"$ne": "admin"}}, {"_id": 1, "email": 1, "company_name": 1})
    users    = await cursor.to_list(length=None)

    missing  = [u for u in users if str(u["_id"]) not in existing_set]

    if not missing:
        print("✓ No missing subscriptions — all users already have a record.")
        return

    print(f"Found {len(missing)} user(s) with no subscription record:\n")
    for u in missing:
        print(f"  {u['_id']}  {u.get('email', '?')}  ({u.get('company_name', '?')})")

    print("\nCreating free-tier subscription records …")
    now = datetime.now(timezone.utc)

    for u in missing:
        company_id = str(u["_id"])
        doc = {
            "company_id":             company_id,
            "stripe_customer_id":     None,
            "stripe_subscription_id": None,
            "subscription_tier":      "free",
            "billing_cycle":          "monthly",
            "payment_amount":         0.0,
            "currency":               "usd",
            "conversation_limit":     CONVERSATION_LIMITS["free"],
            "subscription_status":    "active",
            "cancel_at_period_end":   False,
            "current_period_start":   now,
            "current_period_end":     None,
            "trial_start":            None,
            "trial_end":              None,
            "free_trial_used":        False,
            "conversations_used":     0,
            "created_at":             now,
            "updated_at":             now,
        }
        await db["subscriptions"].update_one(
            {"company_id": company_id},
            {"$setOnInsert": doc},
            upsert=True,
        )
        print(f"  ✓ Created free-tier record for {u.get('email', company_id)}")

    print("\nDone. Restart the service if needed: systemctl restart chatbot_backend")
    client.close()


if __name__ == "__main__":
    asyncio.run(main())
