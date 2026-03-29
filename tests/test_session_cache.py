"""
tests/test_session_cache.py
────────────────────────────
Unit tests for services/chatbot/session_cache.py.
No external dependencies — all in-process.
"""

import asyncio
import time
import unittest

from services.chatbot.session_cache import (
    SessionData,
    _session_store,
    create_or_refresh_session,
    get_session,
    invalidate_session,
    update_session_lead,
)

COMPANY_ID = "aabbccddeeff001122334455"
SESSION_ID = "test-session-001"
THREAD_ID  = f"{COMPANY_ID}:{SESSION_ID}"
CTX        = {"company_name": "Test Firm", "company_type": "law-firm", "is_trained": True}


def _clear() -> None:
    """Reset the module-level store between tests."""
    _session_store.clear()


class TestGetSession(unittest.TestCase):
    def setUp(self) -> None:
        _clear()

    def test_miss_on_empty_store(self) -> None:
        self.assertIsNone(get_session(THREAD_ID))

    def test_hit_after_create(self) -> None:
        asyncio.get_event_loop().run_until_complete(
            create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX)
        )
        session = get_session(THREAD_ID)
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.thread_id, THREAD_ID)
        self.assertEqual(session.company_id, COMPANY_ID)

    def test_miss_after_ttl_expires(self) -> None:
        asyncio.get_event_loop().run_until_complete(
            create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX)
        )
        # Manually expire the session
        _session_store[THREAD_ID].expires_at = time.monotonic() - 1
        self.assertIsNone(get_session(THREAD_ID))


class TestCreateOrRefreshSession(unittest.TestCase):
    def setUp(self) -> None:
        _clear()

    def test_creates_new_session(self) -> None:
        session = asyncio.get_event_loop().run_until_complete(
            create_or_refresh_session(
                THREAD_ID, COMPANY_ID, CTX,
                lead_state={
                    "lead_captured": True,
                    "lead_name": "Sahak",
                    "lead_phone": "0178523694",
                    "lead_email": None,
                },
                user_timezone="Asia/Dhaka",
            )
        )
        self.assertEqual(session.lead_name, "Sahak")
        self.assertEqual(session.lead_phone, "0178523694")
        self.assertTrue(session.lead_captured)
        self.assertEqual(session.user_timezone, "Asia/Dhaka")

    def test_refresh_returns_same_object(self) -> None:
        s1 = asyncio.get_event_loop().run_until_complete(
            create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX)
        )
        s2 = asyncio.get_event_loop().run_until_complete(
            create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX)
        )
        self.assertIs(s1, s2)

    def test_refresh_adds_timezone(self) -> None:
        asyncio.get_event_loop().run_until_complete(
            create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX, user_timezone=None)
        )
        session = asyncio.get_event_loop().run_until_complete(
            create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX, user_timezone="UTC")
        )
        self.assertEqual(session.user_timezone, "UTC")


class TestUpdateSessionLead(unittest.TestCase):
    def setUp(self) -> None:
        _clear()

    def test_merges_lead_info(self) -> None:
        asyncio.get_event_loop().run_until_complete(
            create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX)
        )
        ok = update_session_lead(THREAD_ID, name="Sahak", phone="0178523694")
        self.assertTrue(ok)
        session = get_session(THREAD_ID)
        assert session is not None
        self.assertEqual(session.lead_name, "Sahak")
        self.assertEqual(session.lead_phone, "0178523694")
        self.assertTrue(session.lead_captured)

    def test_does_not_overwrite_existing_lead(self) -> None:
        asyncio.get_event_loop().run_until_complete(
            create_or_refresh_session(
                THREAD_ID, COMPANY_ID, CTX,
                lead_state={"lead_captured": True, "lead_name": "Sahak",
                            "lead_phone": "0178523694", "lead_email": None},
            )
        )
        # Try to overwrite name with a new value — should be ignored
        update_session_lead(THREAD_ID, name="OtherName")
        session = get_session(THREAD_ID)
        assert session is not None
        self.assertEqual(session.lead_name, "Sahak")

    def test_returns_false_on_cache_miss(self) -> None:
        ok = update_session_lead("no-such-thread", name="X")
        self.assertFalse(ok)

    def test_resets_ttl(self) -> None:
        asyncio.get_event_loop().run_until_complete(
            create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX)
        )
        before = _session_store[THREAD_ID].expires_at
        time.sleep(0.01)
        update_session_lead(THREAD_ID, name="Sahak")
        after = _session_store[THREAD_ID].expires_at
        self.assertGreater(after, before)


class TestInvalidateSession(unittest.TestCase):
    def setUp(self) -> None:
        _clear()

    def test_evicts_session(self) -> None:
        asyncio.get_event_loop().run_until_complete(
            create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX)
        )
        self.assertIsNotNone(get_session(THREAD_ID))
        invalidate_session(THREAD_ID)
        self.assertIsNone(get_session(THREAD_ID))

    def test_noop_on_missing(self) -> None:
        # Should not raise
        invalidate_session("unknown-thread")


class TestToLeadDict(unittest.TestCase):
    def test_returns_correct_keys(self) -> None:
        s = SessionData(
            thread_id=THREAD_ID,
            company_id=COMPANY_ID,
            lead_name="Sahak",
            lead_phone="0178523694",
            lead_email=None,
            lead_captured=True,
        )
        d = s.to_lead_dict()
        self.assertEqual(d["lead_name"], "Sahak")
        self.assertEqual(d["lead_phone"], "0178523694")
        self.assertIsNone(d["lead_email"])
        self.assertTrue(d["lead_captured"])


if __name__ == "__main__":
    unittest.main()
