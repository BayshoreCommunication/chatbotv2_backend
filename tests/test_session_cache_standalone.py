"""
Standalone isolated test for session_cache.py.
Does NOT import via packages — loads the file directly via importlib
so NO config.py / DB / LLM imports are triggered.

Run: .venv\Scripts\python.exe tests\test_session_cache_standalone.py
"""
import asyncio
import importlib.util
import os
import time
import unittest

# --- Load session_cache.py directly without touching __init__.py ---
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SC_PATH = os.path.join(_BASE, "services", "chatbot", "session_cache.py")

spec = importlib.util.spec_from_file_location("session_cache_mod", _SC_PATH)
sc = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
spec.loader.exec_module(sc)                   # type: ignore[union-attr]

SessionData               = sc.SessionData
_session_store            = sc._session_store
_session_locks            = sc._session_locks
create_or_refresh_session = sc.create_or_refresh_session
get_session               = sc.get_session
invalidate_session        = sc.invalidate_session
update_session_lead       = sc.update_session_lead

# --- Constants ---
COMPANY_ID = "aabbccddeeff001122334455"
THREAD_ID  = f"{COMPANY_ID}:test-session-001"
CTX        = {"company_name": "Test Firm", "company_type": "law-firm", "is_trained": True}


def _clear() -> None:
    """Reset store and locks between tests to avoid cross-test state."""
    _session_store.clear()
    _session_locks.clear()


# --- Tests ---

class TestGetSession(unittest.TestCase):
    def setUp(self) -> None:
        _clear()

    def test_miss_on_empty_store(self) -> None:
        self.assertIsNone(get_session(THREAD_ID))

    def test_hit_after_create(self) -> None:
        asyncio.run(create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX))
        session = get_session(THREAD_ID)
        self.assertIsNotNone(session)
        self.assertEqual(session.thread_id, THREAD_ID)

    def test_miss_after_ttl_expires(self) -> None:
        asyncio.run(create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX))
        _session_store[THREAD_ID].expires_at = time.monotonic() - 1
        self.assertIsNone(get_session(THREAD_ID))


class TestCreateOrRefresh(unittest.TestCase):
    def setUp(self) -> None:
        _clear()

    def test_creates_with_lead_state(self) -> None:
        session = asyncio.run(
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

    def test_refresh_keeps_same_entry(self) -> None:
        asyncio.run(create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX))
        entry1_id = id(_session_store[THREAD_ID])
        _session_locks.clear()  # clear so second run creates a fresh lock
        asyncio.run(create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX))
        entry2_id = id(_session_store[THREAD_ID])
        self.assertEqual(entry1_id, entry2_id)

    def test_refresh_adds_timezone_if_missing(self) -> None:
        asyncio.run(create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX, user_timezone=None))
        _session_locks.clear()
        session = asyncio.run(
            create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX, user_timezone="UTC")
        )
        self.assertEqual(session.user_timezone, "UTC")


class TestUpdateSessionLead(unittest.TestCase):
    def setUp(self) -> None:
        _clear()

    def test_merges_lead_info(self) -> None:
        asyncio.run(create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX))
        ok = update_session_lead(THREAD_ID, name="Sahak", phone="0178523694")
        self.assertTrue(ok)
        s = get_session(THREAD_ID)
        self.assertEqual(s.lead_name, "Sahak")
        self.assertEqual(s.lead_phone, "0178523694")
        self.assertTrue(s.lead_captured)

    def test_does_not_overwrite_existing_name(self) -> None:
        asyncio.run(
            create_or_refresh_session(
                THREAD_ID, COMPANY_ID, CTX,
                lead_state={"lead_captured": True, "lead_name": "Sahak",
                            "lead_phone": "0178523694", "lead_email": None},
            )
        )
        update_session_lead(THREAD_ID, name="OtherName")
        self.assertEqual(get_session(THREAD_ID).lead_name, "Sahak")

    def test_returns_false_on_miss(self) -> None:
        self.assertFalse(update_session_lead("no-such-thread", name="X"))

    def test_resets_ttl(self) -> None:
        asyncio.run(create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX))
        before = _session_store[THREAD_ID].expires_at
        time.sleep(0.01)
        update_session_lead(THREAD_ID, name="Sahak")
        self.assertGreater(_session_store[THREAD_ID].expires_at, before)


class TestInvalidate(unittest.TestCase):
    def setUp(self) -> None:
        _clear()

    def test_evicts_session(self) -> None:
        asyncio.run(create_or_refresh_session(THREAD_ID, COMPANY_ID, CTX))
        invalidate_session(THREAD_ID)
        self.assertIsNone(get_session(THREAD_ID))

    def test_noop_on_missing(self) -> None:
        invalidate_session("unknown-thread")  # must not raise


class TestToLeadDict(unittest.TestCase):
    def test_correct_keys(self) -> None:
        s = SessionData(
            thread_id=THREAD_ID, company_id=COMPANY_ID,
            lead_name="Sahak", lead_phone="0178523694",
            lead_email=None, lead_captured=True,
        )
        d = s.to_lead_dict()
        self.assertEqual(d["lead_name"], "Sahak")
        self.assertEqual(d["lead_phone"], "0178523694")
        self.assertIsNone(d["lead_email"])
        self.assertTrue(d["lead_captured"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
