import unittest

from routers.chat_router import (
    _build_post_lead_capture_reply_if_needed,
    _looks_like_contact_share_message,
)
from services.chatbot.lead_extractor import extract_lead_info


class ChatRouterFlowTests(unittest.TestCase):
    def test_detects_contact_share_message(self) -> None:
        self.assertTrue(_looks_like_contact_share_message("my name is sahak my phone is 0147852369"))
        self.assertFalse(_looks_like_contact_share_message("yesterday"))

    def test_short_circuit_reply_after_new_lead_capture(self) -> None:
        text = "my name is sahak my phone is 0147852369"
        lead = extract_lead_info(text)
        reply = _build_post_lead_capture_reply_if_needed(
            message=text,
            session_lead_captured=False,
            message_lead=lead,
            effective_lead_name=lead.name,
            effective_lead_phone=lead.phone,
            effective_lead_email=lead.email,
        )
        self.assertIsNotNone(reply)
        self.assertIn("I have your details", reply or "")
        self.assertIn("15-30 minutes", reply or "")

    def test_no_short_circuit_when_already_captured(self) -> None:
        text = "my name is sahak my phone is 0147852369"
        lead = extract_lead_info(text)
        reply = _build_post_lead_capture_reply_if_needed(
            message=text,
            session_lead_captured=True,
            message_lead=lead,
            effective_lead_name=lead.name,
            effective_lead_phone=lead.phone,
            effective_lead_email=lead.email,
        )
        self.assertIsNone(reply)


if __name__ == "__main__":
    unittest.main()
