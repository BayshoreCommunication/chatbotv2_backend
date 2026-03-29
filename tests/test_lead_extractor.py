import unittest

from services.chatbot.lead_extractor import extract_lead_info


class LeadExtractorTests(unittest.TestCase):
    def test_rejects_iso_like_dates_as_phone(self) -> None:
        info = extract_lead_info("It happened on 2026-02-24.")
        self.assertIsNone(info.phone)

    def test_rejects_dotted_dates_as_phone(self) -> None:
        info = extract_lead_info("Date is 2025.11.12")
        self.assertIsNone(info.phone)

    def test_does_not_capture_at_as_name(self) -> None:
        info = extract_lead_info("Call me at (555) 123-4567")
        self.assertIsNone(info.name)
        self.assertEqual(info.phone, "5551234567")

    def test_extracts_valid_name_and_phone(self) -> None:
        info = extract_lead_info("My name is John Doe and my number is +1 555-222-3333")
        self.assertEqual(info.name, "John Doe")
        self.assertEqual(info.phone, "+15552223333")

    def test_extracts_lowercase_name_without_is(self) -> None:
        info = extract_lead_info("my name sahak and phone is 0147852369")
        self.assertEqual(info.name, "Sahak")
        self.assertEqual(info.phone, "0147852369")


if __name__ == "__main__":
    unittest.main()
