"""Unit tests for deterministic feedback input helpers.

Run from 26535016 with:
    python -m unittest discover -s tests
"""
import unittest

from feedback_analyzer import FeedbackAnalyzer, clean_text


class FeedbackTextTests(unittest.TestCase):
    def test_normalizes_case_whitespace_and_urls(self):
        self.assertEqual(
            clean_text("  GREAT   work! https://example.com/path  "),
            "great work!",
        )

    def test_converts_non_string_values_for_cleaning(self):
        self.assertEqual(clean_text(123), "123")


class ReviewValidationTests(unittest.TestCase):
    def test_strips_outer_whitespace(self):
        self.assertEqual(FeedbackAnalyzer._validate("  Good work  "), "Good work")

    def test_rejects_short_review(self):
        with self.assertRaises(ValueError):
            FeedbackAnalyzer._validate("ok")

    def test_rejects_non_string_review(self):
        with self.assertRaises(ValueError):
            FeedbackAnalyzer._validate(None)

    def test_caps_review_length(self):
        self.assertEqual(len(FeedbackAnalyzer._validate("x" * 2500)), 2000)


if __name__ == "__main__":
    unittest.main()
