"""Small unit tests for deterministic feedback-processing helpers.

Run from the 26535016 directory with: python -m unittest discover -s tests
"""
import unittest

from feedback_analyzer import FeedbackAnalyzer, clean_text


class CleanTextTests(unittest.TestCase):
    def test_lowercases_removes_urls_and_normalizes_whitespace(self):
        self.assertEqual(
            clean_text("  GREAT   work! https://example.com/path  "),
            "great work!",
        )

    def test_converts_non_string_input_for_normalization(self):
        self.assertEqual(clean_text(123), "123")


class InputValidationTests(unittest.TestCase):
    def test_strips_outer_whitespace(self):
        self.assertEqual(FeedbackAnalyzer._validate("  Good work  "), "Good work")

    def test_rejects_too_short_review(self):
        with self.assertRaises(ValueError):
            FeedbackAnalyzer._validate("ok")

    def test_rejects_non_string_review(self):
        with self.assertRaises(ValueError):
            FeedbackAnalyzer._validate(None)

    def test_truncates_very_long_review(self):
        self.assertEqual(len(FeedbackAnalyzer._validate("x" * 2500)), 2000)


if __name__ == "__main__":
    unittest.main()
