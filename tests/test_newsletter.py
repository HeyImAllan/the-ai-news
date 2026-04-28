import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src import newsletter


class MainTests(unittest.TestCase):
    @patch("src.newsletter.compact_previous_month_news")
    @patch("src.newsletter.save_today_markdown")
    @patch("src.newsletter.save_newsletter")
    @patch("src.newsletter.wrap_newsletter")
    @patch("src.newsletter.generate_newsletter")
    @patch("src.newsletter.fetch_all_articles")
    @patch("src.newsletter.OpenAI")
    def test_main_skips_generation_when_no_articles(
        self,
        mock_openai,
        mock_fetch_all_articles,
        mock_generate_newsletter,
        mock_wrap_newsletter,
        mock_save_newsletter,
        mock_save_today_markdown,
        mock_compact_previous_month_news,
    ):
        mock_fetch_all_articles.return_value = ([], {})

        with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}, clear=True):
            with self.assertRaises(SystemExit) as context:
                newsletter.main()

        self.assertEqual(context.exception.code, 0)
        mock_openai.assert_called_once()
        mock_generate_newsletter.assert_not_called()
        mock_wrap_newsletter.assert_not_called()
        mock_save_newsletter.assert_not_called()
        mock_save_today_markdown.assert_not_called()
        mock_compact_previous_month_news.assert_called_once()

    @patch("src.newsletter.compact_previous_month_news")
    @patch("src.newsletter.save_today_markdown")
    @patch("src.newsletter.save_newsletter")
    @patch("src.newsletter.wrap_newsletter")
    @patch("src.newsletter.generate_newsletter")
    @patch("src.newsletter.fetch_all_articles")
    @patch("src.newsletter.OpenAI")
    def test_main_generates_newsletter_when_articles_exist(
        self,
        mock_openai,
        mock_fetch_all_articles,
        mock_generate_newsletter,
        mock_wrap_newsletter,
        mock_save_newsletter,
        mock_save_today_markdown,
        mock_compact_previous_month_news,
    ):
        articles = [{"source": "A", "title": "T", "url": "U", "summary": "S", "published": ""}]
        mock_fetch_all_articles.return_value = (articles, {})
        mock_generate_newsletter.return_value = "body"
        mock_wrap_newsletter.return_value = "wrapped"

        with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}, clear=True):
            newsletter.main()

        mock_openai.assert_called_once()
        mock_generate_newsletter.assert_called_once_with(articles, mock_openai.return_value)
        mock_wrap_newsletter.assert_called_once_with("body", len(articles), articles, {})
        mock_save_newsletter.assert_called_once_with("wrapped")
        mock_save_today_markdown.assert_called_once_with("wrapped")
        mock_compact_previous_month_news.assert_called_once()

    def test_main_exits_when_github_token_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit) as context:
                newsletter.main()

        self.assertEqual(context.exception.code, 1)


class OutputHelpersTests(unittest.TestCase):
    def test_normalize_newsletter_body_enforces_required_sections(self):
        normalized = newsletter.normalize_newsletter_body(
            """# Temporary title

## Today’s Highlights
Summary paragraph.
"""
        )

        self.assertNotIn("# Temporary title", normalized)
        self.assertIn("## ✨ Today's Highlights", normalized)
        self.assertIn("## 🚀 What Changed Today", normalized)
        self.assertIn("## 📚 Deep Dive by Theme", normalized)
        self.assertIn("## ✅ Key Takeaways", normalized)

    def test_build_prompt_includes_consistent_modern_template_sections(self):
        prompt = newsletter.build_prompt(
            [{"source": "A", "title": "T", "url": "U", "summary": "S", "published": ""}]
        )

        self.assertIn("## ✨ Today's Highlights", prompt)
        self.assertIn("## 🚀 What Changed Today", prompt)
        self.assertIn("## 📚 Deep Dive by Theme", prompt)
        self.assertIn("## ✅ Key Takeaways", prompt)
        self.assertIn("Do not add a top-level H1 title", prompt)

    def test_save_today_markdown_writes_to_expected_path(self):
        with TemporaryDirectory() as tmp_dir:
            today_path = Path(tmp_dir) / "TODAY.MD"
            result = newsletter.save_today_markdown("sample", today_file=today_path)

            self.assertEqual(result, today_path)
            self.assertEqual(today_path.read_text(encoding="utf-8"), "sample")

    def test_compact_previous_month_news_creates_overview_on_first_day(self):
        with TemporaryDirectory() as tmp_dir:
            newsletters_dir = Path(tmp_dir)
            (newsletters_dir / "2026-03-01.md").write_text(
                "day one: launch news", encoding="utf-8"
            )
            (newsletters_dir / "2026-03-15.md").write_text(
                "day fifteen: model updates", encoding="utf-8"
            )
            (newsletters_dir / "2026-03-31.md").write_text(
                "day thirty-one: security release", encoding="utf-8"
            )

            result = newsletter.compact_previous_month_news(
                now=datetime(2026, 4, 1, tzinfo=timezone.utc),
                newsletters_dir=newsletters_dir,
            )

            self.assertEqual(result, newsletters_dir / "2026-03.md")
            self.assertTrue(result.exists())
            content = result.read_text(encoding="utf-8")
            self.assertIn("# 📅 March 2026 News Overview", content)
            self.assertIn("## 2026-03-01", content)
            self.assertIn("## 2026-03-15", content)
            self.assertIn("## 2026-03-31", content)
            self.assertIn("day one: launch news", content)
            self.assertIn("day fifteen: model updates", content)
            self.assertIn("day thirty-one: security release", content)

    def test_compact_previous_month_news_skips_when_not_first_day(self):
        with TemporaryDirectory() as tmp_dir:
            newsletters_dir = Path(tmp_dir)
            (newsletters_dir / "2026-03-01.md").write_text("a", encoding="utf-8")

            result = newsletter.compact_previous_month_news(
                now=datetime(2026, 4, 2, tzinfo=timezone.utc),
                newsletters_dir=newsletters_dir,
            )

            self.assertIsNone(result)
            self.assertFalse((newsletters_dir / "2026-03.md").exists())


if __name__ == "__main__":
    unittest.main()
