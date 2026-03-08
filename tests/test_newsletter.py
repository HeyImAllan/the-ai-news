import unittest
from unittest.mock import patch

from src import newsletter


class MainTests(unittest.TestCase):
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
    ):
        articles = [{"source": "A", "title": "T", "url": "U", "summary": "S", "published": ""}]
        mock_fetch_all_articles.return_value = (articles, {})
        mock_generate_newsletter.return_value = "body"
        mock_wrap_newsletter.return_value = "wrapped"

        with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}, clear=True):
            newsletter.main()

        mock_openai.assert_called_once()
        mock_generate_newsletter.assert_called_once_with(articles, mock_openai.return_value)
        mock_wrap_newsletter.assert_called_once_with("body", 1, articles, {})
        mock_save_newsletter.assert_called_once_with("wrapped")

    def test_main_exits_when_github_token_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit) as context:
                newsletter.main()

        self.assertEqual(context.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
