"""
AI News Newsletter Generator

Crawls a curated list of GitHub blogs and AI news pages, then uses the
GitHub Models inference API (no external subscription required) to analyze
and summarize the articles into a Markdown newsletter.

Usage:
    python src/newsletter.py

Required environment variable:
    GITHUB_TOKEN             - GitHub token (automatically set in Actions)

Optional environment variable:
    MAX_ARTICLES_PER_SOURCE  - Maximum articles to fetch per source (default: 5)
    GITHUB_MODEL             - GitHub Models model to use (default: gpt-5)
"""

import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

SOURCES = [
    {
        "name": "GitHub Blog",
        "url": "https://github.blog/feed/",
        "type": "rss",
    },
    {
        "name": "GitHub Changelog – Copilot",
        "url": "https://github.blog/changelog/label/copilot/feed/",
        "type": "rss",
    },
    {
        "name": "GitHub Changelog – Actions",
        "url": "https://github.blog/tag/github-actions/feed/",
        "type": "rss",
    },
    {
        "name": "OpenAI News",
        "url": "https://openai.com/news/rss.xml",
        "type": "rss",
    },
    {
        "name": "Anthropic News",
        # /rss.xml returns 404; feedparser auto-discovers any RSS link from the HTML page
        "url": "https://www.anthropic.com/news",
        "type": "rss",
    },
    {
        "name": "Google DeepMind Blog",
        "url": "https://deepmind.google/blog/rss.xml",
        "type": "rss",
    },
    {
        "name": "Microsoft AI Blog",
        "url": "https://blogs.microsoft.com/ai/feed/",
        "type": "rss",
    },
    {
        "name": "HuggingFace Blog",
        "url": "https://huggingface.co/blog/feed.xml",
        "type": "rss",
    },
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_ARTICLES_PER_SOURCE = int(os.environ.get("MAX_ARTICLES_PER_SOURCE", "5"))
GITHUB_MODEL = os.environ.get("GITHUB_MODEL", "gpt-5")
NEWSLETTERS_DIR = Path(__file__).parent.parent / "newsletters"

REQUEST_TIMEOUT = 15  # seconds
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; the-ai-news-bot/1.0; "
        "+https://github.com/HeyImAllan/the-ai-news)"
    )
}


# ---------------------------------------------------------------------------
# Fetching helpers
# ---------------------------------------------------------------------------


def fetch_rss(source: dict) -> tuple[list[dict], str | None]:
    """Parse an RSS/Atom feed and return (articles, error_message)."""
    try:
        response = requests.get(
            source["url"], headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        articles = []
        for entry in feed.entries[:MAX_ARTICLES_PER_SOURCE]:
            summary = entry.get("summary", "")
            # Strip HTML tags from summary
            if summary:
                summary = BeautifulSoup(summary, "lxml").get_text(
                    separator=" ", strip=True
                )
            articles.append(
                {
                    "source": source["name"],
                    "title": entry.get("title", "").strip(),
                    "url": entry.get("link", ""),
                    "summary": summary[:500] if summary else "",
                    "published": entry.get("published", ""),
                }
            )
        print(
            f"  ✓ {source['name']}: fetched {len(articles)} article(s)",
            flush=True,
        )
        return articles, None
    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)
        print(f"  ✗ {source['name']}: {error_msg}", flush=True)
        return [], error_msg


def fetch_all_articles() -> tuple[list[dict], dict[str, str]]:
    """Fetch articles from all configured sources.

    Returns a tuple of (articles, failed_sources) where failed_sources maps
    source name to error message.
    """
    all_articles: list[dict] = []
    failed_sources: dict[str, str] = {}
    print("Fetching articles…")
    for source in SOURCES:
        articles, error = fetch_rss(source)
        all_articles.extend(articles)
        if error is not None:
            failed_sources[source["name"]] = error
    print(f"Total articles fetched: {len(all_articles)}\n")
    return all_articles, failed_sources


# ---------------------------------------------------------------------------
# Newsletter generation
# ---------------------------------------------------------------------------


def build_prompt(articles: list[dict]) -> str:
    """Build the prompt that will be sent to the LLM."""
    article_lines = []
    for i, article in enumerate(articles, start=1):
        lines = [
            f"[{i}] Source: {article['source']}",
            f"    Title: {article['title']}",
            f"    URL: {article['url']}",
        ]
        # Truncate summary to 100 chars in the prompt to stay within model token limits
        summary = article["summary"]
        if summary:
            lines.append(f"    Summary: {summary[:100]}")
        article_lines.append("\n".join(lines))

    articles_block = "\n\n".join(article_lines)

    today = datetime.now(timezone.utc).strftime("%B %d, %Y")

    return textwrap.dedent(f"""\
        You are an expert AI and developer-tools journalist.

        Today is {today}.

        Below is a list of recent articles from GitHub blogs and AI news pages.
        Your task is to write a concise, well-structured daily newsletter in
        Markdown format that:

        1. Starts with a short "Today's Highlights" paragraph (2-4 sentences)
           summarizing the most important themes.
        2. Groups articles into thematic sections (e.g. "GitHub & Copilot",
           "Foundation Models", "AI Agents & Tooling", "Research", "Other").
        3. For each article, writes a 1-3 sentence analysis explaining *why*
           it matters for AI agent developers and what to watch.
        4. Ends with a "Key Takeaways" bullet list (3-5 bullets).

        Use proper Markdown: headings, bullet points, and hyperlinks.
        Do NOT invent facts – only use information from the articles provided.
        If an article is not relevant to AI or developer tooling, skip it.

        ---

        ARTICLES:

        {articles_block}
    """)


def generate_newsletter(articles: list[dict], client: OpenAI) -> str:
    """Send articles to GitHub Models and return the generated newsletter text."""
    if not articles:
        raise ValueError("No articles to summarize.")

    print(f"Generating newsletter with {GITHUB_MODEL}…")
    prompt = build_prompt(articles)

    response = client.chat.completions.create(
        model=GITHUB_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def build_recent_headlines_section(
    articles: list[dict], failed_sources: dict[str, str]
) -> str:
    """Build a Markdown section listing recent headlines grouped by source."""
    by_source: dict[str, list[dict]] = {}
    for article in articles:
        by_source.setdefault(article["source"], []).append(article)

    lines = ["## 📰 Recent Headlines by Source", ""]
    for source_name, source_articles in by_source.items():
        lines.append(f"### {source_name}")
        for article in source_articles:
            title = article["title"] or "(no title)"
            url = article["url"]
            if url:
                lines.append(f"- [{title}]({url})")
            else:
                lines.append(f"- {title}")
        lines.append("")

    if failed_sources:
        lines.append("### ⚠️ Sources That Could Not Be Retrieved")
        for source_name, error_msg in failed_sources.items():
            lines.append(f"- **{source_name}**: {error_msg}")
        lines.append("")

    return "\n".join(lines)


def wrap_newsletter(
    body: str, article_count: int, articles: list[dict], failed_sources: dict[str, str]
) -> str:
    """Add a standard header/footer to the LLM-generated body."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%B %d, %Y")
    filename_date = now.strftime("%Y-%m-%d")

    header = textwrap.dedent(f"""\
        ---
        title: "AI & GitHub Agent News – {date_str}"
        date: {filename_date}
        articles_analyzed: {article_count}
        model: {GITHUB_MODEL}
        ---

        # 🤖 AI & GitHub Agent News – {date_str}

    """)

    footer = textwrap.dedent(f"""

        ---
        *Generated on {date_str} · {article_count} articles analyzed · \
model: {GITHUB_MODEL}*
    """)

    recent_headlines = build_recent_headlines_section(articles, failed_sources)

    return header + body + "\n\n" + recent_headlines + footer


def save_newsletter(content: str) -> Path:
    """Write the newsletter to the newsletters/ directory and return the path."""
    NEWSLETTERS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_path = NEWSLETTERS_DIR / f"{date_str}.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("ERROR: GITHUB_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # GitHub Models is OpenAI-SDK-compatible; only the base_url differs.
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=github_token,
    )

    articles, failed_sources = fetch_all_articles()
    if not articles:
        print("ERROR: No articles fetched from any source.", file=sys.stderr)
        sys.exit(1)

    newsletter_body = generate_newsletter(articles, client)
    full_newsletter = wrap_newsletter(newsletter_body, len(articles), articles, failed_sources)
    output_path = save_newsletter(full_newsletter)

    print(f"\n✅ Newsletter saved to: {output_path}")
    print(f"   Articles analyzed:  {len(articles)}")


if __name__ == "__main__":
    main()
