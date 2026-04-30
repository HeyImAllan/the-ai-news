"""
AI News Newsletter Generator

Crawls a curated list of GitHub blogs and AI news pages, then uses the
GitHub Models inference API (no external subscription required) to analyze
and summarize the articles into a Markdown newsletter.

Only articles published within the last 24 hours are included to avoid
duplicate topics across newsletter runs.

Usage:
    python src/newsletter.py

Required environment variable:
    GITHUB_TOKEN             - GitHub token (automatically set in Actions)

Optional environment variable:
    GITHUB_MODEL             - GitHub Models model to use (default: gpt-5)
"""

import os
import re
import sys
import textwrap
from datetime import datetime, timedelta, timezone
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

GITHUB_MODEL = os.environ.get("GITHUB_MODEL") or "gpt-5"
LOOKBACK_HOURS = 24
REPO_ROOT = Path(__file__).parent.parent
NEWSLETTERS_DIR = REPO_ROOT / "newsletters"
TODAY_FILE = REPO_ROOT / "TODAY.MD"
ENGLISH_MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

REQUIRED_H2_SECTIONS = (
    "✨ Today's Highlights",
    "🚀 What Changed Today",
    "📚 Deep Dive by Theme",
    "✅ Key Takeaways",
)

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
    """Parse an RSS/Atom feed and return (articles, error_message).

    Only articles published within the last LOOKBACK_HOURS hours are included.
    """
    try:
        response = requests.get(
            source["url"], headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
        articles = []
        undated_count = 0
        for entry in feed.entries:
            # Use published_parsed (UTC time.struct_time) when available
            published_parsed = entry.get("published_parsed")
            if published_parsed is not None:
                published_dt = datetime(*published_parsed[:6], tzinfo=timezone.utc)
                if published_dt < cutoff:
                    continue
            else:
                undated_count += 1
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
        dated_count = len(articles) - undated_count
        msg = f"  ✓ {source['name']}: fetched {len(articles)} article(s) from the last {LOOKBACK_HOURS}h"
        if undated_count:
            msg += f" ({dated_count} dated, {undated_count} undated)"
        print(msg, flush=True)
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

        Use this exact section structure for a consistent modern look:
        1. ## ✨ Today's Highlights
           - one short paragraph (2-4 sentences) covering the top themes.
        2. ## 🚀 What Changed Today
           - 3-6 concise bullet points describing meaningful updates.
        3. ## 📚 Deep Dive by Theme
           - Use H3 subsections for themes (for example: "GitHub & Copilot",
             "Foundation Models", "AI Agents & Tooling", "Research", "Other").
           - For each article, add a bullet with a markdown link, followed by
             a 1-3 sentence analysis of why it matters and what to watch next.
        4. ## ✅ Key Takeaways
           - 3-5 concise bullets.

        Use proper Markdown: headings, bullet points, and hyperlinks.
        Keep tone professional, modern, and skimmable.
        Do not add a top-level H1 title because the wrapper already provides it.
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
    return normalize_newsletter_body(response.choices[0].message.content.strip())


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


def save_today_markdown(content: str, today_file: Path = TODAY_FILE) -> Path:
    """Write today's generated newsletter to TODAY.MD in the repository root."""
    today_file.write_text(content, encoding="utf-8")
    return today_file


def normalize_newsletter_body(body: str) -> str:
    """Normalize model output so required H2 sections are always present."""
    heading_aliases = {
        "todays highlights": "✨ Today's Highlights",
        "what changed today": "🚀 What Changed Today",
        "deep dive by theme": "📚 Deep Dive by Theme",
        "key takeaways": "✅ Key Takeaways",
    }
    default_section_content = {
        "✨ Today's Highlights": "No major highlights were identified today.",
        "🚀 What Changed Today": "- No major updates were identified in today's sources.",
        "📚 Deep Dive by Theme": "### Other\n- No additional thematic analysis was generated.",
        "✅ Key Takeaways": "- No additional takeaways were generated.",
    }

    normalized_lines: list[str] = []
    seen_required_sections: set[str] = set()
    for line in body.splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            heading_text = line[3:].strip()
            # Normalize heading aliases by stripping emojis/punctuation and
            # collapsing whitespace to preserve word boundaries.
            # The regex keeps only lowercase letters, digits, and spaces.
            normalized_heading_key = " ".join(
                re.sub(r"[^a-z0-9 ]+", " ", heading_text.lower()).split()
            )
            canonical_heading = heading_aliases.get(normalized_heading_key)
            if canonical_heading:
                normalized_lines.append(f"## {canonical_heading}")
                seen_required_sections.add(canonical_heading)
                continue
        normalized_lines.append(line)

    normalized_body = "\n".join(normalized_lines).strip()
    missing_sections = [
        heading for heading in REQUIRED_H2_SECTIONS if heading not in seen_required_sections
    ]
    if missing_sections:
        additions: list[str] = []
        for heading in missing_sections:
            additions.extend(["", f"## {heading}", default_section_content[heading]])
        normalized_body = normalized_body + "\n" + "\n".join(additions)

    return normalized_body.strip()


def compact_previous_month_news(
    now: datetime | None = None, newsletters_dir: Path = NEWSLETTERS_DIR
) -> Path | None:
    """Create a monthly file containing all previous-month daily newsletters."""
    current = now or datetime.now(timezone.utc)
    if current.day != 1:
        return None

    previous_month_last_day = current.replace(day=1) - timedelta(days=1)
    month_prefix = previous_month_last_day.strftime("%Y-%m")
    overview_path = newsletters_dir / f"{month_prefix}.md"
    if overview_path.exists():
        return None

    daily_files = sorted(newsletters_dir.glob(f"{month_prefix}-*.md"))
    if not daily_files:
        return None

    previous_month_name = ENGLISH_MONTH_NAMES[previous_month_last_day.month]
    current_month_name = ENGLISH_MONTH_NAMES[current.month]
    lines = [
        f"# 📅 {previous_month_name} {previous_month_last_day.year} News Overview",
        "",
        (
            f"Compacted daily newsletters for {previous_month_name} "
            f"{previous_month_last_day.year}."
        ),
        "",
    ]
    for daily_file in daily_files:
        day = daily_file.stem
        try:
            daily_content = daily_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"Failed to read daily newsletter '{daily_file.name}'"
            ) from exc
        lines.extend(
            [
                f"## {day}",
                "",
                f"Original daily newsletter: [{daily_file.name}]({daily_file.name})",
                "",
                daily_content,
                "",
            ]
        )
    lines.append("")
    lines.append("---")
    lines.append(
        f"*Compiled monthly news overview generated on "
        f"{current_month_name} {current.day}, {current.year} from "
        f"{len(daily_files)} daily newsletters.*"
    )
    lines.append("")

    overview_path.write_text("\n".join(lines), encoding="utf-8")
    return overview_path


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

    monthly_overview_path = compact_previous_month_news()
    articles, failed_sources = fetch_all_articles()
    if not articles:
        if monthly_overview_path:
            print(f"✅ Monthly overview saved to: {monthly_overview_path}")
        print("No articles fetched from any source. Skipping newsletter generation.")
        sys.exit(0)

    newsletter_body = generate_newsletter(articles, client)
    full_newsletter = wrap_newsletter(newsletter_body, len(articles), articles, failed_sources)
    output_path = save_newsletter(full_newsletter)
    today_path = save_today_markdown(full_newsletter)

    print(f"\n✅ Newsletter saved to: {output_path}")
    print(f"✅ Today file saved to: {today_path}")
    if monthly_overview_path:
        print(f"✅ Monthly overview saved to: {monthly_overview_path}")
    print(f"   Articles analyzed:  {len(articles)}")


if __name__ == "__main__":
    main()
