# the-ai-news

A GitHub Actions agentic workflow that crawls GitHub blogs and AI news pages,
then uses OpenAI to analyze and summarize them into a Markdown newsletter.

## How it works

1. The workflow fetches the latest articles from a curated list of RSS feeds
   (GitHub Blog, GitHub Changelog, OpenAI, Anthropic, Google DeepMind,
   Microsoft AI, HuggingFace).
2. It sends the article titles and summaries to an OpenAI model.
3. The model produces a structured Markdown newsletter grouped by theme.
4. The newsletter is committed to the `newsletters/` directory as
   `YYYY-MM-DD.md`.

## Setup

1. **Add your OpenAI API key** as a repository secret named `OPENAI_API_KEY`
   (Settings → Secrets and variables → Actions → New repository secret).

2. **Run the workflow manually** via
   Actions → *Generate AI News Newsletter* → *Run workflow*.

   You can optionally override:
   - `max_articles_per_source` (default `5`)
   - `openai_model` (default `gpt-4o-mini`)

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python src/newsletter.py
```

The newsletter is written to `newsletters/YYYY-MM-DD.md`.

## Scheduling (future)

A `schedule` trigger is intentionally **not configured** yet to avoid
unnecessary token costs. Once the output quality is satisfactory, add a
`cron` entry to `.github/workflows/newsletter.yml`.

## Project structure

```
.github/workflows/newsletter.yml   # GitHub Actions workflow (manual trigger)
src/newsletter.py                  # Main script – fetch, summarize, write
requirements.txt                   # Python dependencies
newsletters/                       # Generated newsletters (YYYY-MM-DD.md)
```
