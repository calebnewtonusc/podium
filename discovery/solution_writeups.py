"""
Winning solution writeup discovery.
Scrapes post-competition discussion posts from top finishers.
"""

import json
import re
import time
from pathlib import Path

import httpx
from loguru import logger
from tqdm import tqdm

# Kaggle discussion API base
KAGGLE_API = "https://www.kaggle.com/api/v1"

SOLUTION_TITLE_PATTERNS = [
    r"\b(solution|approach|1st|2nd|3rd|gold|silver|bronze|winner|winning)\b",
    r"\bplace\b",
    r"\bwrite.?up\b",
    r"\bsummary\b",
]

SOLUTION_REGEX = re.compile(
    "|".join(SOLUTION_TITLE_PATTERNS),
    re.IGNORECASE,
)


def is_solution_post(title: str, author_rank: int | None = None) -> bool:
    """Heuristic: is this a winning solution writeup?"""
    title_match = bool(SOLUTION_REGEX.search(title))
    top_finisher = author_rank is not None and author_rank <= 25
    return title_match or top_finisher


def fetch_competition_discussions(
    competition_slug: str,
    session: httpx.Client,
    min_votes: int = 5,
) -> list[dict]:
    """Fetch all discussion topics for a competition, filtered by votes."""
    topics = []
    page = 1
    while True:
        resp = session.get(
            f"{KAGGLE_API}/competitions/{competition_slug}/topics",
            params={"pageSize": 100, "page": page, "sortBy": "voteCount"},
        )
        if resp.status_code != 200:
            break
        data = resp.json()
        batch = data.get("topics", [])
        if not batch:
            break
        for topic in batch:
            if topic.get("voteCount", 0) < min_votes:
                break
            topics.append(topic)
        if len(batch) < 100:
            break
        page += 1
        time.sleep(0.15)
    return topics


def fetch_topic_content(
    competition_slug: str,
    topic_id: int,
    session: httpx.Client,
) -> str | None:
    """Fetch full text content of a discussion topic."""
    resp = session.get(f"{KAGGLE_API}/competitions/{competition_slug}/topics/{topic_id}")
    if resp.status_code != 200:
        return None
    data = resp.json()
    return data.get("content", "")


def collect_solution_writeups(
    competitions: list[str],
    output_dir: Path,
    kaggle_username: str,
    kaggle_key: str,
) -> None:
    """
    For each competition, fetch top-voted discussion posts that look like
    solution writeups. Save as JSONL.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "solution_writeups.jsonl"

    auth = (kaggle_username, kaggle_key)
    collected = 0

    with httpx.Client(auth=auth, timeout=30.0) as session:
        with open(output_path, "a") as out_f:
            for slug in tqdm(competitions, desc="Collecting writeups"):
                try:
                    topics = fetch_competition_discussions(slug, session)
                except Exception as e:
                    logger.warning(f"Failed to fetch discussions for {slug}: {e}")
                    continue

                for topic in topics:
                    title = topic.get("title", "")
                    topic_id = topic.get("id")
                    author_rank = topic.get("author", {}).get("rank")
                    if not is_solution_post(title, author_rank):
                        continue

                    content = fetch_topic_content(slug, topic_id, session)
                    if not content or len(content) < 200:
                        continue

                    entry = {
                        "competition": slug,
                        "topic_id": topic_id,
                        "title": title,
                        "votes": topic.get("voteCount"),
                        "author": topic.get("author", {}).get("name"),
                        "content": content,
                    }
                    out_f.write(json.dumps(entry) + "\n")
                    collected += 1

                time.sleep(0.1)

    logger.info(f"Collected {collected} solution writeups → {output_path}")


if __name__ == "__main__":
    import os
    import typer

    def main(
        competitions_file: str = "./data/raw/competitions.json",
        output_dir: str = "./data/raw/writeups",
    ):
        with open(competitions_file) as f:
            competitions = [c["slug"] for c in json.load(f)]
        kaggle_username = os.environ.get("KAGGLE_USERNAME")
        kaggle_key = os.environ.get("KAGGLE_KEY")
        if not kaggle_username:
            raise ValueError("KAGGLE_USERNAME not set. Export it: export KAGGLE_USERNAME=your-username")
        if not kaggle_key:
            raise ValueError("KAGGLE_KEY not set. Export it: export KAGGLE_KEY=your-key")
        collect_solution_writeups(
            competitions,
            Path(output_dir),
            kaggle_username=kaggle_username,
            kaggle_key=kaggle_key,
        )

    typer.run(main)
