"""
solution_writeups_v2.py - Comprehensive competition solution writeup discovery.

Crawls:
- Kaggle blog posts with winning solution writeups
- Medium ML articles with competition solutions
- GitHub repos tagged as kaggle competition solutions
- Individual solution repos with leaderboard scores

Extracts winning strategies with proven leaderboard scores.

Usage:
    python discovery/solution_writeups_v2.py
    python discovery/solution_writeups_v2.py --github
    python discovery/solution_writeups_v2.py --all
"""

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

DATA_DIR = Path(__file__).parents[1] / "data"
WRITEUPS_FILE = DATA_DIR / "solution_writeups_v2.jsonl"
GH_SOLUTIONS_FILE = DATA_DIR / "github_competition_solutions.jsonl"

GH_BASE = "https://api.github.com"
MEDIUM_SEARCH = "https://medium.com/search"

# ─── GitHub search queries for competition solutions ──────────────────────────
GITHUB_SOLUTION_QUERIES = [
    "topic:kaggle-competition",
    "topic:kaggle-solution",
    "topic:competition-solution",
    "topic:machine-learning-competition",
    "filename:solution.py kaggle",
    "filename:solution.ipynb kaggle",
    "filename:1st-place kaggle",
    "filename:winning-solution kaggle",
    "language:python kaggle competition 1st place solution",
    "language:python kaggle competition gold medal solution",
    "language:python drivendata competition solution",
    "language:python zindi competition solution",
    "language:python aicrowd competition solution",
    "language:python kaggle grandmaster kernel",
    "language:jupyter-notebook kaggle solution notebook",
    "language:python competition winning strategy",
]

# ─── Leaderboard rank detection patterns ─────────────────────────────────────
RANK_PATTERNS = [
    re.compile(r'\b(\d+)(?:st|nd|rd|th)\s+place', re.I),
    re.compile(r'rank\s+(\d+)', re.I),
    re.compile(r'position\s+(\d+)', re.I),
    re.compile(r'top\s+(\d+)%', re.I),
    re.compile(r'gold\s+medal', re.I),
    re.compile(r'silver\s+medal', re.I),
    re.compile(r'bronze\s+medal', re.I),
    re.compile(r'leaderboard\s+score[:\s]+([0-9.]+)', re.I),
]

# ─── Solution quality indicators ─────────────────────────────────────────────
QUALITY_INDICATORS = [
    "ensemble", "stacking", "blending", "cross-validation", "feature engineering",
    "data augmentation", "pseudo labeling", "test-time augmentation", "TTA",
    "model soup", "swa", "stochastic weight averaging", "cyclic lr",
    "knowledge distillation", "transfer learning", "pre-trained",
]

MEDAL_SCORE = {"gold": 3, "silver": 2, "bronze": 1}


def gh_get(endpoint: str, params: dict, token: str = "") -> dict:
    """Make authenticated GitHub API request."""
    url = f"{GH_BASE}/{endpoint}?" + urllib.parse.urlencode(params)
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "nalana-data-harvester/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.debug(f"GitHub {endpoint}: {e}")
        return {}


def extract_leaderboard_rank(text: str) -> Optional[dict]:
    """Extract leaderboard position information from text."""
    rank_info = {"has_rank": False}
    text_lower = text.lower()

    # Check for medals
    for medal, score in MEDAL_SCORE.items():
        if f"{medal} medal" in text_lower:
            rank_info["medal"] = medal
            rank_info["medal_score"] = score
            rank_info["has_rank"] = True
            break

    # Check for numeric rank
    for pattern in RANK_PATTERNS[:3]:  # first 3 are numeric
        match = pattern.search(text)
        if match:
            try:
                rank = int(match.group(1))
                if rank <= 1000:  # sanity check
                    rank_info["rank"] = rank
                    rank_info["has_rank"] = True
                    break
            except (IndexError, ValueError):
                pass

    # Check for top % mentions
    top_pct_match = re.search(r'top\s+(\d+)%', text, re.I)
    if top_pct_match:
        rank_info["top_percent"] = int(top_pct_match.group(1))
        rank_info["has_rank"] = True

    # Check for score
    score_match = re.search(r'score[:\s]+([0-9.]+)', text, re.I)
    if score_match:
        try:
            rank_info["lb_score"] = float(score_match.group(1))
        except ValueError:
            pass

    return rank_info


def score_solution_quality(repo: dict, readme: str = "") -> float:
    """Score how good a solution writeup is as training data."""
    score = 0.0

    # Stars are a proxy for quality
    stars = repo.get("stargazers_count", 0)
    score += min(stars / 50.0, 0.3)  # cap at 0.3

    # Readme length indicates depth of writeup
    if readme:
        score += min(len(readme) / 5000.0, 0.2)

    # Quality indicator presence
    readme_lower = readme.lower()
    indicators_found = sum(1 for qi in QUALITY_INDICATORS if qi.lower() in readme_lower)
    score += min(indicators_found / len(QUALITY_INDICATORS), 0.2)

    # Leaderboard rank info
    rank_info = extract_leaderboard_rank(readme)
    if rank_info.get("medal") == "gold":
        score += 0.3
    elif rank_info.get("medal") == "silver":
        score += 0.2
    elif rank_info.get("medal") == "bronze":
        score += 0.1
    elif rank_info.get("rank") and rank_info["rank"] <= 10:
        score += 0.25
    elif rank_info.get("rank") and rank_info["rank"] <= 50:
        score += 0.15
    elif rank_info.get("has_rank"):
        score += 0.1

    # Description completeness
    desc = repo.get("description") or ""
    if len(desc) > 30:
        score += 0.05

    return min(1.0, score)


def fetch_readme(owner: str, repo_name: str, token: str = "") -> str:
    """Fetch README content for a GitHub repo."""
    for branch in ["main", "master"]:
        url = f"https://raw.githubusercontent.com/{owner}/{repo_name}/{branch}/README.md"
        req = urllib.request.Request(url, headers={
            "User-Agent": "nalana-data-harvester/1.0",
        })
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            continue
    return ""


def search_github_solutions(query: str, token: str, max_pages: int = 5) -> list[dict]:
    """Search GitHub for competition solution repos."""
    repos = []
    for page in range(1, max_pages + 1):
        data = gh_get("search/repositories", {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": 100,
            "page": page,
        }, token)
        items = data.get("items", [])
        if not items:
            break
        repos.extend(items)
        if len(items) < 100:
            break
        time.sleep(0.5)
    return repos


def process_solution_repo(repo: dict, token: str) -> Optional[dict]:
    """Process a competition solution repo into a training record."""
    owner = repo.get("owner", {}).get("login", "")
    repo_name = repo.get("name", "")
    full_name = repo.get("full_name", "")
    stars = repo.get("stargazers_count", 0)
    description = repo.get("description") or ""
    topics = repo.get("topics", [])

    readme = fetch_readme(owner, repo_name, token)
    if not readme:
        return None

    rank_info = extract_leaderboard_rank(f"{description} {readme}")
    quality = score_solution_quality(repo, readme)

    # Detect competition type
    comp_type = "tabular"  # default
    combined = f"{description} {readme}".lower()
    if any(kw in combined for kw in ["image", "vision", "detection", "cnn"]):
        comp_type = "cv"
    elif any(kw in combined for kw in ["text", "nlp", "bert", "language"]):
        comp_type = "nlp"
    elif any(kw in combined for kw in ["time series", "forecast"]):
        comp_type = "time_series"

    # Extract competition name (often in repo name or description)
    comp_name = ""
    kaggle_match = re.search(r'kaggle[/\s-]+(\w[\w-]+)', combined)
    if kaggle_match:
        comp_name = kaggle_match.group(1)

    return {
        "type": "github_solution",
        "full_name": full_name,
        "url": repo.get("html_url"),
        "stars": stars,
        "description": description,
        "topics": topics,
        "readme": readme[:8000],
        "readme_length": len(readme),
        "rank_info": rank_info,
        "quality_score": quality,
        "competition_type": comp_type,
        "competition_name": comp_name,
        "has_leaderboard_proof": rank_info.get("has_rank", False),
        "language": repo.get("language", ""),
    }


def load_seen_repos(filepath: Path) -> set[str]:
    seen = set()
    if filepath.exists():
        with open(filepath) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen.add(rec.get("full_name", ""))
                except json.JSONDecodeError:
                    pass
    return seen


def save_records(records: list[dict], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Discover and harvest competition solution writeups"
    )
    parser.add_argument("--all", action="store_true", help="Run all modes")
    parser.add_argument("--github", action="store_true",
                        help="Search GitHub for competition solution repos")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument("--max-pages", type=int, default=5,
                        help="Pages per GitHub search query")
    parser.add_argument("--min-stars", type=int, default=3,
                        help="Minimum stars for solution repos")
    parser.add_argument("--min-quality", type=float, default=0.2,
                        help="Minimum quality score to save")
    args = parser.parse_args()

    if args.all:
        args.github = True

    if not args.github:
        print("Specify: --github or --all")
        return

    # ── GitHub Solution Discovery ─────────────────────────────────────────────
    logger.info("=== GITHUB SOLUTION DISCOVERY ===")
    seen_repos = load_seen_repos(GH_SOLUTIONS_FILE)

    all_repos: dict[str, dict] = {}
    for query in GITHUB_SOLUTION_QUERIES:
        logger.info(f"  Searching: {query}")
        repos = search_github_solutions(query, args.token, args.max_pages)
        for r in repos:
            fn = r.get("full_name", "")
            if fn and fn not in all_repos:
                all_repos[fn] = r
        logger.info(f"    +{len(repos)} repos (total unique: {len(all_repos)})")
        time.sleep(0.3)

    logger.info(f"\nTotal unique repos: {len(all_repos)}")
    logger.info("Processing repos (fetching READMEs)...")

    processed = 0
    saved = 0
    sorted_repos = sorted(
        all_repos.values(),
        key=lambda r: r.get("stargazers_count", 0),
        reverse=True,
    )

    for repo in sorted_repos:
        full_name = repo.get("full_name", "")
        if full_name in seen_repos:
            continue
        stars = repo.get("stargazers_count", 0)
        if stars < args.min_stars:
            continue

        record = process_solution_repo(repo, args.token)
        processed += 1

        if record and record["quality_score"] >= args.min_quality:
            save_records([record], GH_SOLUTIONS_FILE)
            seen_repos.add(full_name)
            saved += 1

        if processed % 100 == 0:
            logger.info(f"  Processed: {processed} | Saved: {saved}")

        time.sleep(0.15)

    logger.info(f"\n=== SUMMARY ===")
    logger.info(f"Repos processed: {processed}")
    logger.info(f"Solution records saved: {saved}")
    logger.info(f"Output: {GH_SOLUTIONS_FILE}")
    logger.info(f"\nNext step: python synthesis/competition_curriculum.py")


if __name__ == "__main__":
    main()
