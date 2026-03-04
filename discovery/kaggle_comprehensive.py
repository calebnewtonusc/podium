"""
kaggle_comprehensive.py - Exhaustive ML competition solution discovery.

Pulls from multiple competition platforms:
- Kaggle: all competitions, notebooks, discussion posts with code
- DrivenData: competition solution writeups
- Zindi: African ML competition solutions
- AICrowd: competition notebooks
- Analytics Vidhya: competition writeups

Also fetches grandmaster kernels (upvotes > 50) and discussion posts
that often contain better explanations than top notebooks.

Target: 500k+ competition code snippets.

Usage:
    python discovery/kaggle_comprehensive.py --help
    python discovery/kaggle_comprehensive.py --all
    python discovery/kaggle_comprehensive.py --kaggle --min-votes 20
    python discovery/kaggle_comprehensive.py --discussions  # Kaggle discussion posts with code
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from kaggle.api.kaggle_api_extended import KaggleApiExtended

    HAS_KAGGLE = True
except ImportError:
    HAS_KAGGLE = False

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

DATA_DIR = Path(__file__).parents[1] / "data"
COMP_INDEX_FILE = DATA_DIR / "competition_index.jsonl"
NOTEBOOK_INDEX_FILE = DATA_DIR / "notebook_index.jsonl"
DISCUSSION_INDEX_FILE = DATA_DIR / "discussion_index.jsonl"
EXTERNAL_SOLUTIONS_FILE = DATA_DIR / "external_solutions.jsonl"

# ─── Platform base URLs ───────────────────────────────────────────────────────
DRIVENDATA_BASE = "https://www.drivendata.org"
ZINDI_BASE = "https://zindi.africa/api/competitions"
AICROWD_BASE = "https://www.aicrowd.com/api/v1"

# ─── Code detection in discussion posts ──────────────────────────────────────
CODE_BLOCK_PATTERN = re.compile(r"```[\w]*\n(.*?)```", re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`([^`]{10,})`")
PYTHON_INDICATORS = [
    "import",
    "def ",
    "class ",
    "pd.read_csv",
    "np.",
    "sklearn",
    "torch",
    "tensorflow",
    "keras",
    "xgboost",
    "lgbm",
    "catboost",
    "train_test_split",
    "cross_val_score",
    "GridSearchCV",
    "Pipeline",
]


def has_code_content(text: str) -> bool:
    """Check if text contains meaningful code."""
    if CODE_BLOCK_PATTERN.search(text):
        return True
    for indicator in PYTHON_INDICATORS:
        if indicator in text:
            return True
    return False


def extract_code_blocks(text: str) -> list[str]:
    """Extract code blocks from markdown text."""
    blocks = []
    for match in CODE_BLOCK_PATTERN.finditer(text):
        code = match.group(1).strip()
        if len(code) > 20:
            blocks.append(code)
    return blocks


def classify_competition_type(comp_info: dict) -> str:
    """
    Classify competition type for curriculum ordering.
    Returns: tabular, cv, nlp, time_series, multimodal, other
    """
    title = (comp_info.get("title") or "").lower()
    desc = (comp_info.get("description") or "").lower()
    tags = [t.lower() for t in (comp_info.get("tags") or [])]
    combined = f"{title} {desc} {' '.join(tags)}"

    if any(
        kw in combined
        for kw in [
            "image",
            "vision",
            "object detection",
            "segmentation",
            "classification image",
            "cnn",
            "resnet",
            "efficientnet",
        ]
    ):
        return "cv"
    if any(
        kw in combined
        for kw in [
            "nlp",
            "text",
            "natural language",
            "sentiment",
            "bert",
            "transformer",
            "language model",
            "speech",
        ]
    ):
        return "nlp"
    if any(
        kw in combined
        for kw in [
            "time series",
            "forecasting",
            "temporal",
            "sequence",
            "lstm",
            "arima",
            "sales forecast",
        ]
    ):
        return "time_series"
    if any(
        kw in combined
        for kw in [
            "tabular",
            "structured",
            "csv",
            "classification",
            "regression",
            "gradient boosting",
            "xgboost",
        ]
    ):
        return "tabular"
    if any(
        kw in combined
        for kw in [
            "multimodal",
            "multi-modal",
            "image and text",
            "video",
            "audio",
            "3d",
        ]
    ):
        return "multimodal"
    return "tabular"  # default for unknown = tabular (most common)


# ─── Kaggle-specific functions ────────────────────────────────────────────────


def get_kaggle_api() -> Optional[object]:
    if not HAS_KAGGLE:
        logger.warning("kaggle package not installed: pip install kaggle")
        return None
    try:
        api = KaggleApiExtended()
        api.authenticate()
        return api
    except Exception as e:
        logger.error(f"Kaggle auth failed: {e}")
        return None


def fetch_all_competitions(api) -> list[dict]:
    """Fetch complete list of Kaggle competitions."""
    competitions = []
    page = 1
    while True:
        try:
            batch = api.competitions_list(page=page, search="")
        except Exception as e:
            logger.warning(f"competitions_list page {page}: {e}")
            break
        if not batch:
            break
        for comp in batch:
            c = comp.__dict__
            competitions.append(
                {
                    "ref": c.get("ref"),
                    "slug": (c.get("ref") or "").split("/")[-1],
                    "title": c.get("title"),
                    "description": c.get("subtitle") or "",
                    "deadline": str(c.get("deadline") or ""),
                    "total_teams": c.get("totalTeams", 0),
                    "reward": str(c.get("reward") or ""),
                    "evaluation_metric": c.get("evaluationMetric"),
                    "tags": [t.get("name") for t in (c.get("tags") or [])],
                    "competition_type": classify_competition_type(c),
                    "platform": "kaggle",
                }
            )
        page += 1
        time.sleep(0.2)
        if len(batch) < 20:
            break
    logger.info(f"Fetched {len(competitions)} Kaggle competitions")
    return competitions


def fetch_competition_notebooks(api, slug: str, min_votes: int = 20) -> list[dict]:
    """Fetch notebooks for a competition, including lower-vote notebooks."""
    notebooks = []
    page = 1
    while True:
        try:
            batch = api.kernels_list(
                competition=slug,
                page_size=100,
                page=page,
                sort_by="voteCount",
            )
        except Exception as e:
            logger.debug(f"kernels_list {slug} page {page}: {e}")
            break
        if not batch:
            break
        for kernel in batch:
            k = kernel.__dict__
            votes = k.get("totalVotes", 0)
            if votes < min_votes:
                break  # sorted desc, done
            if k.get("hasOutput") and not k.get("isRunning"):
                notebooks.append(
                    {
                        "ref": k.get("ref"),
                        "title": k.get("title"),
                        "competition": slug,
                        "votes": votes,
                        "language": k.get("language"),
                        "last_run_time": str(k.get("lastRunTime") or ""),
                        "platform": "kaggle",
                    }
                )
        if len(batch) < 100:
            break
        page += 1
        time.sleep(0.1)
    return notebooks


def fetch_competition_discussions(api, slug: str, min_votes: int = 5) -> list[dict]:
    """
    Fetch discussion posts for a competition.
    Discussion posts often contain gold-quality solution writeups.
    """
    discussions = []
    try:
        batch = api.competitions_list_topics(slug, page_size=100, sort_by="votes")
    except Exception as e:
        logger.debug(f"discussions {slug}: {e}")
        return []

    for topic in batch or []:
        t = topic.__dict__ if hasattr(topic, "__dict__") else topic
        votes = t.get("voteCount", t.get("totalVotes", 0))
        if votes < min_votes:
            continue
        body = t.get("body") or t.get("content") or ""
        if has_code_content(body):
            discussions.append(
                {
                    "id": t.get("id") or t.get("topicId"),
                    "title": t.get("title"),
                    "body": body[:5000],
                    "code_blocks": extract_code_blocks(body),
                    "votes": votes,
                    "competition": slug,
                    "platform": "kaggle",
                    "type": "discussion",
                }
            )
    return discussions


def download_notebook_content(api, kernel_ref: str, output_dir: Path) -> Optional[Path]:
    """Download notebook to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = kernel_ref.replace("/", "__")
    output_path = output_dir / f"{safe_name}.ipynb"
    if output_path.exists():
        return output_path
    try:
        api.kernels_pull(kernel_ref, path=str(output_dir), metadata=False)
        files = list(output_dir.glob("*.ipynb")) + list(output_dir.glob("*.py"))
        if files:
            newest = max(files, key=lambda f: f.stat().st_mtime)
            newest.rename(output_path)
            return output_path
    except Exception as e:
        logger.debug(f"download {kernel_ref}: {e}")
    return None


# ─── DrivenData functions ─────────────────────────────────────────────────────


def fetch_drivendata_competitions() -> list[dict]:
    """Fetch DrivenData competition list from their public API."""
    comps = []
    try:
        url = f"{DRIVENDATA_BASE}/api/v1/competitions?offset=0&limit=100"
        req_headers = {"User-Agent": "nalana-data-harvester/1.0"}
        if HAS_HTTPX:
            with httpx.Client(timeout=20) as client:
                resp = client.get(url, headers=req_headers)
                if resp.status_code == 200:
                    data = resp.json()
                    for comp in data.get("competitions", []):
                        comps.append(
                            {
                                "ref": str(comp.get("id")),
                                "slug": comp.get("slug"),
                                "title": comp.get("title"),
                                "description": comp.get("short_description", "")[:300],
                                "platform": "drivendata",
                                "competition_type": "tabular",  # DrivenData is mostly tabular
                                "url": f"{DRIVENDATA_BASE}/competitions/{comp.get('id')}",
                            }
                        )
        logger.info(f"DrivenData: found {len(comps)} competitions")
    except Exception as e:
        logger.warning(f"DrivenData fetch failed: {e}")
    return comps


def fetch_zindi_competitions() -> list[dict]:
    """Fetch Zindi competition list."""
    comps = []
    try:
        url = f"{ZINDI_BASE}?per_page=100&page=1"
        if HAS_HTTPX:
            with httpx.Client(timeout=20) as client:
                resp = client.get(
                    url, headers={"User-Agent": "nalana-data-harvester/1.0"}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for comp in data.get("data", []):
                        comps.append(
                            {
                                "ref": str(comp.get("id")),
                                "slug": comp.get("slug"),
                                "title": comp.get("title"),
                                "description": (comp.get("description") or "")[:300],
                                "platform": "zindi",
                                "competition_type": classify_competition_type(comp),
                            }
                        )
        logger.info(f"Zindi: found {len(comps)} competitions")
    except Exception as e:
        logger.warning(f"Zindi fetch failed: {e}")
    return comps


# ─── Save functions ────────────────────────────────────────────────────────────


def save_competitions(competitions: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(COMP_INDEX_FILE, "a") as f:
        for c in competitions:
            f.write(json.dumps(c) + "\n")


def save_notebooks(notebooks: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_INDEX_FILE, "a") as f:
        for nb in notebooks:
            f.write(json.dumps(nb) + "\n")


def save_discussions(discussions: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DISCUSSION_INDEX_FILE, "a") as f:
        for d in discussions:
            f.write(json.dumps(d) + "\n")


def load_seen_slugs(filepath: Path) -> set[str]:
    seen = set()
    if filepath.exists():
        with open(filepath) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen.add(
                        rec.get("slug") or rec.get("ref") or rec.get("competition", "")
                    )
                except json.JSONDecodeError:
                    pass
    return seen


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive ML competition solution discovery"
    )
    parser.add_argument("--all", action="store_true", help="Run all collection modes")
    parser.add_argument(
        "--kaggle", action="store_true", help="Kaggle notebooks and competitions"
    )
    parser.add_argument(
        "--discussions",
        action="store_true",
        help="Collect Kaggle discussion posts with code",
    )
    parser.add_argument(
        "--external", action="store_true", help="Collect from DrivenData and Zindi"
    )
    parser.add_argument(
        "--min-votes", type=int, default=20, help="Minimum votes for notebooks"
    )
    parser.add_argument(
        "--min-discussion-votes",
        type=int,
        default=3,
        help="Minimum votes for discussion posts",
    )
    parser.add_argument(
        "--max-comps",
        type=int,
        default=2000,
        help="Max competitions to process on Kaggle",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually download notebook files (slow, large)",
    )
    args = parser.parse_args()

    if args.all:
        args.kaggle = args.discussions = args.external = True

    if not any([args.kaggle, args.discussions, args.external]):
        print("Specify a mode: --kaggle, --discussions, --external, or --all")
        return

    # ── External platforms ────────────────────────────────────────────────────
    if args.external:
        logger.info("=== EXTERNAL PLATFORMS ===")
        all_external = []
        all_external.extend(fetch_drivendata_competitions())
        all_external.extend(fetch_zindi_competitions())
        save_competitions(all_external)
        logger.info(f"External platform competitions: {len(all_external)}")

    # ── Kaggle ────────────────────────────────────────────────────────────────
    if not (args.kaggle or args.discussions):
        return

    api = get_kaggle_api()
    if not api:
        return

    logger.info("=== KAGGLE COMPETITION DISCOVERY ===")
    competitions = fetch_all_competitions(api)
    save_competitions(competitions)
    logger.info(f"Saved {len(competitions)} Kaggle competitions to index")

    seen_notebooks = load_seen_slugs(NOTEBOOK_INDEX_FILE)
    seen_discussions = load_seen_slugs(DISCUSSION_INDEX_FILE)

    total_notebooks = 0
    total_discussions = 0
    raw_notebook_dir = DATA_DIR / "raw" / "notebooks"

    for i, comp in enumerate(competitions[: args.max_comps]):
        slug = comp.get("slug")
        if not slug:
            continue

        if args.kaggle and slug not in seen_notebooks:
            notebooks = fetch_competition_notebooks(api, slug, args.min_votes)
            save_notebooks(notebooks)
            total_notebooks += len(notebooks)
            seen_notebooks.add(slug)

            if args.download:
                for nb in notebooks:
                    ref = nb.get("ref")
                    if ref:
                        download_notebook_content(api, ref, raw_notebook_dir)
                        time.sleep(0.3)

        if args.discussions and slug not in seen_discussions:
            discussions = fetch_competition_discussions(
                api, slug, args.min_discussion_votes
            )
            save_discussions(discussions)
            total_discussions += len(discussions)
            seen_discussions.add(slug)

        if (i + 1) % 50 == 0:
            logger.info(
                f"Progress: {i + 1}/{min(len(competitions), args.max_comps)} competitions | "
                f"notebooks: {total_notebooks} | discussions: {total_discussions}"
            )

        time.sleep(0.2)

    logger.info("\n=== SUMMARY ===")
    logger.info(f"Total notebooks indexed: {total_notebooks}")
    logger.info(f"Total discussion posts with code: {total_discussions}")
    logger.info(f"Notebook index: {NOTEBOOK_INDEX_FILE}")
    logger.info("\nNext step: python discovery/ml_papers_with_code.py")


if __name__ == "__main__":
    main()
