"""
Kaggle public notebook discovery and collection.
Fetches high-quality competition notebooks via the Kaggle API.
"""

import json
import time
from pathlib import Path
from typing import Iterator

from kaggle.api.kaggle_api_extended import KaggleApiExtended
from loguru import logger
from tqdm import tqdm


QUALITY_FILTERS = {
    "min_votes": 50,
    "must_be_competition": True,
    "must_have_output": True,
}


def get_all_competitions(api: KaggleApiExtended) -> list[dict]:
    """Fetch full list of past competitions from Kaggle API."""
    competitions = []
    page = 1
    while True:
        batch = api.competitions_list(page=page, search="")
        if not batch:
            break
        competitions.extend([c.__dict__ for c in batch])
        page += 1
        time.sleep(0.2)
    logger.info(f"Found {len(competitions)} competitions")
    return competitions


def get_competition_notebooks(
    api: KaggleApiExtended,
    competition_slug: str,
    min_votes: int = 50,
) -> list[dict]:
    """Fetch public notebooks for a single competition, filtered by quality."""
    notebooks = []
    page = 1
    while True:
        try:
            batch = api.kernels_list(
                competition=competition_slug,
                page_size=100,
                page=page,
                sort_by="voteCount",
            )
        except Exception as e:
            logger.warning(f"Failed to list kernels for {competition_slug}: {e}")
            break
        if not batch:
            break
        for kernel in batch:
            k = kernel.__dict__
            if k.get("totalVotes", 0) < min_votes:
                # PO-68: Don't break — the API sorts by voteCount but may not be perfectly
                # monotone (ties, missing values). Use continue to avoid silently dropping
                # high-vote notebooks that follow a low-vote entry in the same page.
                continue
            # PO-67: Use `not k.get("isRunning", False)` instead of `k.get("isRunning") is False`
            # so kernels where isRunning is absent (not returned by the API) are still included.
            if k.get("hasOutput") and not k.get("isRunning", False):
                notebooks.append(k)
        if len(batch) < 100:
            break
        page += 1
        time.sleep(0.1)
    return notebooks


def download_notebook(
    api: KaggleApiExtended,
    kernel_ref: str,
    output_dir: Path,
) -> Path | None:
    """Download a single notebook JSON to output_dir. Returns path or None on failure."""
    output_path = output_dir / f"{kernel_ref.replace('/', '__')}.json"
    if output_path.exists():
        return output_path
    try:
        api.kernels_pull(kernel_ref, path=str(output_dir), metadata=True)
        # Rename downloaded file to our convention
        ipynb_files = list(output_dir.glob("*.ipynb"))
        if ipynb_files:
            ipynb_files[0].rename(output_path.with_suffix(".ipynb"))
            return output_path.with_suffix(".ipynb")
        return None
    except Exception as e:
        logger.debug(f"Failed to download {kernel_ref}: {e}")
        return None


def stream_competition_notebooks(
    competitions: list[dict],
    output_dir: Path,
    max_per_competition: int = 50,
) -> Iterator[dict]:
    """
    Yield notebook metadata dicts as they are discovered.
    Downloads happen lazily — callers can parallelize with fetch_bulk.py.
    """
    api = KaggleApiExtended()
    api.authenticate()

    output_dir.mkdir(parents=True, exist_ok=True)

    for comp in tqdm(competitions, desc="Scanning competitions"):
        slug = (comp.get("ref") or "").split("/")[-1]
        if not slug:
            continue
        notebooks = get_competition_notebooks(api, slug, QUALITY_FILTERS["min_votes"])
        for nb in notebooks[:max_per_competition]:
            yield {
                "competition": slug,
                "kernel_ref": nb.get("ref"),
                "votes": nb.get("totalVotes"),
                "language": nb.get("language"),
                "metric": comp.get("evaluationMetric"),
            }


def fetch_notebook_metadata_bulk(
    output_path: Path, max_competitions: int = 1000
) -> None:
    """
    Discover all high-quality competition notebooks and save metadata index.
    This is the first step — run before fetch_bulk.py.
    """
    api = KaggleApiExtended()
    api.authenticate()

    competitions = get_all_competitions(api)[:max_competitions]
    index = []

    for meta in stream_competition_notebooks(competitions, output_path / "raw"):
        index.append(meta)

    index_path = output_path / "notebook_index.jsonl"
    with open(index_path, "w") as f:
        for entry in index:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Saved {len(index)} notebook metadata entries to {index_path}")


if __name__ == "__main__":
    import typer

    def main(
        output_dir: str = "./data/raw/notebooks",
        max_competitions: int = 1000,
    ):
        fetch_notebook_metadata_bulk(Path(output_dir), max_competitions)

    typer.run(main)
