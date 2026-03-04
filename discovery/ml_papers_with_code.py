"""
ml_papers_with_code.py - ML techniques from Papers With Code mapped to competition domains.

Crawls Papers With Code for ML technique implementations and creates:
    (paper_technique, competition_domain, performance_improvement) triples

This creates a bridge between research papers and competition-winning code,
which is the core intellectual value proposition of Podium.

Papers With Code API: https://paperswithcode.com/api/v1/
Docs: https://paperswithcode.com/api/v1/docs/

Usage:
    python discovery/ml_papers_with_code.py
    python discovery/ml_papers_with_code.py --task image-classification
    python discovery/ml_papers_with_code.py --methods --results
"""

import argparse
import json
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
PWC_METHODS_FILE = DATA_DIR / "pwc_methods.jsonl"
PWC_RESULTS_FILE = DATA_DIR / "pwc_results.jsonl"
PWC_REPOS_FILE = DATA_DIR / "pwc_repos.jsonl"

PWC_BASE = "https://paperswithcode.com/api/v1"

# ─── Task to competition domain mapping ──────────────────────────────────────
# Maps Papers With Code task slugs to Podium competition domains
TASK_TO_DOMAIN = {
    # Computer Vision
    "image-classification": "cv",
    "object-detection": "cv",
    "semantic-segmentation": "cv",
    "image-segmentation": "cv",
    "instance-segmentation": "cv",
    "image-generation": "cv",
    "face-recognition": "cv",
    "action-recognition": "cv",
    "pose-estimation": "cv",
    "depth-estimation": "cv",
    "optical-flow-estimation": "cv",
    "super-resolution": "cv",
    "image-inpainting": "cv",
    "medical-image-segmentation": "cv",
    "document-classification": "cv",
    "scene-recognition": "cv",

    # NLP
    "text-classification": "nlp",
    "sentiment-analysis": "nlp",
    "natural-language-inference": "nlp",
    "machine-translation": "nlp",
    "question-answering": "nlp",
    "named-entity-recognition": "nlp",
    "text-generation": "nlp",
    "language-modelling": "nlp",
    "relation-extraction": "nlp",
    "text-summarization": "nlp",
    "coreference-resolution": "nlp",
    "information-extraction": "nlp",

    # Tabular / Structured
    "regression": "tabular",
    "classification": "tabular",
    "tabular-data-binary-classification": "tabular",
    "anomaly-detection": "tabular",
    "fraud-detection": "tabular",

    # Time Series
    "time-series-forecasting": "time_series",
    "time-series-classification": "time_series",
    "traffic-prediction": "time_series",

    # Audio
    "speech-recognition": "nlp",
    "audio-classification": "other",
    "music-generation": "other",
}

# ─── Key techniques that win competitions ─────────────────────────────────────
HIGH_VALUE_METHODS = [
    # Boosting / Tabular
    "gradient-boosting",
    "xgboost",
    "lightgbm",
    "catboost",
    "random-forests",

    # Deep Learning Architectures
    "vision-transformer-vit",
    "resnet",
    "efficientnet",
    "bert",
    "roberta",
    "gpt-2",
    "t5",
    "unet",
    "yolo",
    "detr",

    # Training Techniques
    "transfer-learning",
    "data-augmentation",
    "mixup",
    "cutmix",
    "test-time-augmentation",
    "stochastic-weight-averaging",
    "label-smoothing",
    "knowledge-distillation",
    "self-supervised-learning",
    "semi-supervised-learning",

    # Ensemble
    "ensemble-learning",
    "stacking",
    "bagging",
    "blending",
]


def pwc_get(endpoint: str, params: dict = None) -> dict:
    """Make Papers With Code API request."""
    if params is None:
        params = {}
    url = f"{PWC_BASE}/{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": "nalana-dataset-harvester/1.0",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.debug(f"PWC API {endpoint}: {e}")
        return {}


def fetch_all_methods(max_pages: int = 100) -> list[dict]:
    """Fetch all ML methods from Papers With Code."""
    methods = []
    page = 1
    while page <= max_pages:
        data = pwc_get("methods/", {"page": page, "items_per_page": 500})
        results = data.get("results", [])
        if not results:
            break
        methods.extend(results)
        logger.info(f"Methods page {page}: {len(results)} items (total: {len(methods)})")
        if not data.get("next"):
            break
        page += 1
        time.sleep(0.3)
    return methods


def fetch_task_results(task_slug: str, max_pages: int = 20) -> list[dict]:
    """Fetch benchmark results for a specific task (SOTA leaderboard data)."""
    results = []
    page = 1
    while page <= max_pages:
        data = pwc_get(f"tasks/{task_slug}/results/", {"page": page, "items_per_page": 100})
        batch = data.get("results", [])
        if not batch:
            break
        results.extend(batch)
        if not data.get("next"):
            break
        page += 1
        time.sleep(0.2)
    return results


def fetch_method_implementations(method_id: str, max_results: int = 50) -> list[dict]:
    """Fetch GitHub repo implementations for a method."""
    data = pwc_get(f"methods/{method_id}/repositories/", {"items_per_page": max_results})
    return data.get("results", [])


def fetch_all_tasks() -> list[dict]:
    """Fetch all task categories from Papers With Code."""
    tasks = []
    page = 1
    while True:
        data = pwc_get("tasks/", {"page": page, "items_per_page": 500})
        results = data.get("results", [])
        if not results:
            break
        tasks.extend(results)
        if not data.get("next"):
            break
        page += 1
        time.sleep(0.3)
    return tasks


def build_technique_triple(
    method: dict,
    result: dict,
    competition_domain: str,
) -> Optional[dict]:
    """
    Build a (paper_technique, competition_domain, performance_improvement) triple.
    """
    method_name = method.get("name", "")
    if not method_name:
        return None

    # Extract performance metrics
    metric_name = result.get("metric_name", "")
    metric_value = result.get("metrics", {})
    paper_title = (result.get("paper") or {}).get("title", "")
    paper_url = (result.get("paper") or {}).get("url_pdf", "")

    # Build natural language description
    description_parts = [
        f"Technique: {method_name}",
    ]
    if method.get("description"):
        description_parts.append(f"Description: {method.get('description', '')[:500]}")
    if paper_title:
        description_parts.append(f"From paper: {paper_title}")
    if competition_domain:
        description_parts.append(f"Competition domain: {competition_domain}")
    if metric_name and metric_value:
        description_parts.append(f"Performance metric: {metric_name} = {metric_value}")

    return {
        "type": "technique_triple",
        "method_name": method_name,
        "method_id": method.get("id"),
        "competition_domain": competition_domain,
        "paper_title": paper_title,
        "paper_url": paper_url,
        "metric_name": metric_name,
        "metric_value": str(metric_value),
        "description": ". ".join(description_parts),
        "method_description": (method.get("description") or "")[:500],
        "task_name": result.get("task", {}).get("name", "") if isinstance(result.get("task"), dict) else "",
        "dataset_name": result.get("dataset", {}).get("name", "") if isinstance(result.get("dataset"), dict) else "",
        "year": (result.get("paper") or {}).get("published", "")[:4] if result.get("paper") else "",
    }


def build_method_record(method: dict, repos: list[dict]) -> dict:
    """Build a training record from a method and its implementations."""
    return {
        "type": "method_with_code",
        "method_name": method.get("name", ""),
        "method_id": method.get("id"),
        "description": (method.get("description") or "")[:1000],
        "paper": method.get("paper"),
        "num_implementations": len(repos),
        "top_repos": [
            {
                "url": r.get("url"),
                "stars": r.get("stars", 0),
                "framework": r.get("framework"),
                "language": r.get("language"),
            }
            for r in sorted(repos, key=lambda x: x.get("stars", 0), reverse=True)[:5]
        ],
        "categories": method.get("categories", []),
        "area": method.get("area", {}).get("name", "") if isinstance(method.get("area"), dict) else "",
    }


def save_records(records: list[dict], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Collect ML technique data from Papers With Code"
    )
    parser.add_argument("--all", action="store_true", help="Run all collection modes")
    parser.add_argument("--methods", action="store_true",
                        help="Collect all ML methods with implementations")
    parser.add_argument("--results", action="store_true",
                        help="Collect SOTA benchmark results (technique triples)")
    parser.add_argument("--task", type=str, default=None,
                        help="Specific task slug to fetch results for")
    parser.add_argument("--max-method-pages", type=int, default=50)
    parser.add_argument("--max-task-pages", type=int, default=10)
    args = parser.parse_args()

    if args.all:
        args.methods = args.results = True

    if not any([args.methods, args.results, args.task]):
        print("Specify: --methods, --results, --task SLUG, or --all")
        return

    # ── Collect methods with implementations ──────────────────────────────────
    if args.methods:
        logger.info("=== COLLECTING ML METHODS ===")
        all_methods = fetch_all_methods(args.max_method_pages)
        logger.info(f"Total methods fetched: {len(all_methods)}")

        method_records = []
        for i, method in enumerate(all_methods):
            mid = method.get("id")
            if not mid:
                continue
            repos = fetch_method_implementations(mid)
            rec = build_method_record(method, repos)
            method_records.append(rec)
            if (i + 1) % 100 == 0:
                logger.info(f"Methods processed: {i+1}/{len(all_methods)}")
            time.sleep(0.2)

        save_records(method_records, PWC_METHODS_FILE)
        logger.info(f"Saved {len(method_records)} method records to {PWC_METHODS_FILE}")

    # ── Collect SOTA results (technique triples) ───────────────────────────────
    if args.results or args.task:
        logger.info("=== COLLECTING SOTA RESULTS ===")

        # Determine which tasks to process
        if args.task:
            task_slugs = [(args.task, TASK_TO_DOMAIN.get(args.task, "other"))]
        else:
            task_slugs = [(slug, domain) for slug, domain in TASK_TO_DOMAIN.items()]

        all_triples = []
        for task_slug, domain in task_slugs:
            logger.info(f"  Task: {task_slug} -> {domain}")
            results = fetch_task_results(task_slug, args.max_task_pages)

            for result in results:
                # We don't have the method object here, build a simplified record
                paper = result.get("paper") or {}
                triple = {
                    "type": "sota_result",
                    "task": task_slug,
                    "competition_domain": domain,
                    "method_name": result.get("method_name", ""),
                    "paper_title": paper.get("title", ""),
                    "paper_url": paper.get("url_pdf", ""),
                    "metric_name": result.get("metric_name", ""),
                    "metrics": result.get("metrics", {}),
                    "dataset": (result.get("dataset") or {}).get("name", ""),
                    "year": paper.get("published", "")[:4],
                    "evaluation": result.get("evaluation", ""),
                }
                all_triples.append(triple)

            time.sleep(0.3)

        save_records(all_triples, PWC_RESULTS_FILE)
        logger.info(f"Saved {len(all_triples)} SOTA result triples to {PWC_RESULTS_FILE}")

    logger.info(f"\nNext step: python discovery/solution_writeups_v2.py")


if __name__ == "__main__":
    main()
