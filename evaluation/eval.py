"""
PodiumBench — evaluation suite for Kaggle competition agents.
75 competitions stratified by type, difficulty, and era.
Grades against actual Kaggle leaderboards (medal positions).
"""

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from tqdm import tqdm


# Medal thresholds (% of teams) from Kaggle's official rules
MEDAL_THRESHOLDS = {
    "gold":   0.005,   # Top 0.5% — first place only in small competitions
    "silver": 0.05,    # Top 5%
    "bronze": 0.10,    # Top 10%
}

# 75 representative competitions (subset of MLE-bench)
def _metric_direction(metric: str) -> str:
    """Infer whether higher or lower scores are better for a given metric name."""
    lower_is_better = {"rmse", "mse", "mae", "rmsle", "log_loss", "logloss", "mcrmse", "wrmsse", "mape"}
    return "lower_is_better" if metric.lower() in lower_is_better else "higher_is_better"


PODIUM_BENCH_COMPETITIONS = [
    # Tabular — 30 competitions
    {"id": "titanic", "type": "tabular", "metric": "accuracy", "difficulty": "low"},
    {"id": "house-prices-advanced-regression-techniques", "type": "tabular", "metric": "rmse", "difficulty": "low"},
    {"id": "porto-seguro-safe-driver-prediction", "type": "tabular", "metric": "gini", "difficulty": "medium"},
    {"id": "santander-customer-transaction-prediction", "type": "tabular", "metric": "auc", "difficulty": "medium"},
    {"id": "ieee-fraud-detection", "type": "tabular", "metric": "auc", "difficulty": "high"},
    {"id": "amex-default-prediction", "type": "tabular", "metric": "amex_metric", "difficulty": "high"},
    {"id": "otto-group-product-classification-challenge", "type": "tabular", "metric": "log_loss", "difficulty": "medium"},
    {"id": "tabular-playground-series-jan-2022", "type": "tabular", "metric": "auc", "difficulty": "low"},
    {"id": "playground-series-s3e26", "type": "tabular", "metric": "auc", "difficulty": "medium"},
    # ... (full list in podium_bench_competitions.json)

    # Computer Vision — 20 competitions
    {"id": "dogs-vs-cats-redux-kernels-edition", "type": "cv", "metric": "log_loss", "difficulty": "low"},
    {"id": "plant-pathology-2020-fgvc7", "type": "cv", "metric": "roc_auc", "difficulty": "medium"},
    {"id": "rsna-pneumonia-detection-challenge", "type": "cv", "metric": "iou", "difficulty": "high"},

    # NLP — 15 competitions
    {"id": "tweet-sentiment-extraction", "type": "nlp", "metric": "jaccard", "difficulty": "medium"},
    {"id": "feedback-prize-english-language-learning", "type": "nlp", "metric": "mcrmse", "difficulty": "high"},
    {"id": "commonlit-readability-prize", "type": "nlp", "metric": "rmse", "difficulty": "medium"},

    # Time Series — 10 competitions
    {"id": "store-sales-time-series-forecasting", "type": "time_series", "metric": "rmsle", "difficulty": "medium"},
    {"id": "m5-forecasting-accuracy", "type": "time_series", "metric": "wrmsse", "difficulty": "high"},
]

# Inject metric_direction into each competition entry
for _comp in PODIUM_BENCH_COMPETITIONS:
    _comp.setdefault("metric_direction", _metric_direction(_comp["metric"]))


@dataclass
class CompetitionResult:
    competition_id: str
    competition_type: str
    agent_score: float
    leaderboard_percentile: float  # 0.0 = top, 1.0 = bottom
    medal: str | None  # "gold", "silver", "bronze", or None


def score_to_percentile(
    agent_score: float,
    leaderboard: list[float],
    direction: str = "higher_is_better",
) -> float:
    """Convert agent score to leaderboard percentile (0 = top)."""
    if not leaderboard:
        return 0.5
    if direction == "higher_is_better":
        rank = sum(1 for s in leaderboard if s > agent_score)
    else:
        rank = sum(1 for s in leaderboard if s < agent_score)
    return rank / len(leaderboard)


def percentile_to_medal(percentile: float, n_teams: int) -> str | None:
    """Assign medal based on leaderboard position."""
    if n_teams < 100:
        # Small competitions: absolute thresholds
        if percentile <= 0.01:
            return "gold"
        if percentile <= 0.10:
            return "silver"
        if percentile <= 0.20:
            return "bronze"
        return None

    if percentile <= MEDAL_THRESHOLDS["gold"]:
        return "gold"
    if percentile <= MEDAL_THRESHOLDS["silver"]:
        return "silver"
    if percentile <= MEDAL_THRESHOLDS["bronze"]:
        return "bronze"
    return None


def evaluate_agent(
    agent_fn,
    competition_ids: list[str] | None = None,
    leaderboard_dir: str = "./data/leaderboards",
    data_dir: str = "./data/bench_data",
    results_dir: str = "./results",
    time_budget_hours: float = 4.0,  # 4h per competition for evaluation
) -> dict:
    """
    Run agent on PodiumBench competitions and compute medal rate.
    agent_fn: callable(competition_id, data_path, time_budget_hours) → float (score)
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    competitions = PODIUM_BENCH_COMPETITIONS
    if competition_ids:
        competitions = [c for c in competitions if c["id"] in competition_ids]

    results = []
    for comp in tqdm(competitions, desc="PodiumBench evaluation"):
        comp_id = comp["id"]
        data_path = Path(data_dir) / comp_id
        lb_path = Path(leaderboard_dir) / f"{comp_id}.json"

        if not data_path.exists():
            logger.warning(f"Missing bench data for {comp_id}, skipping")
            continue

        # Run agent
        try:
            agent_score = agent_fn(comp_id, str(data_path), time_budget_hours)
        except Exception as e:
            logger.error(f"Agent failed on {comp_id}: {e}")
            agent_score = None

        # Load leaderboard
        leaderboard = []
        n_teams = 1000
        if lb_path.exists():
            with open(lb_path) as f:
                lb_data = json.load(f)
                leaderboard = lb_data.get("scores") or []
                n_teams = lb_data.get("n_teams", len(leaderboard))

        if agent_score is None:
            percentile = 1.0
            medal = None
        else:
            direction = comp.get("metric_direction", "higher_is_better")
            percentile = score_to_percentile(agent_score, leaderboard, direction)
            medal = percentile_to_medal(percentile, n_teams)

        result = CompetitionResult(
            competition_id=comp_id,
            competition_type=comp["type"],
            agent_score=agent_score or 0.0,
            leaderboard_percentile=percentile,
            medal=medal,
        )
        results.append(result)
        score_str = f"{agent_score:.4f}" if agent_score is not None else "None"
        logger.info(f"{comp_id}: score={score_str}, percentile={percentile:.1%}, medal={medal}")

    # Aggregate metrics
    medal_counts = {m: sum(1 for r in results if r.medal == m) for m in ["gold", "silver", "bronze"]}
    any_medal = sum(1 for r in results if r.medal is not None)
    avg_percentile = sum(r.leaderboard_percentile for r in results) / max(len(results), 1)

    summary = {
        "n_competitions": len(results),
        "medal_rate": any_medal / max(len(results), 1),
        "gold_rate": medal_counts["gold"] / max(len(results), 1),
        "silver_rate": medal_counts["silver"] / max(len(results), 1),
        "bronze_rate": medal_counts["bronze"] / max(len(results), 1),
        "avg_leaderboard_percentile": avg_percentile,
        "medal_counts": medal_counts,
        "by_type": _aggregate_by_type(results),
        "individual_results": [
            {
                "id": r.competition_id,
                "type": r.competition_type,
                "score": r.agent_score,
                "percentile": r.leaderboard_percentile,
                "medal": r.medal,
            }
            for r in results
        ],
    }

    # Save
    summary_path = Path(results_dir) / "podium_bench_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nPodiumBench Results:")
    logger.info(f"  Medal rate: {summary['medal_rate']:.1%}")
    logger.info(f"  Gold: {medal_counts['gold']} | Silver: {medal_counts['silver']} | Bronze: {medal_counts['bronze']}")
    logger.info(f"  Avg percentile: top {avg_percentile:.1%}")

    return summary


def _aggregate_by_type(results: list[CompetitionResult]) -> dict:
    by_type = {}
    for r in results:
        if r.competition_type not in by_type:
            by_type[r.competition_type] = {"total": 0, "medals": 0}
        by_type[r.competition_type]["total"] += 1
        if r.medal:
            by_type[r.competition_type]["medals"] += 1
    for t in by_type:
        total = by_type[t]["total"]
        by_type[t]["medal_rate"] = by_type[t]["medals"] / max(total, 1)
    return by_type


if __name__ == "__main__":
    import typer

    def main(
        model_path: str = "./checkpoints/dpo",
        results_dir: str = "./results/bench",
        time_budget_hours: float = 4.0,
    ):
        from agents.competition_runner import CompetitionRunner
        import asyncio

        runner = CompetitionRunner(model_path=model_path)

        def agent_fn(competition_id, data_path, time_budget):
            results = asyncio.run(runner.compete(
                competition_url=f"https://www.kaggle.com/c/{competition_id}",
                data_path=data_path,
                output_dir=f"./results/bench/{competition_id}",
                time_budget_hours=time_budget,
            ))
            return results.get("final_cv_score", 0)

        summary = evaluate_agent(
            agent_fn,
            results_dir=results_dir,
            time_budget_hours=time_budget_hours,
        )
        print(f"\nMedal rate: {summary['medal_rate']:.1%}")
        print(f"Avg percentile: top {summary['avg_leaderboard_percentile']:.1%}")

    typer.run(main)
