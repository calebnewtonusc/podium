"""
Prepare RL execution tasks for Stage 2 GRPO training.
Builds (prompt, data_path, target_column, metric, baseline_cv) tuples
from the validated training pairs. These are the tasks the model will
generate code for, execute, and receive CV reward signal.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
from loguru import logger
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from tqdm import tqdm


def compute_baseline_cv(
    data_path: Path,
    target_column: str,
    metric: str,
    n_folds: int = 3,
) -> float:
    """
    Compute a simple LightGBM baseline CV score for a dataset.
    This becomes the baseline that the RL agent must beat for positive reward.
    """
    train_path = data_path / "train.csv"
    if not train_path.exists():
        return 0.5

    train = pd.read_csv(train_path, nrows=100_000)
    if target_column not in train.columns:
        return 0.5

    target = train[target_column]
    feature_cols = [c for c in train.columns if c != target_column]
    X = train[feature_cols].select_dtypes("number").fillna(-999)

    if X.shape[1] == 0:
        return 0.5

    is_classification = target.nunique() <= 20
    cv = (
        StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        if is_classification
        else KFold(n_splits=n_folds, shuffle=True, random_state=42)
    )

    scores = []
    for train_idx, val_idx in cv.split(X, target):
        model = (
            lgb.LGBMClassifier(n_estimators=100, verbose=-1)
            if is_classification
            else lgb.LGBMRegressor(n_estimators=100, verbose=-1)
        )
        model.fit(X.iloc[train_idx], target.iloc[train_idx])

        if is_classification:
            proba = model.predict_proba(X.iloc[val_idx])
            if proba.shape[1] == 2:
                # Binary: use positive-class probability
                preds = proba[:, 1]
                scores.append(roc_auc_score(target.iloc[val_idx], preds))
            else:
                # Multi-class: use OvR macro AUC
                scores.append(
                    roc_auc_score(target.iloc[val_idx], proba, multi_class="ovr")
                )
        else:
            preds = model.predict(X.iloc[val_idx])
            scores.append(mean_squared_error(target.iloc[val_idx], preds) ** 0.5)

    return float(np.mean(scores))


RL_PROMPT_TEMPLATE = """\
<competition>
Type: {competition_type}
Metric: {metric}
Target column: {target_column}
Data description: {data_description}
Number of features: {n_features}
Training samples: {n_train}
Baseline CV score (simple LightGBM, numeric features only): {baseline_cv:.5f}

Task: Write Python code that loads /data/train.csv and improves the cross-validation score
above the baseline. Your code MUST set the variable `cv_score` at the end.

Apply advanced feature engineering, better models, or better CV strategy to beat the baseline.
</competition>

<think>"""


def build_rl_tasks(
    pairs_path: Path,
    bench_data_dir: Path,
    output_path: Path,
    max_tasks: int = 5000,
) -> None:
    """
    Build RL execution task JSONL from validated pairs + competition datasets.
    For each task: prompt + execution metadata (data_path, target, metric, baseline_cv).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(pairs_path) as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    tasks = []
    for pair in tqdm(pairs[:max_tasks], desc="Building RL tasks"):
        competition_id = pair.get("_competition", "")
        competition_type = pair.get("competition_type", "tabular")
        metric = pair.get("evaluation_metric", "auc")
        target_column = pair.get("target_column", "target")

        data_path = bench_data_dir / competition_id
        if not data_path.exists():
            continue

        # Compute baseline
        baseline_cv = compute_baseline_cv(data_path, target_column, metric)

        # Count features (read header only) and rows (count lines efficiently)
        try:
            header = pd.read_csv(data_path / "train.csv", nrows=0)
            n_features = len(header.columns) - 1
            with open(data_path / "train.csv") as _f:
                n_train = sum(1 for _ in _f) - 1  # subtract header line
        except Exception:
            n_features, n_train = 0, 0

        prompt = RL_PROMPT_TEMPLATE.format(
            competition_type=competition_type,
            metric=metric,
            target_column=target_column,
            data_description=pair.get("problem_summary", "Tabular competition dataset"),
            n_features=n_features,
            n_train=n_train,
            baseline_cv=baseline_cv,
        )

        tasks.append(
            {
                "prompt": prompt,
                "data_path": str(data_path),
                "target_column": target_column,
                "metric": metric,
                "baseline_cv": baseline_cv,
                "competition_type": competition_type,
                "metric_direction": "lower_is_better"
                if any(
                    kw in metric.lower()
                    for kw in ["loss", "error", "mse", "rmse", "mae"]
                )
                else "higher_is_better",
            }
        )

    logger.info(f"Built {len(tasks)} RL execution tasks → {output_path}")
    with open(output_path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")


if __name__ == "__main__":
    import typer

    def main(
        pairs: str = "./data/train/all_pairs_clean.jsonl",
        bench_data: str = "./data/bench_data",
        output: str = "./data/rl/competition_execution_tasks.jsonl",
        max_tasks: int = 5000,
    ):
        build_rl_tasks(Path(pairs), Path(bench_data), Path(output), max_tasks)

    typer.run(main)
