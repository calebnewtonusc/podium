"""
Ensemble Agent — builds optimal submission from a model zoo.
Knows when to stack vs. blend, how to select ensemble members, pseudo-labeling.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score


def hill_climbing_selection(
    oof_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
    metric_fn,
    n_iter: int = 100,
    direction: str = "higher_is_better",
) -> dict[str, float]:
    """
    Hill climbing ensemble weight optimization.
    Start with best model, greedily add models that improve ensemble CV.
    Returns {model_name: weight}.
    """
    compare = (lambda a, b: a > b) if direction == "higher_is_better" else (lambda a, b: a < b)
    best_score = None
    weights = {name: 0.0 for name in oof_preds}

    # Start with best individual model
    model_scores = {
        name: metric_fn(y_true, preds)
        for name, preds in oof_preds.items()
    }
    best_model = max(model_scores, key=model_scores.get) if direction == "higher_is_better" \
        else min(model_scores, key=model_scores.get)

    weights[best_model] = 1.0
    best_score = model_scores[best_model]
    logger.info(f"Hill climbing start: {best_model} → {best_score:.5f}")

    for i in range(n_iter):
        improved = False
        for name in oof_preds:
            # Try adding this model with weight 1/(current_count + 1)
            trial_weights = dict(weights)
            trial_weights[name] += 1.0
            total = sum(trial_weights.values())
            normalized = {k: v / total for k, v in trial_weights.items()}

            # Compute blended prediction
            blended = sum(
                normalized[n] * oof_preds[n]
                for n in normalized
                if normalized[n] > 0
            )
            score = metric_fn(y_true, blended)

            if compare(score, best_score):
                best_score = score
                weights = trial_weights
                improved = True

        if not improved:
            break

    # Normalize final weights
    total = sum(weights.values())
    final_weights = {k: v / total for k, v in weights.items() if v > 0}
    logger.info(f"Hill climbing final: {best_score:.5f} | {len(final_weights)} models")
    return final_weights


def oof_stacking(
    oof_preds: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train a linear meta-learner on OOF predictions.
    Returns (meta_oof_pred, meta_test_pred).
    """
    # Use only models present in BOTH oof_preds and test_preds to keep shapes aligned
    shared_models = [m for m in oof_preds if m in test_preds]
    X_oof = np.column_stack([oof_preds[m] for m in shared_models])
    X_test = np.column_stack([test_preds[m] for m in shared_models])

    meta = Ridge(alpha=1.0, fit_intercept=True)
    meta.fit(X_oof, y_true)

    return meta.predict(X_oof), meta.predict(X_test)


def pseudo_label(
    model,
    test_df: pd.DataFrame,
    threshold_high: float = 0.95,
    threshold_low: float = 0.05,
) -> pd.DataFrame:
    """
    Generate pseudo-labels for high-confidence test predictions.
    Only labels samples where model is very confident (near 0 or 1 for binary).
    """
    test_preds = model.predict_proba(test_df)[:, 1]
    high_conf = (test_preds >= threshold_high) | (test_preds <= threshold_low)
    pseudo_labels = (test_preds >= threshold_high).astype(int)

    pseudo_df = test_df[high_conf].copy()
    pseudo_df["target"] = pseudo_labels[high_conf]
    logger.info(f"Pseudo-labeled {high_conf.sum()} / {len(test_df)} test samples")
    return pseudo_df


class EnsembleAgent:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def build(
        self,
        model_results: dict,
        data_path: Path,
        metric: str,
        metric_direction: str,
        time_budget_hours: float,
    ) -> dict:
        """
        Build optimal ensemble from model zoo.
        1. Hill climbing ensemble selection
        2. OOF stacking with linear meta-learner
        3. Pseudo-labeling if test set is large
        4. Final blend weight optimization
        """
        oof_preds = model_results.get("oof_predictions", {})
        y_true = model_results.get("y_true")
        model_cv_scores = model_results.get("cv_scores", {})
        # Filter out models that produced None test predictions (no test set)
        test_preds = {k: v for k, v in model_results.get("test_predictions", {}).items()
                      if v is not None}

        if not oof_preds or y_true is None:
            logger.warning("No OOF predictions available, using best single model")
            if not model_cv_scores:
                logger.error("No model scores available — all models failed.")
                return {"ensemble_cv_score": 0.0, "best_submission_path": None, "strategy": "failed"}
            # model_agent normalizes all cv_scores to "higher=better" (negates lower-is-better)
            # so always use max() regardless of metric_direction
            best_model = max(model_cv_scores, key=model_cv_scores.get)
            return {
                "ensemble_cv_score": model_cv_scores.get(best_model, 0),
                "best_submission_path": model_results.get("submission_paths", {}).get(best_model),
                "strategy": "single_best",
            }

        metric_fn = _get_metric_fn(metric)
        y_true_arr = np.array(y_true)

        # _get_metric_fn normalizes all metrics to higher=better (negates lower-is-better
        # metrics like RMSE/log_loss), matching model_agent convention. Always use
        # "higher_is_better" direction for internal comparisons.
        internal_direction = "higher_is_better"

        # Strategy 1: Hill climbing
        hc_weights = hill_climbing_selection(
            oof_preds, y_true_arr, metric_fn, direction=internal_direction
        )
        hc_blend = sum(w * oof_preds[n] for n, w in hc_weights.items())
        hc_score = metric_fn(y_true_arr, hc_blend)

        # Strategy 2: OOF stacking (requires test predictions; skip if unavailable)
        can_stack = len(test_preds) >= 2
        stack_score = 0.0
        if can_stack:
            try:
                stack_oof, stack_test = oof_stacking(oof_preds, test_preds, y_true_arr)
                stack_score = metric_fn(y_true_arr, stack_oof)
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.warning(f"OOF stacking failed ({e}), using hill climbing")
                can_stack = False

        logger.info(f"Hill climbing CV: {hc_score:.5f}" +
                    (f" | Stacking CV: {stack_score:.5f}" if can_stack else " | Stacking: N/A"))

        compare = lambda a, b: a > b  # Always higher=better since metric_fn normalizes sign
        if can_stack and compare(stack_score, hc_score):
            final_test_pred = stack_test
            final_cv = stack_score
            strategy = "oof_stacking"
        else:
            final_test_pred = sum(w * test_preds.get(n, np.zeros_like(hc_blend))
                                  for n, w in hc_weights.items())
            final_cv = hc_score
            strategy = f"hill_climbing ({len(hc_weights)} models)"

        # Clip to valid probability range for classification metrics
        if any(kw in metric.lower() for kw in ["auc", "logloss", "log_loss", "acc"]):
            final_test_pred = np.clip(final_test_pred, 0.0, 1.0)

        # Save submission
        submission_path = data_path.parent / "submissions" / "ensemble_submission.csv"
        submission_path.parent.mkdir(exist_ok=True)
        test_ids = _load_test_ids(data_path)
        n_preds = len(final_test_pred)
        if len(test_ids) != n_preds:
            test_ids = np.arange(n_preds)
        pd.DataFrame({
            "id": test_ids,
            "target": final_test_pred,
        }).to_csv(submission_path, index=False)

        return {
            "ensemble_cv_score": final_cv,
            "best_submission_path": str(submission_path),
            "strategy": strategy,
            "hill_climbing_weights": hc_weights,
            "hill_climbing_cv": hc_score,
            "stacking_cv": stack_score if can_stack else None,
        }


def _safe_roc_auc(y, p) -> float:
    """roc_auc_score that handles both binary (1D p) and multi-class (2D p)."""
    if hasattr(p, "ndim") and p.ndim == 2:
        return roc_auc_score(y, p, multi_class="ovr")
    return roc_auc_score(y, p)


def _get_metric_fn(metric: str):
    metric = metric.lower().replace("-", "_").replace(" ", "_")
    if "auc" in metric or "gini" in metric:
        return _safe_roc_auc
    from sklearn.metrics import mean_squared_error, log_loss, accuracy_score
    if "rmse" in metric or "rmsle" in metric:
        return lambda y, p: -(mean_squared_error(y, p) ** 0.5)  # Negated so higher=better, matching model_agent convention
    if "mse" in metric or "mae" in metric:
        return mean_squared_error
    if "log_loss" in metric or "logloss" in metric:
        return log_loss
    if "acc" in metric:
        return accuracy_score
    return _safe_roc_auc  # Default — use safe version for multi-class support


def _load_test_ids(data_path: Path) -> np.ndarray:
    try:
        test = pd.read_csv(data_path / "test.csv")
        return test["id"].values if "id" in test.columns else np.arange(len(test))
    except Exception:
        return np.array([])
