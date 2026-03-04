"""
Model Agent — trains a zoo of models with cross-validation and OOF predictions.
Knows which architectures to try based on competition type and data characteristics.
"""

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from loguru import logger
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold


optuna.logging.set_verbosity(optuna.logging.WARNING)


MODEL_ZOO = {
    "tabular": ["lightgbm", "xgboost", "catboost", "neural_tabular"],
    "tabular_regression": ["lightgbm", "xgboost", "catboost", "ridge"],
    "nlp": ["deberta", "roberta"],
    "computer_vision": ["efficientnet", "convnext", "vit"],
    "time_series": ["lightgbm", "xgboost", "lstm"],
    "multimodal": ["lightgbm", "efficientnet"],
}


class ModelAgent:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def train(
        self,
        data_path: Path,
        feature_results: dict,
        competition_type: str,
        metric: str,
        target_column: str,
        metric_direction: str = "higher_is_better",
        time_budget_hours: float = 55.0,
        n_folds: int = 5,
    ) -> dict:
        """
        Train model zoo with full K-fold CV.
        Returns OOF predictions for ensembling.
        """
        deadline = time.time() + time_budget_hours * 3600

        train_df = feature_results.get("train_engineered") or pd.read_csv(data_path / "train.csv")
        test_df = feature_results.get("test_engineered") or (
            pd.read_csv(data_path / "test.csv") if (data_path / "test.csv").exists() else None
        )
        feature_cols = feature_results.get("feature_cols", [])
        feature_cols = [c for c in feature_cols if c in train_df.columns]

        target = train_df[target_column]
        X = train_df[feature_cols].fillna(-999)
        X_test = test_df[feature_cols].fillna(-999) if test_df is not None else None

        is_classification = target.nunique() <= 20
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42) if is_classification \
            else KFold(n_splits=n_folds, shuffle=True, random_state=42)

        models_to_train = MODEL_ZOO.get(competition_type, MODEL_ZOO["tabular"])
        # Filter to tabular models if CV/NLP (those require separate handling)
        if competition_type in ("tabular", "tabular_regression"):
            pass
        else:
            models_to_train = ["lightgbm", "xgboost"]  # Fallback for non-tabular

        oof_predictions = {}
        test_predictions = {}
        cv_scores = {}
        submission_paths = {}

        for model_name in models_to_train:
            if time.time() > deadline:
                logger.info(f"Time budget exhausted after {model_name}")
                break

            logger.info(f"Training {model_name}...")
            try:
                oof_pred, test_pred, cv_score = self._train_model_cv(
                    model_name, X, target, X_test, cv, metric, metric_direction, n_folds
                )
                oof_predictions[model_name] = oof_pred
                test_predictions[model_name] = test_pred
                cv_scores[model_name] = cv_score
                logger.info(f"{model_name} CV: {cv_score:.5f}")

                # Save individual model submission
                if test_df is not None and X_test is not None:
                    sub_path = data_path.parent / "submissions" / f"{model_name}_submission.csv"
                    sub_path.parent.mkdir(exist_ok=True)
                    test_ids = test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
                    pd.DataFrame({"id": test_ids, "target": test_pred}).to_csv(sub_path, index=False)
                    submission_paths[model_name] = str(sub_path)

            except Exception as e:
                logger.warning(f"{model_name} failed: {e}")
                continue

        if not cv_scores:
            logger.error("All models failed to train — returning empty model results.")
            return {
                "best_cv_score": 0.0,
                "best_model": None,
                "cv_scores": {},
                "oof_predictions": {},
                "test_predictions": {},
                "submission_paths": {},
                "y_true": target.tolist(),
                "best_models": [],
            }

        # _get_metric_fn already normalizes all metrics to higher=better (negates lower-is-better
        # metrics), so always pick the maximum score regardless of metric_direction.
        best_model = max(cv_scores, key=cv_scores.get)

        logger.info(f"Model zoo complete. Best: {best_model} ({cv_scores.get(best_model, 0):.5f})")

        return {
            "best_cv_score": cv_scores.get(best_model, 0),
            "best_model": best_model,
            "cv_scores": cv_scores,
            "oof_predictions": oof_predictions,
            "test_predictions": test_predictions,
            "submission_paths": submission_paths,
            "y_true": target.tolist(),
            "best_models": sorted(cv_scores, key=cv_scores.get, reverse=True)[:3],
        }

    def _train_model_cv(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame | None,
        cv,
        metric: str,
        metric_direction: str,
        n_folds: int,
    ) -> tuple[np.ndarray, np.ndarray | None, float]:
        """Train one model with full K-fold CV. Returns (oof_preds, test_preds, cv_score)."""
        is_classification = y.nunique() <= 20
        oof = np.zeros(len(X))
        test_preds_list = []
        metric_fn = _get_metric_fn(metric)

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = _build_model(model_name, is_classification)
            _fit_model(model, model_name, X_tr, y_tr, X_val, y_val)

            oof[val_idx] = _predict(model, model_name, X_val, is_classification)

            if X_test is not None:
                test_preds_list.append(_predict(model, model_name, X_test, is_classification))

        cv_score = metric_fn(y, oof)
        test_pred = np.mean(test_preds_list, axis=0) if test_preds_list else None

        return oof, test_pred, cv_score


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _build_model(model_name: str, is_classification: bool):
    gpu = _has_gpu()
    if model_name == "lightgbm":
        cls = lgb.LGBMClassifier if is_classification else lgb.LGBMRegressor
        return cls(n_estimators=1000, learning_rate=0.05, num_leaves=127,
                   subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=8)
    if model_name == "xgboost":
        cls = xgb.XGBClassifier if is_classification else xgb.XGBRegressor
        kwargs = dict(n_estimators=1000, learning_rate=0.05, max_depth=6,
                      subsample=0.8, colsample_bytree=0.8,
                      eval_metric="logloss" if is_classification else "rmse",
                      tree_method="hist", verbosity=0)
        if gpu:
            kwargs["device"] = "cuda"
        return cls(**kwargs)
    if model_name == "catboost":
        cls = CatBoostClassifier if is_classification else CatBoostRegressor
        return cls(iterations=1000, learning_rate=0.05, depth=6,
                   eval_metric="AUC" if is_classification else "RMSE",
                   verbose=0, task_type="GPU" if gpu else "CPU")
    if model_name == "ridge":
        return Ridge(alpha=1.0)
    if model_name == "logistic":
        return LogisticRegression(max_iter=1000)
    raise ValueError(f"Unknown model: {model_name}")


def _fit_model(model, model_name: str, X_tr, y_tr, X_val, y_val):
    if model_name in ("lightgbm",):
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    elif model_name in ("xgboost",):
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    elif model_name in ("catboost",):
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    else:
        model.fit(X_tr, y_tr)


def _predict(model, model_name: str, X, is_classification: bool) -> np.ndarray:
    if is_classification and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # Binary: return positive-class probability (1D).
        # Multi-class: return full probability matrix (n_samples × n_classes).
        # Downstream metric functions (log_loss, roc_auc with multi-class) handle both.
        return proba[:, 1] if proba.shape[1] == 2 else proba
    return model.predict(X)


def _safe_roc_auc(y, p) -> float:
    """roc_auc_score that handles both binary (1D p) and multiclass (2D p)."""
    from sklearn.metrics import roc_auc_score
    if hasattr(p, "ndim") and p.ndim == 2:
        return roc_auc_score(y, p, multi_class="ovr")
    return roc_auc_score(y, p)


def _get_metric_fn(metric: str):
    from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
    metric = metric.lower()
    if "auc" in metric or "gini" in metric:
        return _safe_roc_auc
    if "rmse" in metric or "rmsle" in metric:
        return lambda y, p: -mean_squared_error(y, p) ** 0.5
    if "mse" in metric or "mae" in metric:
        return lambda y, p: -mean_squared_error(y, p)
    if "log" in metric or "logloss" in metric:
        return lambda y, p: -log_loss(y, p)
    if "acc" in metric:
        return accuracy_score
    # Unknown metric: fall back to _safe_roc_auc rather than bare roc_auc_score
    # so multi-class predictions (2D proba arrays) are handled correctly.
    logger.warning(f"Unknown metric '{metric}', defaulting to AUC. Update _get_metric_fn() if incorrect.")
    return _safe_roc_auc
