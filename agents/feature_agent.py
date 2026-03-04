"""
Feature Agent — iterative feature engineering for Kaggle competitions.
Generates feature sets, evaluates them via CV, keeps winners.
"""

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder


class FeatureAgent:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def engineer(
        self,
        data_path: Path,
        eda_results: dict,
        competition_type: str,
        metric: str,
        target_column: str,
        time_budget_hours: float = 48.0,
    ) -> dict:
        """
        Iterative feature engineering loop.
        Generates feature sets, validates each via quick LGBM CV, keeps winners.
        """
        deadline = time.time() + time_budget_hours * 3600
        train = pd.read_csv(data_path / "train.csv")
        test = pd.read_csv(data_path / "test.csv") if (data_path / "test.csv").exists() else None

        target = train[target_column]
        numeric_cols = eda_results.get("numeric_cols", [])
        categorical_cols = eda_results.get("categorical_cols", [])
        datetime_cols = eda_results.get("datetime_cols", [])
        high_card_cols = eda_results.get("high_cardinality_cols", [])

        # Encode categoricals
        train, test = self._encode_categoricals(train, test, categorical_cols)

        # Start with baseline features
        feature_cols = [c for c in numeric_cols if c != target_column]
        baseline_score, cv_fn = self._quick_cv(train, target, feature_cols, metric)
        logger.info(f"Baseline CV: {baseline_score:.5f} ({len(feature_cols)} features)")

        best_score = baseline_score
        best_feature_cols = feature_cols.copy()
        all_generated = []

        # Feature engineering iterations
        feature_generators = [
            ("group_stats", lambda: self._group_statistics(train, test, categorical_cols, numeric_cols[:10], target)),
            ("ratio_features", lambda: self._ratio_features(train, test, numeric_cols[:15])),
            ("polynomial", lambda: self._polynomial_features(train, test, eda_results.get("top_correlated_features", {}))),
            ("datetime_features", lambda: self._datetime_features(train, test, datetime_cols)),
            ("target_encoding", lambda: self._target_encoding(train, test, high_card_cols, target)),
            ("frequency_encoding", lambda: self._frequency_encoding(train, test, categorical_cols)),
            ("lag_features", lambda: self._lag_features(train, test, datetime_cols, numeric_cols[:5], target_column)),
        ]

        for gen_name, gen_fn in feature_generators:
            if time.time() > deadline:
                logger.info("Feature engineering time budget exhausted")
                break

            try:
                new_features = gen_fn()
            except Exception as e:
                logger.debug(f"Feature generator {gen_name} failed: {e}")
                continue

            if not new_features:
                continue

            # Test if new features improve CV
            trial_cols = best_feature_cols + new_features
            trial_score, _ = self._quick_cv(train, target, trial_cols, metric)
            improvement = trial_score - best_score

            if improvement > 0.0001:
                best_score = trial_score
                best_feature_cols = trial_cols
                all_generated.extend(new_features)
                logger.info(f"{gen_name}: +{improvement:.5f} → {best_score:.5f} ({len(new_features)} new features)")
            else:
                logger.debug(f"{gen_name}: no improvement ({trial_score:.5f} vs {best_score:.5f})")

        logger.info(f"Feature engineering complete. CV: {baseline_score:.5f} → {best_score:.5f}")
        logger.info(f"Features: {len(feature_cols)} → {len(best_feature_cols)} (+{len(all_generated)} generated)")

        return {
            "best_cv_score": best_score,
            "baseline_cv_score": baseline_score,
            "feature_cols": best_feature_cols,
            "generated_features": all_generated,
            "top_features": best_feature_cols[:20],
            "train_engineered": train,
            "test_engineered": test,
            "target_column": target_column,
        }

    def _quick_cv(
        self,
        train: pd.DataFrame,
        target: pd.Series,
        feature_cols: list[str],
        metric: str,
        n_folds: int = 3,
        n_estimators: int = 100,
    ) -> tuple[float, any]:
        """Fast 3-fold LGBM CV to evaluate feature sets."""
        valid_cols = [c for c in feature_cols if c in train.columns]
        if not valid_cols:
            return 0.0, None

        X = train[valid_cols].fillna(-999)
        y = target

        is_classification = y.nunique() <= 20
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42) if is_classification \
            else KFold(n_splits=n_folds, shuffle=True, random_state=42)

        params = {
            "n_estimators": n_estimators,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "verbose": -1,
            "n_jobs": 4,
        }

        scores = []
        metric_fn = _get_metric_fn(metric)

        for train_idx, val_idx in cv.split(X, y):
            model = lgb.LGBMClassifier(**params) if is_classification else lgb.LGBMRegressor(**params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx], categorical_feature="auto")
            if is_classification:
                proba = model.predict_proba(X.iloc[val_idx])
                if proba.shape[1] == 2:
                    # Binary: use positive-class probability
                    scores.append(metric_fn(y.iloc[val_idx], proba[:, 1]))
                else:
                    # Multi-class: OvR AUC, fall back to accuracy
                    try:
                        from sklearn.metrics import roc_auc_score as _auc
                        scores.append(_auc(y.iloc[val_idx], proba, multi_class="ovr"))
                    except Exception:
                        from sklearn.metrics import accuracy_score as _acc
                        scores.append(_acc(y.iloc[val_idx], np.argmax(proba, axis=1)))
            else:
                scores.append(metric_fn(y.iloc[val_idx], model.predict(X.iloc[val_idx])))

        return float(np.mean(scores)), metric_fn

    def _encode_categoricals(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame | None,
        categorical_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        for col in categorical_cols:
            if col not in train.columns:
                continue
            le = LabelEncoder()
            combined = pd.concat([train[col], test[col] if test is not None and col in test.columns else pd.Series()])
            le.fit(combined.astype(str).fillna("__NA__"))
            train[col] = le.transform(train[col].astype(str).fillna("__NA__"))
            if test is not None and col in test.columns:
                test[col] = le.transform(test[col].astype(str).fillna("__NA__"))
        return train, test

    def _group_statistics(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame | None,
        group_cols: list[str],
        value_cols: list[str],
        target: pd.Series,
    ) -> list[str]:
        new_features = []
        for g in group_cols[:3]:
            if g not in train.columns:
                continue
            for v in value_cols[:5]:
                if v not in train.columns:
                    continue
                for stat in ["mean", "std", "max", "min"]:
                    feat_name = f"{g}_{v}_{stat}"
                    agg = train.groupby(g)[v].transform(stat)
                    train[feat_name] = agg
                    if test is not None and g in test.columns and v in test.columns:
                        group_map = train.groupby(g)[v].agg(stat)
                        test[feat_name] = test[g].map(group_map).fillna(agg.mean())
                    new_features.append(feat_name)
        return new_features

    def _ratio_features(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame | None,
        numeric_cols: list[str],
    ) -> list[str]:
        new_features = []
        valid = [c for c in numeric_cols if c in train.columns][:8]
        for i, c1 in enumerate(valid):
            for c2 in valid[i+1:]:
                feat_name = f"{c1}_div_{c2}"
                train[feat_name] = train[c1] / (train[c2] + 1e-8)
                if test is not None:
                    test[feat_name] = test[c1] / (test[c2] + 1e-8)
                new_features.append(feat_name)
                if len(new_features) >= 20:
                    return new_features
        return new_features

    def _polynomial_features(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame | None,
        top_features: dict,
    ) -> list[str]:
        new_features = []
        top_cols = list(top_features.keys())[:5]
        valid = [c for c in top_cols if c in train.columns]
        for c in valid:
            feat_name = f"{c}_squared"
            train[feat_name] = train[c] ** 2
            if test is not None and c in test.columns:
                test[feat_name] = test[c] ** 2
            new_features.append(feat_name)

            log_name = f"{c}_log1p"
            train[log_name] = np.log1p(np.abs(train[c]))
            if test is not None and c in test.columns:
                test[log_name] = np.log1p(np.abs(test[c]))
            new_features.append(log_name)
        return new_features

    def _datetime_features(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame | None,
        datetime_cols: list[str],
    ) -> list[str]:
        new_features = []
        for col in datetime_cols:
            if col not in train.columns:
                continue
            try:
                train[col] = pd.to_datetime(train[col])
                if test is not None and col in test.columns:
                    test[col] = pd.to_datetime(test[col])
            except Exception:
                continue
            for part, feat_name in [
                ("year", f"{col}_year"),
                ("month", f"{col}_month"),
                ("day", f"{col}_day"),
                ("dayofweek", f"{col}_dow"),
                ("hour", f"{col}_hour"),
            ]:
                try:
                    train[feat_name] = getattr(train[col].dt, part)
                    if test is not None and col in test.columns:
                        test[feat_name] = getattr(test[col].dt, part)
                    new_features.append(feat_name)
                except Exception:
                    pass
        return new_features

    def _target_encoding(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame | None,
        high_card_cols: list[str],
        target: pd.Series,
        smoothing: float = 10.0,
    ) -> list[str]:
        """Smoothed (Bayesian) target encoding using global-mean regularization.
        Note: encoded on full train set, so CV scores for TE features will be
        slightly optimistic. Acceptable for feature selection purposes.
        """
        new_features = []
        global_mean = target.mean()
        for col in high_card_cols[:5]:
            if col not in train.columns:
                continue
            feat_name = f"{col}_te"
            stats = train.groupby(col)[target.name or "target"].agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
            train[feat_name] = train[col].map(smooth).fillna(global_mean)
            if test is not None and col in test.columns:
                test[feat_name] = test[col].map(smooth).fillna(global_mean)
            new_features.append(feat_name)
        return new_features

    def _frequency_encoding(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame | None,
        categorical_cols: list[str],
    ) -> list[str]:
        new_features = []
        for col in categorical_cols[:10]:
            if col not in train.columns:
                continue
            feat_name = f"{col}_freq"
            freq_map = train[col].value_counts(normalize=True)
            train[feat_name] = train[col].map(freq_map).fillna(0)
            if test is not None and col in test.columns:
                test[feat_name] = test[col].map(freq_map).fillna(0)
            new_features.append(feat_name)
        return new_features

    def _lag_features(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame | None,
        datetime_cols: list[str],
        value_cols: list[str],
        target_col: str,
    ) -> list[str]:
        """Generate lag/rolling features for time series data."""
        if not datetime_cols:
            return []
        new_features = []
        date_col = datetime_cols[0]
        try:
            train = train.sort_values(date_col)
        except Exception:
            return []

        # Combine train + test sorted by date so lag/rolling spans both splits correctly
        use_test = test is not None and date_col in test.columns
        if use_test:
            combined = pd.concat([train.assign(__podium_split__=0), test.assign(__podium_split__=1)]).sort_values(date_col)
        else:
            combined = train.sort_values(date_col)

        for col in value_cols[:3]:
            if col not in train.columns or col == target_col:
                continue
            for lag in [1, 7, 14, 30]:
                feat_name = f"{col}_lag_{lag}"
                combined[feat_name] = combined[col].shift(lag)
                new_features.append(feat_name)
            for window in [7, 30]:
                feat_name = f"{col}_rolling_mean_{window}"
                combined[feat_name] = combined[col].rolling(window, min_periods=1).mean()
                new_features.append(feat_name)

        # Write computed features back into the original dataframes using index alignment
        # (reindex avoids fragile positional alignment when indices differ after sort)
        train_rows = (combined[combined["__podium_split__"] == 0] if use_test else combined).sort_values(date_col)
        train.sort_values(date_col, inplace=True)
        for feat_name in new_features:
            train[feat_name] = train_rows[feat_name].reindex(train.index)
        if use_test:
            test_rows = combined[combined["__podium_split__"] == 1].sort_values(date_col)
            test.sort_values(date_col, inplace=True)
            for feat_name in new_features:
                test[feat_name] = test_rows[feat_name].reindex(test.index)

        return new_features


def _get_metric_fn(metric: str):
    from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss, accuracy_score
    metric = metric.lower()
    if "auc" in metric or "gini" in metric:
        return roc_auc_score
    if "rmse" in metric:
        return lambda y, p: -mean_squared_error(y, p) ** 0.5
    if "mse" in metric:
        return lambda y, p: -mean_squared_error(y, p)
    if "log" in metric and "loss" in metric:
        return lambda y, p: -log_loss(y, p)
    if "acc" in metric:
        return lambda y, p: accuracy_score(y, (np.array(p) > 0.5).astype(int))
    return roc_auc_score
