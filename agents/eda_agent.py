"""
EDA Agent — automated exploratory data analysis for Kaggle competitions.
Produces structured insights that feed into feature engineering decisions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import ks_2samp


class EDAAgent:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def analyze(
        self,
        data_path: Path,
        competition_type: str,
        metric: str,
        time_budget_hours: float = 4.0,
    ) -> dict:
        """
        Run full EDA on competition data.
        Returns structured insights dict consumed by FeatureAgent.
        """
        logger.info(f"EDA Agent starting on {data_path} ({competition_type})")

        train_path = data_path / "train.csv"
        test_path = data_path / "test.csv"

        if not train_path.exists():
            logger.warning(f"No train.csv found at {train_path}")
            return {"insights": [], "target_analysis": {}, "feature_stats": {}}

        train = pd.read_csv(train_path, nrows=500_000)
        test = pd.read_csv(test_path) if test_path.exists() else None

        insights = []

        # Identify target column
        target_col = self._detect_target(train, metric)
        non_feature_cols = {"id", "Id", "ID", target_col}

        # Feature column analysis
        feature_cols = [c for c in train.columns if c not in non_feature_cols]
        numeric_cols = train[feature_cols].select_dtypes(include="number").columns.tolist()
        categorical_cols = train[feature_cols].select_dtypes(include="object").columns.tolist()
        datetime_cols = self._detect_datetime_cols(train[feature_cols])

        # Missing value analysis
        missing = (train[feature_cols].isnull().mean() * 100).sort_values(ascending=False)
        high_missing = missing[missing > 30].to_dict()
        if high_missing:
            insights.append({
                "type": "missing_values",
                "severity": "high",
                "details": f"{len(high_missing)} features have >30% missing: {list(high_missing.keys())[:5]}",
                "action": "consider dropping or imputing with domain-specific logic",
            })

        # Target analysis
        target_analysis = self._analyze_target(train, target_col, metric)
        if target_analysis.get("is_imbalanced"):
            insights.append({
                "type": "class_imbalance",
                "severity": "high",
                "details": f"Target imbalance: minority class = {target_analysis['minority_pct']:.1f}%",
                "action": "use stratified K-fold, consider class weights or oversampling",
            })
        if target_analysis.get("is_skewed"):
            insights.append({
                "type": "target_skew",
                "severity": "medium",
                "details": f"Target skewness = {target_analysis['skewness']:.2f}",
                "action": f"apply log1p transform (metric: {metric})",
            })

        # Cardinality analysis
        high_cardinality = {
            c: train[c].nunique()
            for c in categorical_cols
            if train[c].nunique() > 50
        }
        if high_cardinality:
            insights.append({
                "type": "high_cardinality",
                "severity": "medium",
                "details": f"High cardinality features: {list(high_cardinality.keys())[:5]}",
                "action": "use target encoding or frequency encoding",
            })

        # Correlation with target
        if numeric_cols and target_col in train.columns:
            try:
                correlations = train[numeric_cols].corrwith(train[target_col]).abs().sort_values(ascending=False)
                top_features = correlations.head(10).to_dict()
            except Exception:
                top_features = {}
        else:
            top_features = {}

        # CV/LB leakage warning for time-based data
        if datetime_cols:
            insights.append({
                "type": "temporal_data",
                "severity": "high",
                "details": f"Time-based columns detected: {datetime_cols}",
                "action": "use time-series cross-validation (no shuffle), check for temporal leakage",
            })

        # Train/test distribution shift
        if test is not None:
            shift_features = self._detect_distribution_shift(train, test, numeric_cols[:20])
            if shift_features:
                insights.append({
                    "type": "distribution_shift",
                    "severity": "medium",
                    "details": f"Possible train/test shift in: {shift_features[:5]}",
                    "action": "adversarial validation, consider adversarial features",
                })

        logger.info(f"EDA complete: {len(insights)} insights, {len(feature_cols)} features")

        return {
            "target_column": target_col,
            "n_train": len(train),
            "n_test": len(test) if test is not None else 0,
            "feature_cols": feature_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "high_cardinality_cols": list(high_cardinality.keys()),
            "missing_value_cols": list(high_missing.keys()),
            "top_correlated_features": top_features,
            "target_analysis": target_analysis,
            "insights": insights,
        }

    def _detect_target(self, df: pd.DataFrame, metric: str) -> str:
        """Heuristic target column detection."""
        candidates = ["target", "label", "Target", "Label", "y", "price", "sales",
                      "salary", "survived", "Survived", "outcome", "class"]
        for c in candidates:
            if c in df.columns:
                return c
        # Last column is often target
        return df.columns[-1]

    def _analyze_target(self, df: pd.DataFrame, target_col: str, metric: str) -> dict:
        if target_col not in df.columns:
            return {}
        target = df[target_col].dropna()
        result = {}
        if target.dtype in ["int64", "float64"]:
            result["type"] = "regression" if target.nunique() > 20 else "classification"
            result["skewness"] = float(target.skew())
            result["is_skewed"] = abs(result["skewness"]) > 1.0
            result["mean"] = float(target.mean())
            result["std"] = float(target.std())
        else:
            result["type"] = "classification"
            result["is_skewed"] = False
        if result.get("type") == "classification":
            vc = target.value_counts(normalize=True)
            result["n_classes"] = len(vc)
            result["minority_pct"] = float(vc.min() * 100)
            result["is_imbalanced"] = result["minority_pct"] < 10.0
        else:
            result["is_imbalanced"] = False
        return result

    def _detect_datetime_cols(self, df: pd.DataFrame) -> list[str]:
        datetime_keywords = ["date", "time", "year", "month", "week", "day", "hour"]
        detected = []
        for c in df.columns:
            if any(kw in c.lower() for kw in datetime_keywords):
                detected.append(c)
        return detected

    def _detect_distribution_shift(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        cols: list[str],
    ) -> list[str]:
        """Simple KS test for distribution shift between train and test."""
        shifted = []
        for col in cols:
            if col not in test.columns:
                continue
            try:
                train_col = train[col].dropna()
                test_col = test[col].dropna()
                stat, p_val = ks_2samp(
                    train_col.sample(min(5000, len(train_col))),
                    test_col.sample(min(5000, len(test_col))),
                )
                if p_val < 0.01:
                    shifted.append(col)
            except Exception:
                pass
        return shifted
