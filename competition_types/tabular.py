"""
Tabular competition specialist.
Optimal pipelines for structured/tabular competitions — Kaggle's most common type.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TabularStrategy:
    """Strategy config for a tabular competition."""
    use_target_encoding: bool = True
    use_group_stats: bool = True
    use_pseudo_labeling: bool = True
    primary_models: list[str] | None = None
    ensemble_strategy: str = "hill_climbing"
    # TODO (PO-69): cv_strategy is set by get_strategy() but is never read by ModelAgent or
    # CompetitionRunner.  ModelAgent always uses StratifiedKFold/KFold based solely on target
    # cardinality.  Wire this field through so time_series_split triggers TimeSeriesSplit in
    # ModelAgent._train_model_cv() instead of the default random-shuffle KFold.
    cv_strategy: str = "stratified_kfold"
    n_folds: int = 5

    def __post_init__(self):
        if self.primary_models is None:
            self.primary_models = ["lightgbm", "xgboost", "catboost"]


def get_strategy(competition_meta: dict) -> TabularStrategy:
    """
    Select tabular strategy based on competition characteristics.
    """
    n_rows = competition_meta.get("n_train", 0)
    n_features = competition_meta.get("n_features", 0)
    metric = competition_meta.get("metric", "auc").lower()
    has_time = competition_meta.get("has_datetime_cols", False)
    is_imbalanced = competition_meta.get("is_imbalanced", False)

    strategy = TabularStrategy()

    # Large dataset: add neural tabular
    if n_rows > 500_000:
        strategy.primary_models = ["lightgbm", "xgboost", "catboost", "neural_tabular"]

    # Small dataset: simpler models, avoid overfitting
    if n_rows < 5_000:
        strategy.primary_models = ["lightgbm", "ridge"]
        strategy.n_folds = 10  # More folds for small data

    # Time-based: switch to temporal CV
    if has_time:
        strategy.cv_strategy = "time_series_split"

    # Imbalanced: adjust fold strategy
    if is_imbalanced:
        strategy.cv_strategy = "stratified_kfold"

    # Regression: adjust ensemble
    if any(kw in metric for kw in ["rmse", "mse", "mae", "rmsle"]):
        strategy.use_target_encoding = False  # TE is for classification
        strategy.ensemble_strategy = "blend_weights"

    return strategy


TABULAR_SYSTEM_CONTEXT = """
You are in tabular competition specialist mode.

Key principles for tabular competitions:
1. Start with LightGBM as primary model — fastest iteration, usually competitive
2. Feature engineering is the biggest lever: group statistics, ratio features, target encoding
3. Always use 5-fold stratified CV — trust it over public LB unless CV/LB gap > 0.01
4. Ensemble 4-5 diverse models (LGBM + XGB + CatBoost + neural) via hill climbing OOF
5. Pseudo-labeling works well when test set is large (>3x train) and model confidence is high
6. For imbalanced classification: stratified folds, class_weight='balanced', don't oversample until after CV
7. For regression: check target distribution first — log1p transform if right-skewed
8. Target encoding must be done inside CV folds to prevent leakage

Common competition-winning features:
- Group aggregations (mean/std/max per category)
- Ratio and difference features between numeric columns
- Frequency encoding for high-cardinality categoricals
- Interaction features between top-importance columns
- Date/time decomposition (day of week, month, hour)
"""
