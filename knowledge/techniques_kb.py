"""
Techniques knowledge base — structured catalog of ML competition techniques.
Used to prime agents with relevant technique context based on competition type.
"""

from dataclasses import dataclass


@dataclass
class Technique:
    name: str
    domain: list[str]  # Competition types this applies to
    impact: str  # "high" / "medium" / "low"
    when: str  # Conditions under which to apply
    how: str  # Implementation summary
    pitfall: str  # Common mistake
    typical_gain: str  # Expected CV improvement


TECHNIQUES_KB: list[Technique] = [
    # ── Ensembling ──────────────────────────────────────────────────────────
    Technique(
        name="Hill Climbing Ensemble Selection",
        domain=["tabular", "nlp", "computer_vision", "time_series"],
        impact="high",
        when="You have 3+ trained models with OOF predictions. Final ensemble construction.",
        how="Start with best model. Greedily add models that improve ensemble CV. Normalize weights.",
        pitfall="Can overfit to validation set if applied on full training data without holdout.",
        typical_gain="+0.002 to +0.008 AUC over best single model",
    ),
    Technique(
        name="OOF Stacking",
        domain=["tabular", "nlp"],
        impact="high",
        when="Diverse model zoo exists (LGBM + XGB + Neural). Dataset is large enough (>10k rows).",
        how="Generate OOF predictions from each base model. Train Ridge/LogReg meta-learner on OOF stack.",
        pitfall="Meta-learner overfits with small datasets. Use Ridge (not XGB) as meta-learner.",
        typical_gain="+0.001 to +0.005 over simple blend",
    ),
    Technique(
        name="Pseudo-Labeling",
        domain=["computer_vision", "nlp", "tabular"],
        impact="medium",
        when="Large test set (>3x train), model confidence is high on subset (>95% probability).",
        how="Generate predictions on test. Keep only high-confidence samples. Retrain with train + pseudo-labeled test.",
        pitfall="Propagates model errors. Use confidence threshold ≥0.95. Only 1-2 rounds max.",
        typical_gain="+0.003 to +0.01 AUC in CV/NLP",
    ),
    # ── Feature Engineering ─────────────────────────────────────────────────
    Technique(
        name="Smoothed Target Encoding",
        domain=["tabular"],
        impact="high",
        when="High-cardinality categorical features (>20 unique values) in classification or regression.",
        how="mean_encoded = (count * category_mean + m * global_mean) / (count + m) where m=10.",
        pitfall="MUST be computed inside CV folds to prevent leakage. Never fit on full training data.",
        typical_gain="+0.003 to +0.015 AUC for high-cardinality features",
    ),
    Technique(
        name="Group Statistics Features",
        domain=["tabular"],
        impact="high",
        when="Dataset has meaningful grouping variables (user_id, store_id, product_category).",
        how="For each group × value pair: mean, std, min, max, count, nunique. Use transform() to broadcast back.",
        pitfall="If grouping variable leaks target info, creates data leakage. Validate carefully.",
        typical_gain="+0.005 to +0.02 AUC for datasets with strong group structure",
    ),
    Technique(
        name="Lag Features",
        domain=["time_series"],
        impact="high",
        when="Sequential/time-ordered data. Most important feature engineering for TS competitions.",
        how="shift(1), shift(7), shift(28), rolling(7).mean(), rolling(28).std() — always after sorting by time.",
        pitfall="Must shift AFTER sorting by time. Never compute rolling stats including the target row.",
        typical_gain="Essential — baseline TS models without lags are typically 30-50% worse",
    ),
    # ── Validation ──────────────────────────────────────────────────────────
    Technique(
        name="Adversarial Validation",
        domain=["tabular", "computer_vision", "nlp"],
        impact="medium",
        when="CV/LB gap is unexpected. Suspecting train/test distribution shift.",
        how="Label train=0, test=1. Train classifier. If AUC > 0.6, significant shift exists.",
        pitfall="Identifies shift but doesn't fix it. Use as diagnostic. Remove shifting features or reweight.",
        typical_gain="Diagnostic tool — finding and fixing shift can improve LB by 0.005-0.02",
    ),
    Technique(
        name="Purged K-Fold",
        domain=["time_series"],
        impact="high",
        when="Financial or time-ordered data with autocorrelation. Standard CV leaks future.",
        how="Add gap between train and validation (e.g., 2-week gap). Prevents lookahead.",
        pitfall="Without purging, CV is optimistic by 0.01-0.05 in financial time series.",
        typical_gain="More accurate CV (prevents overfit to LB chasing)",
    ),
    # ── Deep Learning ────────────────────────────────────────────────────────
    Technique(
        name="Multi-Sample Dropout",
        domain=["nlp", "computer_vision"],
        impact="medium",
        when="Transformer fine-tuning, especially for regression tasks.",
        how="Apply 5 different dropout masks to the pooled output. Average the 5 logits. Backprop through all.",
        pitfall="Increases memory usage 5x. Reduce batch size accordingly.",
        typical_gain="+0.001 to +0.003 AUC/Pearson correlation",
    ),
    Technique(
        name="Test-Time Augmentation (TTA)",
        domain=["computer_vision"],
        impact="medium",
        when="Image classification or segmentation. Inference-time improvement with no retraining.",
        how="Generate 5-10 augmented versions of each test image. Average predictions.",
        pitfall="Increases inference time proportionally. Diminishing returns after 5-10 augmentations.",
        typical_gain="+0.002 to +0.005 AUC / +0.5-1% accuracy",
    ),
    Technique(
        name="Log1p Target Transform",
        domain=["tabular", "time_series"],
        impact="high",
        when="Right-skewed regression targets. RMSLE metric. Price, count, or duration targets.",
        how="y_train = np.log1p(y_train). Predict, then inverse: y_pred = np.expm1(y_pred).",
        pitfall="Must inverse-transform predictions. For RMSE metric, check if log transform helps empirically.",
        typical_gain="+3-15% RMSE reduction for right-skewed targets",
    ),
]


def get_techniques_for_competition(
    competition_type: str, n_top: int = 5
) -> list[Technique]:
    """Return most relevant techniques for a competition type."""
    relevant = [
        t for t in TECHNIQUES_KB if competition_type in t.domain or "all" in t.domain
    ]
    high_impact = [t for t in relevant if t.impact == "high"]
    medium_impact = [t for t in relevant if t.impact == "medium"]
    return (high_impact + medium_impact)[:n_top]


def format_for_context(techniques: list[Technique]) -> str:
    """Format techniques as a context string for agent prompts."""
    lines = ["Relevant techniques for this competition:"]
    for t in techniques:
        lines.append(f"\n**{t.name}** ({t.impact} impact, {t.typical_gain})")
        lines.append(f"  When: {t.when}")
        lines.append(f"  Pitfall: {t.pitfall}")
    return "\n".join(lines)
