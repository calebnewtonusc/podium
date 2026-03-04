# Podium Model Card

## Model Overview

| Property | Value |
|----------|-------|
| Model name | Podium-7B-v1 |
| Base model | Qwen2.5-7B-Coder-Instruct |
| Total parameters | 7.6B |
| Trainable parameters | ~168M (LoRA rank 64, 2.2%) |
| Context length | 16,384 tokens |
| Training stages | 3 (SFT → CV-RL → DPO) |
| Output format | `<think>` + `<code>` |
| License | MIT |
| HuggingFace | calebnewtonusc/podium-7b-v1 (post v1 release) |

---

## Training Data Summary

| Stream | Pairs | Focus |
|--------|-------|-------|
| Kaggle public notebooks | 280k | Code patterns, CV scores, EDA workflows |
| Winning solution writeups | 200k | Grandmaster causal reasoning |
| Community discussions | 160k | Technique Q&A, debugging patterns |
| Technique synthesis | 120k | ML methods → competition application |
| Meta-competition pairs | 40k | Competition type → optimal pipeline |
| **Total** | **800k** | |

---

## Capabilities

### Supported Competition Types
- **Tabular**: Regression, binary/multiclass classification, ranking
- **Computer Vision**: Image classification, object detection, segmentation, OCR
- **NLP**: Text classification, NER, question answering, generation, matching
- **Time Series**: Univariate/multivariate forecasting, anomaly detection
- **Multimodal**: Image+text, tabular+image, audio+text

### Core Pipeline Capabilities
- Exploratory data analysis (statistical analysis, distribution identification, anomaly flagging)
- Feature engineering (numeric transforms, interaction features, target encoding, lag features)
- Model selection (competition-type-aware architecture recommendations)
- Hyperparameter optimization (Optuna-guided search with competition-appropriate ranges)
- Cross-validation strategy selection (stratified K-fold, time-series split, group K-fold)
- Ensembling (stacking with OOF predictions, hill climbing selection, blend optimization)
- Pseudo-labeling (semi-supervised learning for large test sets)
- Submission generation and selection strategy

### Reasoning Capabilities
- Explains every decision with causal reasoning chains
- Identifies likely failure modes before they occur
- Flags data leakage risks
- Recommends time budget allocation across pipeline phases
- Suggests when to stop feature engineering and start ensembling

---

## Performance Targets (v1)

| Benchmark | Target | AIDE Baseline | MLE-Agent Baseline |
|-----------|--------|---------------|-------------------|
| PodiumBench any medal | >60% | ~17% | ~63% |
| PodiumBench gold | >10% | <1% | ~5% |
| Tabular medal rate | >75% | ~25% | ~70% |
| CV medal rate | >55% | ~12% | ~55% |
| NLP medal rate | >60% | ~15% | ~60% |
| Time series medal rate | >70% | ~20% | ~65% |
| Avg leaderboard percentile | Top 5% | ~49% | ~15% |

---

## Architecture

### Input Format
```
<competition>
  url: https://www.kaggle.com/c/competition-name
  type: tabular_classification
  metric: auc
  time_budget_hours: 168
  data_path: /data/
</competition>
```

### Output Format
```xml
<think>
  Competition analysis: This is a tabular binary classification problem
  with 50 features, 500k rows, and AUC metric. The target is heavily
  imbalanced (3% positive rate)...

  Key risks: Class imbalance, potential temporal leakage in transaction_date...

  Recommended approach: Start with LightGBM baseline, add group statistics
  features for card_id and merchant, use stratified K-fold with imbalance
  handling...
</think>

<code>
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Feature engineering
def engineer_features(df):
    # Velocity features (key insight: fraud clusters before card block)
    df['velocity_24h'] = df.groupby('card_id')['amount'].transform(
        lambda x: x.rolling(24, min_periods=1).sum()
    )
    # ... full implementation
</code>
```

---

## Limitations

### Known Limitations (v1)
- **Novel competition types**: Limited effectiveness on competitions with truly unprecedented structures (e.g., new scientific domains, unusual evaluation metrics)
- **Very large datasets**: Performance degrades on datasets >10GB without sufficient memory management
- **Code execution environment**: Requires Docker sandbox — does not self-execute in restricted environments
- **Competition-specific rules**: May not correctly parse unusual competition constraints from ambiguous problem statements
- **Real-time data**: Cannot handle competitions requiring live data fetching during inference

### Out of Scope (v1)
- Competitions requiring specialized scientific domain knowledge (e.g., genomics, seismology) beyond what's in training data
- Non-Python competition solutions
- Competitions with less than 50 prior similar historical competitions in training data

### Bias Notes
- Training data is heavily skewed toward competitions from 2019-2025 on Kaggle's English-language platform
- Tabular competitions are overrepresented in winning writeup training data
- May exhibit recency bias toward techniques that were popular in 2024-2025

---

## Ethical Considerations

Podium is designed to assist ML practitioners in learning and competing. Intended use cases:
- Learning ML techniques through Podium's reasoning chains
- Accelerating competition participation for researchers with limited time
- Democratizing access to grandmaster-level techniques (open weights)
- Research into automated ML engineering

Not intended for:
- Circumventing competition rules (e.g., team limits, submission limits)
- Replacing ML education with black-box outputs (use Podium's reasoning chains to learn)
