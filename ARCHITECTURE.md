# Podium Architecture

## Core Insight

Every existing Kaggle AI system (AIDE, AutoKaggle, SELA, R&D-Agent) makes the same architectural mistake: they use general-purpose LLMs (GPT-4, o1, Claude) as the reasoning engine and add scaffolding around them. The scaffolding helps — AIDE achieves 3× more medals than unscaffolded approaches — but hits a hard ceiling because the underlying model's knowledge of Kaggle techniques, competition patterns, and domain expertise is incidental, not primary.

Podium's thesis: **the techniques that win Kaggle competitions are learnable patterns, and they should live in model weights, not in system prompts**.

The analogy: Nalana bakes 10,000 hours of Blender expert tutorials into weights and outperforms "ask GPT-4 how to model this." Podium bakes 2M+ Kaggle notebooks, 50,000+ winning solution writeups, and every grandmaster discussion post into weights and outperforms "ask o1 to win this competition."

---

## 4-Phase Product Vision

### Phase 1 — COMPETE (v1, Q3 2026)
Core competition agent. Given a Kaggle competition URL and time budget, Podium autonomously runs the full pipeline: EDA → feature engineering → model selection → training → ensembling → submission. Targets medal-rate performance superior to any existing system.

### Phase 2 — ADAPT (v1.5, Q3 2026)
Cross-competition meta-learning. Podium identifies structural similarities between new competitions and solved historical ones, transferring winning strategies directly. "This looks like the 2023 tabular playground series — here's what worked."

### Phase 3 — ORCHESTRATE (v2, Q4 2026)
Multi-agent parallel exploration. Dozens of specialized sub-agents explore different approaches simultaneously: one tries gradient boosting, another tries neural tabular, another tries deep feature engineering — results fused via learned ensembling.

### Phase 4 — GENERALIZE (v3, 2027)
Real-world ML engineering beyond Kaggle. Handles production datasets, messy data, client requirements. The competition-honed skills transfer to real enterprise ML challenges.

---

## 7 Technical Differentiators

### 1. Specialist Weights (Not Scaffolded GPT-4)
**The first model with Kaggle expertise in the weights.**

AIDE wraps o1 in a tree search. The o1 model knows about gradient boosting the same way it knows about cooking recipes — surface-level pattern matching from pretraining. Podium trains on Kaggle-specific data until it internalizes the difference between a competition where `log1p` transform dramatically improves score versus one where it doesn't, what the top-10 on the leaderboard was using in the 2024 Optiver competition, and which feature interactions are worth trying in fraud detection.

### 2. CV Score as RL Reward — Free, Verifiable, Objective
**The same insight as DeepSeek-R1, applied to Kaggle.**

DeepSeek-R1's key innovation: mathematical reasoning has a free verifiable reward signal (is the answer correct?), enabling RL without human labelers. Kaggle has the same property: cross-validation score is objective, computable, and perfectly aligned with competition performance.

Training loop:
```
Generate code → Execute in Docker sandbox → Run K-fold CV → Score → Reward = ΔCV
```

No human raters. No reward hacking. Pure signal.

This enables Stage 2 training (GRPO) to directly optimize for what matters: leaderboard improvement.

### 3. Competition-Type Specialization
**Five specialist modes, one unified model.**

Winning a tabular competition (XGBoost ensembles, extensive feature engineering) requires completely different techniques than winning a CV competition (ViT fine-tuning, TTA, progressive resizing). A general model averages over these; Podium routes to specialist modes:

- **Tabular**: Gradient boosting meta-learner, lag/ratio/interaction features, target encoding
- **CV**: Architecture selection (EffNet/ViT/ConvNeXt), augmentation strategy, pseudo-labeling
- **NLP**: Backbone selection (DeBERTa/RoBERTa), pooling strategies, prompt construction
- **Time Series**: Lag feature factory, LGBM + neural ensembles, temporal CV strategy
- **Multimodal**: Fusion architecture selection, modality-specific preprocessing, alignment strategies

Each specialist mode has dedicated training data and routing logic based on competition structure detection.

### 4. Grand Master Reasoning Training
**Trained on the "why", not just the "what".**

A Kaggle notebook that gets 0.91 AUC says: "I added these features and the score went up." A Grandmaster's winning writeup says: "The key insight was that transaction velocity over 24h windows captures fraud behavior better than raw counts because fraudsters cluster transactions before card blocks — this is why the log-velocity feature gave the biggest single lift."

Podium is trained on 50,000+ competition winning solution writeups where grandmasters explain causal reasoning behind every decision. This is the difference between memorizing answers and understanding the domain.

### 5. Solution Memory and Cross-Competition Transfer
**Every solved competition teaches Podium something permanent.**

After each competition, Podium distills learned patterns into a persistent knowledge base:
- Which feature engineering tricks transferred from domain X
- Which model architectures the community converged on
- What the gold medalists did that silver medalists missed

New competitions are embedded and matched against this memory. Structural similarities trigger direct strategy transfer with adaptation.

### 6. Ensemble Orchestration as First-Class Knowledge
**Ensembling is a skill, not an afterthought.**

Top Kaggle solutions almost always involve sophisticated ensembling: stacking with out-of-fold predictions, hill climbing ensemble selection, optuna-optimized blend weights, rank averaging for robust combination. Podium is trained on the *strategy* of ensembling — when to blend vs. stack, how many models to include, how to select diversity vs. performance tradeoff.

### 7. Competition Lifecycle Awareness
**Podium understands the 4 phases of a Kaggle competition.**

Week 1: Establish baselines, understand data quirks, find CV/LB correlation.
Weeks 2-3: Feature engineering exploration, model zoo building.
Week 4+: Ensemble construction, pseudo-labeling, last-mile optimization.
Final 24h: Submission selection strategy (avoid overfitting to public LB).

Podium allocates its time budget accordingly, not just running the same pipeline loop repeatedly.

---

## Universal Competition DSL

Podium uses an intermediate representation called **Universal Competition Description (UCD)** — a structured JSON format that captures a competition's essential characteristics independent of the raw Kaggle page format.

```json
{
  "competition_id": "otto-group-product-classification",
  "type": "tabular_classification",
  "evaluation_metric": "log_loss",
  "target": "target",
  "features": {
    "n_numeric": 93,
    "n_categorical": 0,
    "n_text": 0,
    "n_image": 0
  },
  "dataset": {
    "train_rows": 61878,
    "test_rows": 144368,
    "target_distribution": "9-class_balanced"
  },
  "constraints": {
    "time_budget_hours": 168,
    "submission_limit_daily": 5
  },
  "historical_similar": ["tabular-playground-2022-10", "porto-seguro"],
  "recommended_approach": {
    "primary": "gradient_boosting_ensemble",
    "secondary": "neural_tabular",
    "key_techniques": ["stacking", "feature_engineering", "pseudo_labeling"]
  }
}
```

All specialist agents operate on UCD rather than raw competition HTML, enabling consistent reasoning and cross-competition transfer.

---

## Training Pipeline

### Data Streams → Pairs

```
Stream 1: Kaggle Notebooks (35%)
  Raw notebook (JSON) → Claude extracts: [competition context, approach reasoning, code, CV score]
  → (Competition brief + EDA context) → (Solution code + score) pairs

Stream 2: Winning Solution Writeups (25%)
  Post-competition discussion posts → (Competition summary) → (Strategic decision + reasoning) pairs

Stream 3: Community Discussions (20%)
  Q&A threads → (Question + competition context) → (Expert answer + technique explanation)

Stream 4: Technique Synthesis (15%)
  ML papers + grandmaster guides → (Technique description) → (Competition application + code)

Stream 5: Meta-Competition Pairs (5%)
  Competition type taxonomy → (Competition profile) → (Optimal pipeline + full reasoning chain)
```

### Stage 1: Supervised Fine-Tuning (SFT)
- **Base model**: Qwen2.5-7B-Coder-Instruct
- **Data**: ~500k curated (competition, solution) pairs
- **LoRA**: rank 64, alpha 128, targeting all attention + MLP layers
- **Context**: 16,384 tokens (full notebook context)
- **Duration**: ~6 hours on 18× A6000 with DeepSpeed ZeRO-3
- **Goal**: Model learns to produce syntactically correct, domain-appropriate ML code

### Stage 2: CV-Verified Reinforcement Learning (GRPO)
- **Reward signal**: Cross-validation score improvement vs. baseline
- **Execution harness**: Docker sandbox with sklearn, XGBoost, LightGBM, PyTorch, timm
- **Reward formula**: `r = min((cv_score - baseline_cv) / baseline_cv * 10, 1.0)` — scaled 10×, clamped to [0, 1]
- **Penalty**: -1.0 for code that fails to execute, -0.5 for code that runs but scores below baseline
- **Duration**: ~4 hours on 18× A6000
- **Goal**: Model learns to generate code that actually improves scores, not just looks plausible

### Stage 3: DPO (Preference Optimization)
- **Preference pairs**: Human-ranked competition approaches (grandmaster writeup vs. naive approach)
- **Focus**: When to do more EDA vs. when to jump to modeling, when to stop feature engineering
- **Duration**: ~2 hours
- **Goal**: Model learns competition strategy and time allocation, not just code quality

---

## Multi-Agent Orchestration

```
Competition Runner (Orchestrator)
├── Competition Analyzer
│   ├── parse_competition_page()
│   ├── detect_competition_type()
│   └── build_ucd()
├── EDA Agent
│   ├── statistical_analysis()
│   ├── correlation_mining()
│   ├── target_analysis()
│   └── anomaly_detection()
├── Feature Agent
│   ├── numeric_transforms()
│   ├── interaction_features()
│   ├── target_encoding()
│   └── domain_features()
├── Model Agent
│   ├── baseline_establishment()
│   ├── architecture_selection()
│   ├── hyperparameter_search()
│   └── cross_validation()
├── Ensemble Agent
│   ├── oof_stacking()
│   ├── hill_climbing_selection()
│   ├── blend_weight_optimization()
│   └── pseudo_labeling()
└── Submission Agent
    ├── public_lb_tracking()
    ├── submission_selection()
    └── final_submission()
```

Each agent is a specialized call to the Podium model with role-specific system context. The orchestrator manages the time budget, agent handoffs, and result aggregation.

---

## Model Specification

| Property | Value |
|----------|-------|
| Base model | Qwen2.5-7B-Coder-Instruct |
| Total parameters | 7.6B |
| Trainable (LoRA) | ~168M (2.2%) |
| Context length | 16,384 tokens |
| LoRA rank | 64 |
| Output format | `<think>` + `<code>` |
| Quantization (inference) | 4-bit GPTQ |
| Serving | vLLM with PagedAttention |
| Latency (first token) | <200ms |

---

## Deployment Stack

```
vLLM (model serving, GPU)
    ↕
FastAPI (REST + WebSocket streaming)
    ↕
Redis (competition session state, solution cache)
    ↕
Nginx (reverse proxy, SSL termination)
    ↕
Docker Compose (one-command deployment)
```

---

## PodiumBench

Custom benchmark evaluating Podium on 75 Kaggle competitions drawn from MLE-bench, stratified by:
- Competition type (tabular/CV/NLP/TS/multimodal)
- Difficulty (measured by required domain expertise)
- Era (2019-2025, testing generalization to new competition styles)

Metrics:
- **Medal rate**: % of competitions achieving bronze/silver/gold
- **Percentile**: Average leaderboard percentile achieved
- **Gold rate**: % of competitions achieving gold medal position
- **Submission efficiency**: Medals per submission attempt
