# Podium

**You compete. Podium wins.**

Podium is the world's first trained specialist model for winning Kaggle competitions. Unlike existing systems (AIDE, AutoKaggle, SELA) that scaffold general-purpose LLMs with prompting strategies, Podium bakes Kaggle grandmaster knowledge directly into model weights — trained on 2M+ notebooks, every winning solution writeup, and every competition discussion thread ever posted, with cross-validation score as a free, verifiable RL reward signal.

**Target**: Gold medals across tabular, computer vision, NLP, time series, and multimodal competitions.

---

## Why Podium Is Different

Every existing Kaggle AI agent takes the same approach: wrap GPT-4 or o1 in scaffolding and hope the general-purpose knowledge transfers. AIDE gets bronze/silver on ~17% of competitions. That's the ceiling for scaffolded general models.

Podium's approach: train a specialist. The same insight that separates a Kaggle Grandmaster from a smart engineer isn't access to better tools — it's internalized pattern recognition from thousands of competitions. We train that pattern recognition into weights.

```
Scaffold GPT-4 + tree search  →  ~17% medal rate  (AIDE, 2025)
Trained Podium specialist      →  target 60%+ medal rate
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PODIUM SYSTEM                               │
│                                                                     │
│  Competition Brief ──► Competition Router ──► Specialist Agent      │
│         │                     │                      │              │
│         │               [Type Detection]       [Tabular/CV/         │
│         │                                       NLP/TS/Multi]       │
│         ▼                     ▼                      ▼              │
│  EDA Agent ────────► Feature Agent ────────► Model Agent            │
│         │                     │                      │              │
│         └─────────────────────┴──────────────────────┘              │
│                               │                                     │
│                        Ensemble Agent                               │
│                               │                                     │
│                      [Submission Generator]                         │
│                               │                                     │
│                     CV Score Validator ◄── RL Reward Signal         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/calebnewtonusc/podium
cd podium
pip install -r requirements.txt
cp .env.example .env  # Fill in API keys

# Verify environment
bash scripts/check_env.sh

# Run full pipeline (data → training → eval), ~36 hours on 18× A6000
bash scripts/run_all.sh

# Or step by step:
python pipeline.py --stage discovery    # ~12 hours, collect training data
python pipeline.py --stage synthesis    # ~16 hours, generate training pairs
python pipeline.py --stage train        # ~12 hours, 3-stage training (SFT 6h + GRPO 4h + DPO 2h)
python pipeline.py --stage eval         # ~2 hours, PodiumBench evaluation
```

---

## Run on a Competition

```bash
# Point Podium at any Kaggle competition URL
python agents/competition_runner.py \
  --competition "https://www.kaggle.com/c/competition-name" \
  --time_limit 168  # hours (1 week)

# Or use the API
curl -X POST http://localhost:8000/compete \
  -H "Content-Type: application/json" \
  -d '{"competition_url": "...", "time_limit_hours": 168}'
```

---

## Performance Targets (v1)

| Metric | Target | AIDE (baseline) |
|--------|--------|-----------------|
| Medal rate (any medal) | >60% | ~17% |
| Gold medal rate | >15% | <1% |
| Tabular competitions | >75% medal | ~25% |
| CV competitions | >50% medal | ~12% |
| NLP competitions | >55% medal | ~15% |
| Time series | >70% medal | ~20% |
| Avg leaderboard percentile | Top 15% | ~49% |

---

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — Full technical architecture, 7 differentiators, training pipeline
- [DATA_SOURCES.md](DATA_SOURCES.md) — 5 training data streams with sources
- [DIFFERENTIATORS.md](DIFFERENTIATORS.md) — Why Podium beats every existing system
- [MODEL_CARD.md](MODEL_CARD.md) — Model specification, capabilities, limitations
- [ROADMAP.md](ROADMAP.md) — v1 through v3 feature roadmap
- [SETUP_GPU.md](SETUP_GPU.md) — 18× A6000 cluster configuration

---

## Competition Type Coverage

| Type | Examples | Specialist Mode |
|------|---------|-----------------|
| Tabular | House prices, fraud detection, churn | XGBoost/LightGBM ensembles, feature eng |
| Computer Vision | Image classification, segmentation, OCR | CNN/ViT architectures, augmentation |
| NLP | Sentiment, classification, generation | Transformer fine-tuning, prompt strategies |
| Time Series | Forecasting, anomaly detection | LSTM, temporal ensembles, lag features |
| Multimodal | Image+text, tabular+image | Cross-modal fusion, late/early ensembling |

---

## Hardware Requirements

- **Training**: 18× NVIDIA A6000 (48GB each) = 864GB total VRAM
- **Synthesis**: 4× A6000 per synthesis server (Azure burst for scale)
- **Inference**: 1× A6000 per active competition (parallel competitions supported)
- **Azure Credits**: 25k credits for burst synthesis and parallel evaluation

---

## License

MIT License — open training pipeline, open weights (post v1 release).
