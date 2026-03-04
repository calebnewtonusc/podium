# Podium Roadmap

## v1 — COMPETE (Q3 2026)

Core competition agent. Trained specialist model + inference-time tree search + CV-verified RL. First open-weights Kaggle competition model.

**Goals**:
- Top 5% medal rate across PodiumBench (75 competitions)
- Gold medal rate >10%
- Works offline (no internet required)
- Open weights released on HuggingFace

**Features**:
- 3-stage trained model (SFT → CV-RL → DPO)
- AIDE-style tree search at inference time with trained specialist as brain
- 5 competition type specialist modes (tabular, CV, NLP, TS, multimodal)
- Docker execution harness for CV score validation
- Full EDA → feature engineering → modeling → ensembling pipeline
- Competition brief parser (Kaggle URL → UCD)
- PodiumBench evaluation suite

**Paper Target**: NeurIPS 2026 — "Podium: Training Specialist ML Competition Agents with Verifiable Execution Rewards"

---

## v1.5 — REMEMBER (Q4 2026)

Persistent competition memory. Every competition makes Podium smarter.

**Goals**:
- Measurable performance improvement with each competition completed
- Cross-competition transfer demonstrated (paper ablation)
- Public leaderboard tracking (live performance data)

**Features**:
- Vector memory store (ChromaDB) for competition patterns
- Competition fingerprinting and similarity matching
- Strategy transfer module ("this looks like X competition — here's what worked")
- Continual fine-tuning pipeline (weekly model updates from new competition data)
- Competition history dashboard
- Public Podium Leaderboard (community-submitted results)

---

## v2 — ORCHESTRATE (Q1 2027)

Multi-agent parallel exploration. Dozens of agents run simultaneously, results fused.

**Goals**:
- Top 1% medal rate (gold-competitive on most competitions)
- Real-time collaboration with human competitors ("Podium co-pilot" mode)
- Enterprise deployment package

**Features**:
- Parallel agent swarm (N agents explore different approaches simultaneously)
- Automated diversity enforcement (swarm explores gradient boosting + neural + feature variants in parallel)
- Learned ensemble fusion (meta-model selects and weights agent outputs)
- Human-in-the-loop mode (Podium suggests, human approves/modifies)
- Private competition support (enterprise API, on-premise deployment)
- Azure/cloud burst compute integration
- Competition strategy chat interface

---

## v3 — GENERALIZE (2027)

Beyond Kaggle. Real-world ML engineering with competition-honed skills.

**Goals**:
- Production ML pipelines from problem description
- Enterprise ML engineering assistant
- Integration with MLflow, Weights & Biases, Vertex AI

**Features**:
- Adapts competition skills to production constraints (inference latency, memory, monitoring)
- Handles messy real-world data (schema drift, missing infrastructure, privacy requirements)
- End-to-end MLOps pipeline generation
- Model documentation and reporting automation
- Fine-tuning API (organizations adapt Podium to their domain)

---

## Research Paper Pipeline

| Paper | Target Venue | Core Contribution |
|-------|-------------|-------------------|
| Podium v1 | NeurIPS 2026 | CV-RL training signal + specialist fine-tuning |
| Competition Memory | ICML 2027 | Cross-competition meta-learning |
| Specialist vs. Scaffold | ICLR 2027 | Ablation: trained specialist beats scaffolded frontier |
| PodiumBench | NeurIPS 2027 | Comprehensive ML competition agent benchmark |
