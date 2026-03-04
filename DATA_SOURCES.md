# Podium Training Data Sources

Target dataset size: **~800k training pairs** from 5 streams.

All data collection respects Kaggle's Terms of Service for public notebook access, and uses only publicly available competition materials, discussions, and solution writeups.

---

## Stream 1: Kaggle Public Notebooks (35% — ~280k pairs)

The largest and most code-dense training stream. Kaggle hosts 2M+ public notebooks spanning every competition and dataset. We filter for quality (>50 votes, executed, has CV score reported) and synthesize (competition context, approach reasoning, code, achieved score) pairs.

### Sources

| Source | Volume | Notes |
|--------|--------|-------|
| Kaggle public notebooks API | ~2M notebooks total | Filter: votes >50, has output, is competition notebook |
| Filtered high-quality | ~300k notebooks | After quality filter |
| Synthesis pairs generated | ~280k pairs | 1 pair per notebook avg |

### Collection Method
```
kaggle_notebooks.py → Kaggle API (notebooks list by competition)
fetch_bulk.py → Download notebook JSON + outputs (30 parallel workers)
synthesize_bulk.py → LLM extracts (problem, approach, code, score) from each notebook
```

### Quality Filters
- Minimum 50 upvotes (ensures community validation)
- Must have executed outputs (not just code cells)
- Must belong to a competition (not dataset/tutorial notebooks)
- CV score or LB score mentioned in outputs or markdown
- Deduplicated via MinHash (removes near-duplicate "copy starter code" notebooks)

### Synthesis Prompt Pattern
```
Given this Kaggle notebook:
[NOTEBOOK CONTENT]

Extract:
1. Competition objective and evaluation metric
2. Key EDA insights discovered
3. Feature engineering decisions made (and reasoning if given)
4. Model architecture chosen (and why)
5. CV strategy used
6. Final CV score achieved
7. What the author identified as the most impactful change

Output as structured (competition_context, reasoning_chain, code_solution, cv_score) tuple.
```

---

## Stream 2: Winning Solution Writeups (25% — ~200k pairs)

The highest signal-to-noise stream. After every competition, top finishers write detailed solution explanations in Kaggle Discussions. These are goldmines: grandmasters explaining exactly what they did, why it worked, and what they tried that didn't work. This is the training data that puts expert reasoning into weights.

### Sources

| Source | Volume | Notes |
|--------|--------|-------|
| Kaggle competition discussion posts (gold/silver/bronze solutions) | ~50k writeups | Top 5% finishers across all competitions |
| [kaggle-solutions GitHub repo](https://github.com/faridrashidi/kaggle-solutions) | ~3k curated links | Community-maintained index |
| [Kaggle Solutions & Writeups dataset](https://www.kaggle.com/datasets/samvelkoch/kaggles-competitions-solutions-and-writeups) | ~5k structured | Pre-curated dataset |
| Manually indexed competition forums (2019-2025) | ~42k additional | Script-scraped with voting filter |

### Collection Method
```
solution_writeups.py → Kaggle Discussions API (filter: top-N discussions per competition)
→ Filter: "solution" OR "approach" in title, author in top-10 leaderboard
→ synthesize_bulk.py → Extract (competition summary, strategic decisions, reasoning)
```

### Synthesis Pattern
These writeups often directly contain the training signal — we format them into:
```
Competition: [name, metric, domain]
Approach:
  - Stage 1: [what they did + why]
  - Stage 2: [what they did + why]
  ...
Key Insight: [the "aha" moment that separated gold from silver]
What Didn't Work: [failed approaches + why they failed]
Final Ensemble: [exact blend / stack configuration]
```

This is then used as training target for the "strategic reasoning" component.

### Notable High-Value Competitions (Dense Grandmaster Writeups)
- Optiver Realized Volatility Prediction
- Feedback Prize (NLP series)
- BirdCLEF series (audio classification)
- Google Brain Ventilator Pressure
- Vesuvius Challenge
- ARC Prize (reasoning)
- NFL Big Data Bowl (domain expertise)
- RSNA series (medical imaging)

---

## Stream 3: Community Discussions (20% — ~160k pairs)

Competition discussion threads contain a rich Q&A corpus: "why is my CV/LB correlation terrible?", "what features helped most?", "should I use target encoding here?". These translate directly into technique tutoring pairs.

### Sources

| Source | Volume | Notes |
|--------|--------|-------|
| Kaggle competition forums (top 200 competitions) | ~500k posts | All public discussion posts |
| High-vote Q&A threads | ~200k posts | Filter: topic posts with votes >10 |
| Synthesized pairs | ~160k pairs | ~0.8 pairs per high-value post |

### Collection Method
```
discussions.py → Scrape all competition discussion posts
→ Filter: question posts with expert answers (answer author in top 25% leaderboard)
→ synthesize_bulk.py → Format as (question + competition context) → (expert answer + technique)
```

### Pair Types Generated
1. **Technique application**: "When should I use target encoding vs. OHE in this competition?"
2. **CV strategy**: "My CV shows 0.87 but public LB shows 0.83 — what's happening?"
3. **Feature debugging**: "Adding lag features dropped my score — here's my code"
4. **Architecture advice**: "Tabular competition, 50 features, 500k rows — CNN or LGBM?"
5. **Ensemble guidance**: "I have 5 models ranging 0.89-0.91 — how do I optimally blend?"

---

## Stream 4: Technique Synthesis (15% — ~120k pairs)

Expert ML knowledge from papers, books, and grandmaster guides synthesized into competition-applicable technique pairs. Bridges academic ML knowledge with practical competition execution.

### Sources

| Source | Volume | Notes |
|--------|--------|-------|
| NVIDIA Kaggle Grandmasters Playbook series | 12 articles | Core techniques |
| ArXiv ML papers applied to competition settings | ~500 papers | Feature engineering, ensembling, architectures |
| The Kaggle Book (Banachewicz & Massaron) | Full text | Competition strategy bible |
| ML Mastery guides (tabular, CV, NLP) | ~200 guides | Applied technique walkthroughs |
| AutoML papers (AutoSklearn, FLAML, TPOT) | ~50 papers | Pipeline search strategies |
| Synthesized technique pairs | ~120k pairs | LLM generates (technique, when to use, code example) |

### Synthesis Pattern
For each technique/paper/guide:
```
Source: [paper/guide/article]
Technique: [name]
When to apply: [competition conditions that signal this technique]
Expected impact: [typical score improvement, caveats]
Code template:
  [working Python implementation]
Competition examples: [historical competitions where this was key]
Common mistakes: [what people do wrong with this technique]
```

### Key Technique Categories
**Tabular**:
- Target encoding variants (smoothed, LOO, weight of evidence)
- Feature interaction generation (polynomial, group statistics)
- Gradient boosting hyperparameter strategies (LightGBM, XGBoost, CatBoost)
- Neural tabular architectures (TabNet, FT-Transformer, SAINT)
- Stacking and blending strategies with OOF predictions

**CV**:
- Progressive resizing, test-time augmentation (TTA)
- Architecture selection heuristics (EffNet vs ViT vs ConvNeXt by task)
- Pseudo-labeling and self-training strategies
- Mixup, CutMix, and augmentation intensity scheduling
- Knowledge distillation for competition inference

**NLP**:
- DeBERTa fine-tuning configurations
- Multi-sample dropout, gradient clipping
- Sentence pair vs. single sequence formulations
- Domain-adaptive pretraining (DAPT)
- Token classification vs. sequence classification tradeoffs

**Time Series**:
- Lag feature factories (automated lag/rolling generation)
- Temporal cross-validation strategies (gap, purged)
- LGBM + LSTM/N-BEATS ensemble strategies
- Seasonality and trend decomposition
- External data incorporation strategies

---

## Stream 5: Meta-Competition Pairs (5% — ~40k pairs)

High-level competition strategy: given a competition type and description, what's the optimal pipeline? These pairs teach Podium the meta-reasoning layer — how to look at an unseen competition and immediately know the most promising starting strategy.

### Sources

| Source | Volume | Notes |
|--------|--------|-------|
| MLE-bench 75 competition profiles (OpenAI) | 75 competitions | Gold-standard evaluations |
| Competition taxonomy synthesis | ~5k manually reviewed | Categorized by type/domain/difficulty |
| Grandmaster strategy interviews | ~200 interviews | Top Kaggle profiles, YouTube, blog posts |
| Cross-competition transfer analysis | Synthesized | What worked across structurally similar competitions |

### Pair Format
```
Competition Profile:
  Type: tabular_regression
  Domain: financial
  Metric: RMSPE
  Features: 100 numeric, 5 categorical, 3 time-based
  Train size: 500k rows / Test size: 100k rows
  Target distribution: right-skewed, log-normal
  Data quality: moderate missingness (15%)

Optimal Pipeline:
  Phase 1 (Hours 1-24): EDA
    - Target log-transform (always for right-skewed RMSPE)
    - Group-by analysis on categorical features
    - Temporal correlation analysis
    - Check for data leakage (common in financial competitions)

  Phase 2 (Hours 24-72): Feature Engineering
    - Generate lag features if time component
    - Group statistics (mean/std/min/max per category)
    - Ratio features between related numerics
    - Target encoding with 5-fold smoothing

  Phase 3 (Hours 72-120): Model Building
    - LightGBM baseline (fast iteration)
    - XGBoost with different tree structure
    - CatBoost for categorical handling
    - Simple neural net (TabNet or FT-Transformer)

  Phase 4 (Hours 120-168): Ensemble
    - OOF stacking with linear meta-learner
    - Optuna blend weight optimization
    - Pseudo-labeling if test is large

  Key Risk: Financial data often has distribution shift — prioritize CV/LB correlation early
```

---

## Data Quality Pipeline

```
raw_data/
    ↓ [deduplication: MinHash, 0.85 threshold]
    ↓ [quality filter: votes, execution status, score presence]
    ↓ [synthesis: Qwen2.5-72B via vLLM]
    ↓ [validation: structured output check, code syntax check]
    ↓ [CV execution: actually run generated code, verify scores]
    ↓ [DPO pairing: rank competing approaches by actual CV score]
cleaned_pairs/
```

### Synthesis Model
- **Primary**: Qwen2.5-72B-Instruct (via vLLM on 4× A6000 per server, 4 servers)
- **Fallback**: Claude 3.5 Sonnet (Anthropic API) for complex reasoning pairs
- **Throughput**: ~1,000 pairs/hour per synthesis server
- **Total synthesis time**: ~200 hours of compute (parallelized across 4 servers = ~50 hours wall time)

---

## Final Dataset Statistics (Targets)

| Stream | Pairs | Avg tokens/pair | Total tokens |
|--------|-------|-----------------|--------------|
| Kaggle Notebooks | 280k | 2,400 | 672M |
| Winning Writeups | 200k | 1,800 | 360M |
| Discussions | 160k | 800 | 128M |
| Technique Synthesis | 120k | 1,200 | 144M |
| Meta-Competition | 40k | 2,000 | 80M |
| **Total** | **800k** | **~1,730 avg** | **~1.38B** |
