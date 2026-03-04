# Contributing to Podium

## What We Need Most

### 1. Competition Data
- Winning solution writeups from Kaggle competitions you've participated in
- High-quality notebooks with detailed reasoning (not just code)
- Competition-specific domain knowledge (medical imaging, financial time series, etc.)

### 2. Technique Validation
- Run PodiumBench and report results
- Ablation studies (SFT only vs. +RL vs. +DPO)
- Competition-type breakdowns

### 3. Specialist Mode Extensions
- Domain-specific feature engineering for new competition types
- Architecture recipes for niche domains (satellite imagery, audio, genomics)

### 4. Execution Harness
- Additional metric implementations for PodiumBench
- Competition environment reproducibility improvements

## Development Setup

```bash
git clone https://github.com/calebnewtonusc/podium
cd podium
pip install -r requirements.txt
cp .env.example .env
bash scripts/check_env.sh
```

## Pull Request Guidelines

- All new techniques should have ablation results (before/after CV score)
- Code must be executable (no pseudocode in core modules)
- Add competition examples to any new technique documentation
- Update DATA_SOURCES.md if adding new data streams

## Reporting Results

Run PodiumBench and open a GitHub issue with:
- Hardware used
- Model checkpoint (SFT / SFT+RL / SFT+RL+DPO)
- Medal rate breakdown by competition type
- Any interesting failure cases
