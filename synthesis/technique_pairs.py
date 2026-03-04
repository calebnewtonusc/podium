"""
Technique synthesis — generates (technique, when to apply, code example) training pairs
from ML papers, grandmaster guides, and domain references.
"""

import asyncio
import json
import os
from pathlib import Path

import aiohttp
from loguru import logger


TECHNIQUES = [
    # Tabular
    {
        "name": "Smoothed Target Encoding",
        "domain": "tabular",
        "source": "Daniele Micci-Barreca (2001), popularized by Kaggle grandmasters",
        "when": "High-cardinality categorical features in regression or binary classification",
        "pitfall": "Leakage if not done within CV folds; use leave-one-out or fold-based encoding",
    },
    {
        "name": "Group Statistics Features",
        "domain": "tabular",
        "source": "Standard Kaggle grandmaster technique",
        "when": "Dataset has meaningful grouping variables (user_id, category, region)",
        "pitfall": "Can cause leakage if test-time group statistics incorporate target",
    },
    {
        "name": "Hill Climbing Ensemble Selection",
        "domain": "ensembling",
        "source": "Rich Caruana et al. (2004), staple of Kaggle top solutions",
        "when": "Final ensemble construction from diverse model zoo",
        "pitfall": "Overfits to validation set if performed on full training data",
    },
    {
        "name": "Pseudo-Labeling",
        "domain": "semi-supervised",
        "source": "Lee (2013), widely used in Kaggle CV/NLP competitions",
        "when": "Large unlabeled test set, model is confident on a subset (>95% probability)",
        "pitfall": "Propagates model errors; use only high-confidence predictions",
    },
    {
        "name": "Adversarial Validation",
        "domain": "validation",
        "source": "Kaggle community technique",
        "when": "Suspecting train/test distribution shift; diagnose CV/LB gap",
        "pitfall": "Computationally expensive; use as diagnostic, not feature engineering",
    },
    {
        "name": "Log1p Target Transform",
        "domain": "regression",
        "source": "Standard statistical technique",
        "when": "Right-skewed regression targets (prices, counts, durations); RMSLE metric",
        "pitfall": "Must inverse transform predictions; introduces bias for some metrics",
    },
    {
        "name": "OOF Stacking",
        "domain": "ensembling",
        "source": "Wolpert (1992), popularized in Kaggle",
        "when": "Multiple diverse base models exist; want to learn optimal combination",
        "pitfall": "Overfits with small datasets; meta-learner must be simple (Ridge, LR)",
    },
    {
        "name": "Purged K-Fold Cross-Validation",
        "domain": "time_series",
        "source": "Marcos Lopez de Prado (2018)",
        "when": "Time series data with autocorrelation; financial or sequential data",
        "pitfall": "Standard K-fold leaks future information into past; always purge gap",
    },
    {
        "name": "Test-Time Augmentation (TTA)",
        "domain": "computer_vision",
        "source": "Standard practice in CV competitions",
        "when": "Image classification/segmentation; inference time improvement",
        "pitfall": "Increases inference time proportionally; diminishing returns after 5-10 augmentations",
    },
    {
        "name": "Mixed Precision Training",
        "domain": "deep_learning",
        "source": "NVIDIA (2018)",
        "when": "Any deep learning model on modern GPUs; reduces memory, speeds training",
        "pitfall": "Numerical instability with very small gradients; use gradient clipping",
    },
]


SYNTHESIS_PROMPT = """\
You are an expert Kaggle grandmaster writing training data for an AI competition assistant.

Generate a complete training example for this ML technique:

Technique: {name}
Domain: {domain}
Source: {source}
When to apply: {when}
Common pitfall: {pitfall}

Output a JSON object with:
{{
  "technique_name": "{name}",
  "domain": "{domain}",
  "competition_scenario": "A specific, realistic Kaggle competition scenario where this applies",
  "user_question": "The question a competitor would ask",
  "expert_explanation": "Expert explanation including: what it is, why it works, when to use it, and the pitfall",
  "code_example": "Complete, runnable Python code demonstrating this technique on realistic data",
  "expected_impact": "Typical CV score improvement (e.g., +0.002 AUC, +0.3% accuracy)",
  "similar_techniques": ["list", "of", "related", "techniques"]
}}
"""


async def synthesize_technique(
    session: aiohttp.ClientSession,
    vllm_url: str,
    api_key: str,
    technique: dict,
) -> dict | None:
    prompt = SYNTHESIS_PROMPT.format(**technique)
    payload = {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with session.post(
            f"{vllm_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return json.loads(data["choices"][0]["message"]["content"])
    except Exception as e:
        logger.debug(f"Technique synthesis failed for {technique['name']}: {e}")
        return None


async def synthesize_all_techniques(
    output_path: Path,
    vllm_url: str,
    api_key: str,
    n_variants: int = 10,
) -> None:
    """Generate N variants per technique for a diverse training set."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for technique in TECHNIQUES:
            for _ in range(n_variants):
                tasks.append(synthesize_technique(session, vllm_url, api_key, technique))
        pairs = await asyncio.gather(*tasks)
        results = [p for p in pairs if p is not None]

    logger.info(f"Synthesized {len(results)} technique pairs")
    with open(output_path, "w") as f:
        for pair in results:
            f.write(json.dumps(pair) + "\n")


if __name__ == "__main__":
    import typer

    def main(
        output: str = "./data/synthesized/technique_pairs.jsonl",
        n_variants: int = 10,
    ):
        url = os.environ.get("VLLM_SYNTHESIS_URL")
        key = os.environ.get("VLLM_API_KEY")
        if not url:
            raise ValueError("VLLM_SYNTHESIS_URL not set. Export it: export VLLM_SYNTHESIS_URL=http://...")
        if not key:
            raise ValueError("VLLM_API_KEY not set. Export it: export VLLM_API_KEY=your-key")
        asyncio.run(synthesize_all_techniques(Path(output), url, key, n_variants))

    typer.run(main)
