"""
DPO preference pair generation.
Creates (prompt, chosen, rejected) pairs encoding competition strategy wisdom.
Grandmaster decision → chosen. Naive/suboptimal decision → rejected.
"""

import asyncio
import json
import os
from pathlib import Path

import aiohttp
from loguru import logger


PREFERENCE_SCENARIOS = [
    {
        "scenario": "cv_lb_gap",
        "prompt": "My model gets 0.87 AUC on 5-fold CV but only 0.82 on the public leaderboard. It's week 2 of the competition. What should I do?",
        "chosen": """This is a classic CV/LB gap problem. Most likely causes:

1. **Public LB is small** — public test is only ~30% of test data, so it's noisy. Don't chase it.
2. **Data leakage in your CV** — your CV folds may be contaminated (e.g., shuffling time-series data, group leakage).
3. **Overfitting to CV** — your feature engineering was too tuned to validation.

**What to do:**
- First, run adversarial validation: train a classifier to distinguish train vs. test. If it achieves >0.6 AUC, there's distribution shift.
- Check if your data has temporal structure — if so, switch to time-based CV split.
- Add more regularization (reduce n_estimators, increase min_child_weight).
- **Do NOT optimize for public LB** — it's noisy. Trust your local CV if it's properly structured.

The gap should close in the final evaluation if your CV is sound.""",
        "rejected": """Your public LB score is lower than CV, which means your model is overfitting. You should:
- Try different hyperparameters to improve the LB score
- Submit more variations to figure out what works on the public LB
- The LB is more reliable than your local CV since it uses real test data
- Focus on maximizing the public LB score for the rest of the competition""",
    },
    {
        "scenario": "ensemble_timing",
        "prompt": "I have 3 days left in the competition. My best single model CV is 0.9234. Should I keep engineering features or start building my ensemble?",
        "chosen": """With 3 days left, **start your ensemble now** — don't wait.

Here's the time allocation I'd recommend:
- Day 1: Train your full model zoo (LightGBM, XGBoost, CatBoost, at minimum).
  Each model needs full 5-fold CV to generate proper OOF predictions.
- Day 2: Build the ensemble. Start with hill climbing on OOF predictions, then try OOF stacking with Ridge meta-learner.
- Day 3: Pseudo-labeling (if test set is large), final submission selection.

**Why stop feature engineering now:**
- Feature engineering has diminishing returns after you've covered the obvious ones.
- A well-tuned ensemble of 4-5 models typically adds +0.003 to +0.01 over the best single model.
- You need the OOF predictions from multiple models to build the ensemble — those take time to train.

**The submission selection trap:** Pick 2 final submissions — your best CV ensemble AND your most conservative single model. Don't risk everything on the ensemble if CV/LB correlation is unstable.""",
        "rejected": """3 days is plenty of time. Keep engineering features until the last moment — more features = better score. Once you have the best possible single model, build the ensemble in the final 24 hours. Feature engineering is more impactful than ensembling anyway.""",
    },
    {
        "scenario": "submission_selection",
        "prompt": "Final day. I have two submissions: (A) ensemble CV=0.9412, public LB=0.9287 and (B) single model CV=0.9380, public LB=0.9341. Which do I select?",
        "chosen": """This is one of the most important decisions in any competition, and it's a trap many people fall into.

**My recommendation: Select submission A (the ensemble).**

Here's the reasoning:
- The public LB only uses ~30% of the test data. It's inherently noisy, especially in smaller competitions.
- Your ensemble has **higher CV** (0.9412 > 0.9380), which is based on all your training data with proper validation.
- The fact that B scores higher on the public LB doesn't mean it's actually better — it might just be lucky on that 30% sample.
- Historically, chasing the public LB in final selections is one of the most common causes of dropping rank on the private LB ("shakeup").

**When to trust the public LB over CV:**
- If the LB uses >50% of test data
- If you suspect your CV has data leakage
- If the gap is enormous (>0.01) and consistent across all your submissions

A 0.0054 LB difference with a better ensemble CV is noise — go with A.""",
        "rejected": """Select submission B — it has a higher public leaderboard score (0.9341 > 0.9287), and the public LB uses real test data so it's more reliable than your local CV. You want to maximize your final score, so always go with what performs better on the public leaderboard.""",
    },
    {
        "scenario": "feature_importance",
        "prompt": "My LightGBM feature importance shows 3 features account for 80% of splits. Should I drop the other features?",
        "chosen": """Don't drop them yet — this requires careful analysis, not a quick prune.

**What high-importance concentration means:**
- Your top 3 features are doing most of the heavy lifting — they're genuinely predictive.
- The other features may still be adding marginal value that compounds in an ensemble.

**How to properly evaluate:**
1. Run ablation: train with top-3 only vs. all features. Compare CV scores.
2. Check if low-importance features improve *specific* fold predictions (some features matter in edge cases).
3. LightGBM's default importance can be misleading — use `importance_type='gain'` not `'split'`.

**If you do prune:**
- Remove features with near-zero gain importance (< 0.001% of total gain)
- Prune iteratively — never remove 50+ features at once
- Keep any features that are engineered from domain knowledge, even if LGB doesn't love them yet

**The real benefit of pruning:** Faster training and less overfitting, not necessarily higher CV. If CV doesn't improve after pruning, keep the full feature set — the ensemble will benefit from feature diversity.""",
        "rejected": """Yes, drop the low-importance features. They're adding noise and causing overfitting. Keep only the top features that LightGBM finds important. Fewer, better features almost always beats more features.""",
    },
]


def generate_preference_pair(
    scenario: dict,
) -> dict:
    """Format a scenario as a DPO training pair."""
    return {
        "prompt": f"<|im_start|>user\n{scenario['prompt']}<|im_end|>\n<|im_start|>assistant\n",
        "chosen": scenario["chosen"] + "<|im_end|>",
        "rejected": scenario["rejected"] + "<|im_end|>",
        "scenario_type": scenario["scenario"],
    }


async def generate_additional_pairs(
    session: aiohttp.ClientSession,
    vllm_url: str,
    api_key: str,
    n_extra: int = 100,
) -> list[dict]:
    """Use LLM to generate additional preference pairs beyond the hand-crafted ones."""
    meta_prompt = """Generate a Kaggle competition strategy preference pair.

Output JSON with:
{
  "scenario": "one_word_scenario_name",
  "competition_context": "Brief description of competition and current state",
  "question": "The question a competitor would ask",
  "chosen": "Expert grandmaster response (correct, nuanced, actionable)",
  "rejected": "Plausible but subtly wrong response (a common mistake)",
  "mistake_type": "What error the rejected response makes"
}

The rejected response should be *plausible* — something a smart person might reasonably say, but that a grandmaster would know is wrong or suboptimal. Examples of mistake types: chasing public LB, premature ensembling, over-pruning features, ignoring CV/LB correlation, not checking data leakage."""

    semaphore = asyncio.Semaphore(8)

    async def generate_one():
        async with semaphore:
            payload = {
                "model": "Qwen/Qwen2.5-72B-Instruct",
                "messages": [{"role": "user", "content": meta_prompt}],
                "temperature": 0.8,
                "max_tokens": 1500,
                "response_format": {"type": "json_object"},
            }
            try:
                async with session.post(
                    f"{vllm_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    result = json.loads(data["choices"][0]["message"]["content"])
                    question = result.get("question")
                    chosen = result.get("chosen")
                    rejected = result.get("rejected")
                    if question is None or chosen is None or rejected is None:
                        logger.debug(
                            f"DPO pair missing fields — question={question is None}, "
                            f"chosen={chosen is None}, rejected={rejected is None}; skipping"
                        )
                        return None
                    return {
                        "prompt": f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n",
                        "chosen": chosen + "<|im_end|>",
                        "rejected": rejected + "<|im_end|>",
                        "scenario_type": result.get("scenario", "generated"),
                    }
            except Exception:
                return None

    tasks = [generate_one() for _ in range(n_extra)]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def build_dpo_dataset(
    output_path: Path,
    vllm_url: str,
    api_key: str,
    n_generated: int = 1000,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = []

    # Start with hand-crafted high-quality pairs
    for scenario in PREFERENCE_SCENARIOS:
        pair = generate_preference_pair(scenario)
        pairs.append(pair)

    async with aiohttp.ClientSession() as session:
        # Generate additional pairs via LLM
        logger.info(f"Generating {n_generated} additional DPO pairs...")
        extra = await generate_additional_pairs(session, vllm_url, api_key, n_generated)
        pairs.extend(extra)

    logger.info(f"DPO dataset: {len(pairs)} pairs")
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")


if __name__ == "__main__":
    import typer

    def main(
        output: str = "./data/dpo/competition_preferences.jsonl",
        n_generated: int = 1000,
    ):
        url = os.environ.get("VLLM_SYNTHESIS_URL")
        key = os.environ.get("VLLM_API_KEY")
        if not url:
            raise ValueError(
                "VLLM_SYNTHESIS_URL not set. Export it: export VLLM_SYNTHESIS_URL=http://..."
            )
        if not key:
            raise ValueError(
                "VLLM_API_KEY not set. Export it: export VLLM_API_KEY=your-key"
            )
        asyncio.run(build_dpo_dataset(Path(output), url, key, n_generated))

    typer.run(main)
