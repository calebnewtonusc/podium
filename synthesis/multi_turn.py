"""
Multi-turn competition dialogue synthesis.
Generates realistic back-and-forth between a user and Podium across a full competition lifecycle.
Teaches Podium competition strategy over time (week 1 EDA → week 2 features → week 4 ensemble).
"""

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

import aiohttp
from loguru import logger


LIFECYCLE_STAGES = [
    {
        "stage": "initial_analysis",
        "hours_elapsed": 0,
        "prompt": "I just joined this competition. Here's the overview: {competition_brief}. Where do I start?",
    },
    {
        "stage": "eda_findings",
        "hours_elapsed": 8,
        "prompt": "I've done initial EDA. Here's what I found: {eda_findings}. What features should I engineer first?",
    },
    {
        "stage": "baseline_results",
        "hours_elapsed": 24,
        "prompt": "My baseline {model} gets {baseline_score} on CV. The public LB shows {lb_score}. Is the CV/LB gap a problem?",
    },
    {
        "stage": "feature_iteration",
        "hours_elapsed": 72,
        "prompt": "I added {features_added}. CV went from {old_score} to {new_score}. What should I try next?",
    },
    {
        "stage": "ensemble_planning",
        "hours_elapsed": 120,
        "prompt": "I have {n_models} models with CV scores {scores}. How should I ensemble for the final submission?",
    },
    {
        "stage": "final_submission",
        "hours_elapsed": 166,
        "prompt": "Final 24 hours. My best single model CV is {single_cv}. Ensemble CV is {ensemble_cv}. Public LB is {lb_score}. Which submission should I go with?",
    },
]


SYSTEM_PROMPT = """\
You are Podium, an expert Kaggle competition assistant with grandmaster-level knowledge.
You have internalized patterns from thousands of competitions and can guide competitors through
the entire lifecycle of a Kaggle competition.

When responding:
1. Give specific, actionable advice — not generic ML tips
2. Explain the *why* behind every recommendation
3. Reference relevant patterns from similar past competitions when applicable
4. Flag common traps (data leakage, CV/LB correlation issues, overfitting to public LB)
5. Adapt your advice to the competition's specific metric and domain
"""


async def synthesize_competition_dialogue(
    session: aiohttp.ClientSession,
    vllm_url: str,
    api_key: str,
    competition_context: dict,
) -> list[dict] | None:
    """Generate a full multi-turn dialogue for one competition lifecycle."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    dialogue = []

    # Use a defaultdict so missing keys in competition_context render as
    # "{key}" placeholders rather than raising KeyError.
    safe_ctx = defaultdict(lambda: "[unknown]", competition_context)

    for stage in LIFECYCLE_STAGES:
        user_msg = stage["prompt"].format_map(safe_ctx)
        messages.append({"role": "user", "content": user_msg})

        payload = {
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
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
                assistant_msg = data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.debug(f"Dialogue synthesis failed at stage {stage['stage']}: {e}")
            return None

        messages.append({"role": "assistant", "content": assistant_msg})
        dialogue.append({
            "stage": stage["stage"],
            "hours_elapsed": stage["hours_elapsed"],
            "user": user_msg,
            "assistant": assistant_msg,
        })

    return {
        "competition": competition_context.get("competition_name", "unknown"),
        "competition_type": competition_context.get("competition_type"),
        "dialogue": dialogue,
        "full_messages": messages,  # Full conversation for training
    }


async def synthesize_all_dialogues(
    competitions_path: Path,
    output_path: Path,
    vllm_url: str,
    api_key: str,
    concurrency: int = 8,
) -> None:
    """Generate multi-turn dialogues for all competitions in the dataset."""
    with open(competitions_path) as f:
        competitions = [json.loads(line) for line in f]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(session, comp):
        async with semaphore:
            return await synthesize_competition_dialogue(session, vllm_url, api_key, comp)

    async with aiohttp.ClientSession() as session:
        tasks = [bounded(session, comp) for comp in competitions]
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                results.append(result)

    logger.info(f"Generated {len(results)} competition dialogues")
    with open(output_path, "w") as f:
        for dialogue in results:
            f.write(json.dumps(dialogue) + "\n")


if __name__ == "__main__":
    import typer

    def main(
        competitions: str = "./data/synthesized/competition_contexts.jsonl",
        output: str = "./data/synthesized/multi_turn_dialogues.jsonl",
        concurrency: int = 8,
    ):
        url = os.environ.get("VLLM_SYNTHESIS_URL")
        key = os.environ.get("VLLM_API_KEY")
        if not url:
            raise ValueError("VLLM_SYNTHESIS_URL not set. Export it: export VLLM_SYNTHESIS_URL=http://...")
        if not key:
            raise ValueError("VLLM_API_KEY not set. Export it: export VLLM_API_KEY=your-key")
        asyncio.run(synthesize_all_dialogues(
            Path(competitions), Path(output), url, key, concurrency
        ))

    typer.run(main)
