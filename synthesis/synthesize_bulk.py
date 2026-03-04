"""
Bulk training pair synthesis from raw Kaggle notebooks.
Uses Qwen2.5-72B via vLLM to extract structured (problem, reasoning, code, score) pairs.
"""

import asyncio
import json
import os
from pathlib import Path

import aiohttp
from loguru import logger


SYNTHESIS_PROMPT = """\
You are analyzing a Kaggle competition notebook to extract a structured training pair.

Competition: {competition}
Evaluation metric: {metric}

Notebook content:
{notebook_content}

Extract the following as a JSON object:
{{
  "competition_type": "tabular_classification|tabular_regression|cv|nlp|time_series|multimodal",
  "problem_summary": "2-3 sentence description of what this competition requires",
  "evaluation_metric": "exact metric name",
  "key_eda_insights": ["insight 1", "insight 2", ...],
  "feature_engineering": {{
    "features_created": ["feature 1", "feature 2", ...],
    "reasoning": "why these features were chosen"
  }},
  "model_architecture": {{
    "models_used": ["model 1", ...],
    "reasoning": "why these models"
  }},
  "cv_strategy": "how cross-validation was structured",
  "cv_score": <float or null>,
  "key_insight": "the single most impactful decision that improved score",
  "solution_code": "the most important code block (feature engineering or modeling)"
}}

Return ONLY valid JSON. If information is not present, use null.
"""


async def synthesize_notebook(
    session: aiohttp.ClientSession,
    vllm_url: str,
    api_key: str,
    notebook_path: Path,
    competition_meta: dict,
) -> dict | None:
    """Send one notebook to vLLM for synthesis. Returns structured pair or None."""
    try:
        content = notebook_path.read_text(errors="replace")[
            :12000
        ]  # Trim to context limit
    except Exception:
        return None

    prompt = SYNTHESIS_PROMPT.format(
        competition=competition_meta.get("competition", "unknown"),
        metric=competition_meta.get("metric", "unknown"),
        notebook_content=content,
    )

    payload = {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
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
            text = data["choices"][0]["message"]["content"]
            parsed = json.loads(text)
            parsed["_source_notebook"] = notebook_path.name
            parsed["_competition"] = competition_meta.get("competition")
            return parsed
    except Exception as e:
        logger.debug(f"Synthesis failed for {notebook_path.name}: {e}")
        return None


async def synthesize_all(
    notebook_dir: Path,
    index_path: Path,
    output_path: Path,
    vllm_url: str,
    api_key: str,
    concurrency: int = 32,
) -> None:
    """Synthesize all notebooks in parallel with bounded concurrency."""
    with open(index_path) as f:
        # Avoid walrus operator: empty-dict records ({}) are falsy and would be silently
        # dropped by `if (e := json.loads(line))`. Parse unconditionally instead.
        index = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if "kernel_ref" in e:
                index[e["kernel_ref"].replace("/", "__")] = e

    notebook_files = list(notebook_dir.glob("*.ipynb"))
    logger.info(f"Synthesizing {len(notebook_files)} notebooks → {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_synthesize(session, nb_path):
        async with semaphore:
            key = nb_path.stem
            meta = index.get(key, {})
            return await synthesize_notebook(session, vllm_url, api_key, nb_path, meta)

    async with aiohttp.ClientSession() as session:
        tasks = [bounded_synthesize(session, nb) for nb in notebook_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    valid = [r for r in results if isinstance(r, dict)]
    logger.info(f"Synthesis complete: {len(valid)}/{len(notebook_files)} succeeded")

    with open(output_path, "w") as f:
        for pair in valid:
            f.write(json.dumps(pair) + "\n")


if __name__ == "__main__":
    import typer

    def main(
        notebook_dir: str = "./data/raw/notebooks/files",
        index: str = "./data/raw/notebooks/notebook_index.jsonl",
        output: str = "./data/synthesized/notebook_pairs.jsonl",
        vllm_url: str = None,
        concurrency: int = 32,
    ):
        url = vllm_url or os.environ.get("VLLM_SYNTHESIS_URL")
        key = os.environ.get("VLLM_API_KEY")
        if not url:
            raise ValueError(
                "VLLM_SYNTHESIS_URL not set. Export it: export VLLM_SYNTHESIS_URL=http://..."
            )
        if not key:
            raise ValueError(
                "VLLM_API_KEY not set. Export it: export VLLM_API_KEY=your-key"
            )
        asyncio.run(
            synthesize_all(
                Path(notebook_dir), Path(index), Path(output), url, key, concurrency
            )
        )

    typer.run(main)
