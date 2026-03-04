"""
Data validation and quality filtering pipeline.
Runs deduplication (MinHash), quality scoring, and format validation
across all synthesized training pairs before training.

Extended validation includes:
- Python code syntax checking via AST
- Leaderboard rank scoring (gold > silver > bronze > no medal)
- Low-quality post filtering ("I didn't win but here's what I tried")
"""

import ast
import json
import re
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from loguru import logger
from tqdm import tqdm


# ─── Low quality detection patterns ──────────────────────────────────────────
_NEGATIVE_SIGNALS = [
    re.compile(r"i\s+didn't\s+win\s+but", re.I),
    re.compile(r"didn't\s+make\s+it\s+to\s+the\s+top", re.I),
    re.compile(r"no\s+medal\s+but\s+here", re.I),
    re.compile(r"unfortunately\s+i\s+didn't", re.I),
    re.compile(r"not\s+a\s+winning\s+solution", re.I),
    re.compile(r"just\s+sharing\s+what\s+i\s+tried", re.I),
]

_POSITIVE_SIGNALS = [
    re.compile(r"\b(1st|first)\s+place", re.I),
    re.compile(r"gold\s+medal", re.I),
    re.compile(r"silver\s+medal", re.I),
    re.compile(r"winning\s+solution", re.I),
    re.compile(r"top\s+[123]\b", re.I),
    re.compile(r"public\s+lb[:\s]+[0-9.]+", re.I),
    re.compile(r"private\s+lb[:\s]+[0-9.]+", re.I),
]


def check_python_syntax(code: str) -> tuple[bool, str]:
    """Check if a Python code string is syntactically valid using AST."""
    if not code or len(code) < 10:
        return False, "Code too short"
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def score_by_leaderboard_rank(record: dict) -> float:
    """
    Score 0-1 based on proven leaderboard rank.
    Gold medal = 1.0, unproven = 0.1.
    """
    rank_info = record.get("rank_info", {})
    if isinstance(rank_info, dict):
        medal = rank_info.get("medal", "")
        if medal == "gold":
            return 1.0
        if medal == "silver":
            return 0.85
        if medal == "bronze":
            return 0.70
        rank = rank_info.get("rank")
        if rank:
            if rank == 1:
                return 1.0
            elif rank <= 3:
                return 0.90
            elif rank <= 10:
                return 0.75
            elif rank <= 50:
                return 0.55
            else:
                return 0.30

    # Scan text for positive/negative signals
    text = " ".join(
        [
            record.get("description", ""),
            record.get("answer", record.get("reasoning", "")),
            (record.get("readme") or "")[:2000],
        ]
    )
    pos = sum(1 for p in _POSITIVE_SIGNALS if p.search(text))
    neg = sum(1 for n in _NEGATIVE_SIGNALS if n.search(text))

    if neg > 0 and pos == 0:
        return 0.05  # low-quality "I tried" post
    if pos > 0:
        return min(0.3 + pos * 0.1, 0.7)
    return 0.1


def is_low_quality_post(record: dict) -> bool:
    """Return True if record appears to be a 'I didn't win but here's what I tried' post."""
    text = " ".join(
        [
            record.get("description", ""),
            record.get("answer", ""),
            (record.get("readme") or "")[:2000],
        ]
    )
    neg = sum(1 for n in _NEGATIVE_SIGNALS if n.search(text))
    pos = sum(1 for p in _POSITIVE_SIGNALS if p.search(text))
    return neg > 0 and pos == 0


QUALITY_THRESHOLDS = {
    "min_code_length": 100,  # Minimum chars in code field
    "min_reasoning_length": 50,  # Minimum chars in reasoning/explanation
    "min_total_length": 300,  # Minimum total pair length
    "dedup_threshold": 0.85,  # MinHash similarity threshold for dedup
    "dedup_num_perm": 128,
}


def compute_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Compute MinHash for a text string."""
    m = MinHash(num_perm=num_perm)
    for word in text.lower().split():
        m.update(word.encode("utf8"))
    return m


def quality_score(pair: dict) -> float:
    """
    Score a training pair 0-1 based on quality signals.
    Incorporates leaderboard rank, code syntax, and low-quality detection.
    """
    score = 0.0

    code = (
        pair.get("solution_code", pair.get("code_example", pair.get("code", ""))) or ""
    )
    reasoning = (
        pair.get(
            "key_insight", pair.get("expert_explanation", pair.get("reasoning", ""))
        )
        or ""
    )
    total_text = json.dumps(pair)

    # Length checks
    if len(code) >= QUALITY_THRESHOLDS["min_code_length"]:
        score += 0.2
    if len(reasoning) >= QUALITY_THRESHOLDS["min_reasoning_length"]:
        score += 0.2
    if len(total_text) >= QUALITY_THRESHOLDS["min_total_length"]:
        score += 0.1

    # Code quality signals
    if "import" in code and (
        "sklearn" in code or "lightgbm" in code or "torch" in code
    ):
        score += 0.1
    if "def " in code or "class " in code:
        score += 0.1
    if "cv_score" in code or "cross_val" in code or "KFold" in code:
        score += 0.1

    # Leaderboard rank bonus (up to +0.2)
    lb_score = score_by_leaderboard_rank(pair)
    score += lb_score * 0.2

    # Code syntax validation (if code is present, check it)
    if code and len(code) >= QUALITY_THRESHOLDS["min_code_length"]:
        is_valid, _ = check_python_syntax(code)
        if is_valid:
            score += 0.05
        else:
            score -= 0.1  # penalize broken code

    # Low quality post penalty
    if is_low_quality_post(pair):
        score *= 0.2  # heavy penalty

    return min(max(score, 0.0), 1.0)


def validate_and_deduplicate(
    input_paths: list[Path],
    output_path: Path,
    min_quality: float = 0.5,
) -> dict:
    """
    Load all synthesized pairs, filter by quality, deduplicate, save.
    Returns stats dict.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all pairs
    all_pairs = []
    for path in input_paths:
        if not path.exists():
            logger.warning(f"Input not found: {path}")
            continue
        with open(path) as f:
            pairs = [json.loads(line) for line in f if line.strip()]
        logger.info(f"Loaded {len(pairs)} pairs from {path.name}")
        all_pairs.extend(pairs)

    logger.info(f"Total raw pairs: {len(all_pairs)}")
    original_count = len(all_pairs)

    # Quality filter
    scored = [(pair, quality_score(pair)) for pair in all_pairs]
    quality_filtered = [pair for pair, score in scored if score >= min_quality]
    logger.info(
        f"After quality filter ({min_quality:.1f}): {len(quality_filtered)} pairs"
    )

    # MinHash deduplication
    lsh = MinHashLSH(
        threshold=QUALITY_THRESHOLDS["dedup_threshold"],
        num_perm=QUALITY_THRESHOLDS["dedup_num_perm"],
    )
    unique_pairs = []
    for i, pair in enumerate(tqdm(quality_filtered, desc="Deduplicating")):
        text = json.dumps(pair)[:500]  # Hash first 500 chars
        m = compute_minhash(text, QUALITY_THRESHOLDS["dedup_num_perm"])
        key = f"pair_{i}"
        try:
            neighbors = lsh.query(m)
            if not neighbors:
                lsh.insert(key, m)
                unique_pairs.append(pair)
        except Exception as e:
            # LSH internal error: log and skip rather than silently include potential duplicate
            logger.warning(f"MinHash LSH error for pair_{i}: {e}, skipping")

    logger.info(f"After deduplication: {len(unique_pairs)} pairs")

    # Save
    with open(output_path, "w") as f:
        for pair in unique_pairs:
            f.write(json.dumps(pair) + "\n")

    stats = {
        "original": original_count,
        "after_quality_filter": len(quality_filtered),
        "after_dedup": len(unique_pairs),
        "retention_rate": len(unique_pairs) / max(original_count, 1),
        "output_path": str(output_path),
    }
    logger.info(f"Validation complete: {stats}")
    return stats


if __name__ == "__main__":
    import typer

    def main(
        data_dir: str = "./data/synthesized",
        output: str = "./data/train/all_pairs_clean.jsonl",
        min_quality: float = 0.5,
    ):
        data_path = Path(data_dir)
        input_paths = list(data_path.glob("*.jsonl"))
        validate_and_deduplicate(input_paths, Path(output), min_quality)

    typer.run(main)
