"""
competition_curriculum.py - Smart curriculum ordering for ML competition training data.

Orders training examples from simpler to more complex competition types.
Ensures proper coverage ratios: tabular 40%, CV 25%, NLP 25%, time_series 5%, other 5%.
Deduplicates similar techniques using MinHash.

Complexity ordering:
    1 = Binary classification, tabular (simplest)
    2 = Multi-class classification, regression
    3 = Computer vision (image classification)
    4 = NLP (text classification, NER)
    5 = Time series forecasting
    6 = Object detection, segmentation
    7 = Multi-label, multi-target
    8 = Multimodal, generative
    9 = Reinforcement learning, custom objectives

Usage:
    python synthesis/competition_curriculum.py --input data/ --output data/curriculum/
    python synthesis/competition_curriculum.py --stats
"""

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parents[1] / "data"
CURRICULUM_DIR = DATA_DIR / "curriculum"

# ─── Coverage targets ─────────────────────────────────────────────────────────
COVERAGE_TARGETS = {
    "tabular": 0.40,
    "cv": 0.25,
    "nlp": 0.25,
    "time_series": 0.05,
    "multimodal": 0.03,
    "other": 0.02,
}

# ─── Complexity scoring by competition type and technique ─────────────────────
COMPLEXITY_BY_TYPE = {
    "tabular": {
        "binary_classification": 1,
        "multiclass_classification": 2,
        "regression": 2,
        "ranking": 3,
        "survival_analysis": 4,
        "default": 2,
    },
    "cv": {
        "image_classification": 3,
        "object_detection": 5,
        "segmentation": 6,
        "image_generation": 7,
        "default": 4,
    },
    "nlp": {
        "sentiment_analysis": 3,
        "text_classification": 3,
        "ner": 4,
        "question_answering": 5,
        "machine_translation": 6,
        "generation": 7,
        "default": 4,
    },
    "time_series": {
        "forecasting": 4,
        "classification": 3,
        "anomaly_detection": 5,
        "default": 4,
    },
    "multimodal": {"default": 7},
    "other": {"default": 5},
}

# ─── Technique-level complexity modifiers ─────────────────────────────────────
TECHNIQUE_COMPLEXITY = {
    # Simple
    "logistic_regression": 1,
    "linear_regression": 1,
    "random_forest": 2,
    "gradient_boosting": 2,
    "xgboost": 2,
    "lightgbm": 2,
    "catboost": 2,
    # Medium
    "neural_network": 3,
    "transfer_learning": 3,
    "data_augmentation": 3,
    "cross_validation": 2,
    "feature_engineering": 2,
    # Advanced
    "ensemble": 4,
    "stacking": 4,
    "pseudo_labeling": 4,
    "test_time_augmentation": 4,
    "knowledge_distillation": 5,
    # Expert
    "custom_loss": 5,
    "architecture_search": 6,
    "meta_learning": 6,
    "reinforcement_learning": 8,
    "multimodal_fusion": 7,
}


def minhash_signature(text: str, num_hashes: int = 64) -> list[int]:
    """
    Compute a MinHash signature for deduplication.
    Uses a sliding window of 3-grams for better deduplication.
    """
    # Build shingles (3-word n-grams)
    words = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).split()
    shingles = set()
    for i in range(len(words) - 2):
        shingle = " ".join(words[i:i+3])
        shingles.add(shingle)

    if not shingles:
        return [0] * num_hashes

    # MinHash computation
    signature = []
    for i in range(num_hashes):
        min_hash = float('inf')
        for shingle in shingles:
            h = int(hashlib.md5(f"{i}:{shingle}".encode()).hexdigest(), 16)
            if h < min_hash:
                min_hash = h
        signature.append(min_hash % (2**32))
    return signature


def estimate_jaccard(sig1: list[int], sig2: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not sig1 or not sig2:
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def deduplicate_records(records: list[dict], threshold: float = 0.85) -> list[dict]:
    """
    Remove near-duplicate records using MinHash similarity.
    Keeps the highest-quality version of each near-duplicate cluster.
    """
    if not records:
        return records

    # Compute signatures
    signatures = []
    for rec in records:
        text = " ".join([
            rec.get("description", ""),
            rec.get("question", ""),
            rec.get("answer", ""),
            (rec.get("readme") or "")[:1000],
        ])
        signatures.append(minhash_signature(text))

    # LSH bucketing for efficiency (instead of O(n^2))
    # Use bands of hash values for approximate nearest neighbor
    band_size = 8
    num_bands = len(signatures[0]) // band_size if signatures else 0

    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, sig in enumerate(signatures):
        for band in range(num_bands):
            band_hash = tuple(sig[band * band_size:(band + 1) * band_size])
            buckets[band_hash].append(i)

    # Find duplicate candidates from same buckets
    duplicate_groups: list[set] = []
    processed = set()

    for bucket_indices in buckets.values():
        if len(bucket_indices) < 2:
            continue
        for i in range(len(bucket_indices)):
            for j in range(i + 1, len(bucket_indices)):
                idx_i = bucket_indices[i]
                idx_j = bucket_indices[j]
                if idx_i in processed or idx_j in processed:
                    continue
                sim = estimate_jaccard(signatures[idx_i], signatures[idx_j])
                if sim >= threshold:
                    # Find if either is in an existing group
                    group_found = False
                    for group in duplicate_groups:
                        if idx_i in group or idx_j in group:
                            group.add(idx_i)
                            group.add(idx_j)
                            group_found = True
                            break
                    if not group_found:
                        duplicate_groups.append({idx_i, idx_j})
                    # Only mark idx_i as processed — marking idx_j would prevent
                    # later pairs where idx_j appears as the "i" side from being checked.
                    processed.add(idx_i)

    # Mark duplicates: keep highest quality in each group
    remove_indices = set()
    for group in duplicate_groups:
        group_list = list(group)
        # Sort by quality score descending
        group_list.sort(
            key=lambda i: records[i].get("_quality_score",
                                          records[i].get("quality_score", 0)),
            reverse=True
        )
        # Remove all but the best
        for idx in group_list[1:]:
            remove_indices.add(idx)

    deduped = [r for i, r in enumerate(records) if i not in remove_indices]
    removed = len(records) - len(deduped)
    if removed > 0:
        print(f"  Deduplication removed {removed}/{len(records)} near-duplicates")

    return deduped


def score_record_quality(record: dict) -> float:
    """Score a training record's quality."""
    score = 0.3  # base

    # Pre-existing quality score
    if "quality_score" in record:
        return record["quality_score"]

    # Leaderboard proof
    rank_info = record.get("rank_info", {})
    if rank_info.get("medal") == "gold":
        score += 0.4
    elif rank_info.get("medal") == "silver":
        score += 0.3
    elif rank_info.get("medal") == "bronze":
        score += 0.2
    elif rank_info.get("rank") and rank_info["rank"] <= 10:
        score += 0.35
    elif rank_info.get("rank") and rank_info["rank"] <= 50:
        score += 0.2
    elif rank_info.get("has_rank"):
        score += 0.1

    # Stars
    stars = record.get("stars", 0)
    score += min(stars / 200.0, 0.15)

    # Content richness
    votes = record.get("votes", 0)
    score += min(votes / 50.0, 0.1)

    return min(1.0, score)


def compute_complexity(record: dict) -> int:
    """Compute curriculum complexity level (1-9) for a record."""
    comp_type = record.get("competition_type", record.get("task", "tabular"))
    comp_type_map = COMPLEXITY_BY_TYPE.get(comp_type, COMPLEXITY_BY_TYPE["other"])
    base_complexity = comp_type_map.get("default", 3)

    # Check for technique mentions in description/content
    text = " ".join([
        record.get("description", ""),
        (record.get("readme") or "")[:2000],
        record.get("method_name", ""),
    ]).lower()

    technique_levels = []
    for technique, level in TECHNIQUE_COMPLEXITY.items():
        if technique.replace("_", " ") in text or technique.replace("_", "-") in text:
            technique_levels.append(level)

    if technique_levels:
        # Use the highest technique complexity as a modifier
        max_technique = max(technique_levels)
        return max(base_complexity, min(max_technique, 9))

    return base_complexity


def load_jsonl(filepath: Path) -> list[dict]:
    records = []
    if not filepath.exists():
        return records
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def balance_coverage(
    records_by_type: dict[str, list[dict]],
    total_target: Optional[int] = None,
) -> list[dict]:
    """
    Balance records according to coverage targets.
    Upsamples under-represented types, downsamples over-represented.
    """
    if total_target is None:
        total_available = sum(len(v) for v in records_by_type.values())
        total_target = total_available

    balanced = []
    for comp_type, target_frac in COVERAGE_TARGETS.items():
        target_count = int(total_target * target_frac)
        available = records_by_type.get(comp_type, [])

        # Sort by quality
        available.sort(key=lambda r: r.get("_quality_score", 0), reverse=True)

        if len(available) >= target_count:
            selected = available[:target_count]
        else:
            # Upsample with repetition if needed
            selected = available[:]
            while len(selected) < target_count and available:
                selected.extend(available[:target_count - len(selected)])
            selected = selected[:target_count]

        balanced.extend(selected)

    return balanced


def build_curriculum(records: list[dict]) -> list[dict]:
    """Build the final curriculum-ordered dataset."""
    # Enrich with metadata
    for rec in records:
        rec["_quality_score"] = score_record_quality(rec)
        rec["_complexity"] = compute_complexity(rec)

    # Deduplicate
    records = deduplicate_records(records)

    # Group by competition type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        ct = rec.get("competition_type", rec.get("task", "tabular"))
        if ct not in COVERAGE_TARGETS:
            ct = "other"
        by_type[ct].append(rec)

    print("\nRaw distribution:")
    for ct, recs in sorted(by_type.items()):
        print(f"  {ct:15}: {len(recs):>6}")

    # Balance coverage
    balanced = balance_coverage(by_type)

    # Sort by complexity (curriculum order)
    balanced.sort(key=lambda r: (r["_complexity"], -r["_quality_score"]))

    # Add curriculum position tags
    for i, rec in enumerate(balanced):
        rec["_curriculum_position"] = i
        rec["_curriculum_tier"] = f"complexity_{rec['_complexity']}"

    return balanced


def print_stats(records: list[dict]) -> None:
    by_type: dict[str, int] = defaultdict(int)
    by_complexity: dict[int, int] = defaultdict(int)
    quality_sum = 0.0

    for rec in records:
        ct = rec.get("competition_type", "unknown")
        by_type[ct] += 1
        by_complexity[rec.get("_complexity", 0)] += 1
        quality_sum += rec.get("_quality_score", 0)

    total = len(records)
    print(f"\n=== COMPETITION CURRICULUM STATISTICS ===")
    print(f"Total records: {total}")
    print(f"Average quality: {quality_sum / max(total, 1):.3f}")

    print("\nBy competition type:")
    for ct, count in sorted(by_type.items(), key=lambda x: -x[1]):
        actual_pct = 100 * count / max(total, 1)
        target_pct = 100 * COVERAGE_TARGETS.get(ct, 0)
        print(f"  {ct:15}: {count:>6} ({actual_pct:.1f}% vs {target_pct:.0f}% target)")

    print("\nBy complexity level:")
    for level in sorted(by_complexity.keys()):
        count = by_complexity[level]
        labels = {1: "binary_clf", 2: "multiclass", 3: "cv_basic", 4: "nlp_basic",
                  5: "advanced", 6: "detection", 7: "multimodal", 8: "rl", 9: "expert"}
        label = labels.get(level, "unknown")
        print(f"  Level {level} ({label:12}): {count:>6}")


def main():
    parser = argparse.ArgumentParser(
        description="Build competition curriculum for Podium training data"
    )
    parser.add_argument("--input", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=CURRICULUM_DIR)
    parser.add_argument("--stats", action="store_true", help="Stats only, no output")
    parser.add_argument("--total", type=int, default=None, help="Total examples to include")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dedup-threshold", type=float, default=0.85,
                        help="MinHash similarity threshold for deduplication")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Load all data sources ─────────────────────────────────────────────────
    input_files = [
        args.input / "notebook_index.jsonl",
        args.input / "discussion_index.jsonl",
        args.input / "github_competition_solutions.jsonl",
        args.input / "pwc_results.jsonl",
        args.input / "pwc_methods.jsonl",
        args.input / "external_solutions.jsonl",
    ]

    all_records = []
    for f in input_files:
        if f.exists():
            recs = load_jsonl(f)
            print(f"Loaded {len(recs):>6} records from {f.name}")
            all_records.extend(recs)
        else:
            print(f"  [skip] {f.name}")

    if not all_records:
        print("\nNo data found. Run discovery scripts first.")
        return

    print(f"\nTotal records: {len(all_records)}")

    # ── Build curriculum ──────────────────────────────────────────────────────
    curriculum = build_curriculum(all_records)

    if args.stats:
        print_stats(curriculum)
        return

    print_stats(curriculum)

    # ── Write output ──────────────────────────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)

    # Apply total limit
    if args.total and len(curriculum) > args.total:
        curriculum = curriculum[:args.total]

    # Full curriculum
    full_path = args.output / "competition_curriculum.jsonl"
    with open(full_path, "w") as f:
        for rec in curriculum:
            f.write(json.dumps(rec) + "\n")
    print(f"\nFull curriculum: {full_path} ({len(curriculum)} records)")

    # Per-complexity-level splits
    by_level: dict[int, list] = defaultdict(list)
    for rec in curriculum:
        by_level[rec.get("_complexity", 1)].append(rec)

    for level, recs in sorted(by_level.items()):
        level_path = args.output / f"competition_level_{level}.jsonl"
        with open(level_path, "w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        print(f"  Level {level}: {level_path} ({len(recs)} records)")

    # High quality subset
    hq = [r for r in curriculum if r.get("_quality_score", 0) >= 0.7]
    hq_path = args.output / "competition_high_quality.jsonl"
    with open(hq_path, "w") as f:
        for rec in hq:
            f.write(json.dumps(rec) + "\n")
    print(f"  High quality: {hq_path} ({len(hq)} records)")

    print(f"\nNext step: python training/train.py --data {full_path}")


if __name__ == "__main__":
    main()
