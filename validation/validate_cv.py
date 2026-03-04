"""
CV score execution harness — the RL reward signal for Stage 2 GRPO training.

This is Podium's key technical differentiator: we execute generated code in an
isolated Docker container, run cross-validation, and use ΔCV score as reward.

Same insight as DeepSeek-R1 (verifiable execution reward) applied to Kaggle.
"""

import json
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import docker
from loguru import logger


@dataclass
class ExecutionResult:
    success: bool
    cv_score: float | None
    error: str | None
    execution_time_s: float
    reward: float


EXECUTION_TIMEOUT_S = 120
DOCKER_IMAGE = "podium-execution:latest"

# Module-level Docker client — created once at import time to avoid the overhead
# of docker.from_env() on every invocation (PO-4).
try:
    _DOCKER_CLIENT = docker.from_env()
except Exception:
    _DOCKER_CLIENT = None  # Docker unavailable (e.g. in tests without daemon)


def compute_reward(
    cv_score: float | None,
    baseline_cv: float,
    success: bool,
    metric_direction: str = "higher_is_better",
) -> float:
    """
    Compute GRPO reward from CV execution result.

    - Failed execution:        -1.0 (hard penalty)
    - Score below baseline:    -0.5 (soft penalty)
    - Score improvement:       normalized delta clamped to [0, 1]
    """
    if not success or cv_score is None:
        return -1.0

    # Guard against degenerate baseline (e.g. all-same target → AUC = 0.5, RMSE = 0).
    # When baseline is exactly zero (e.g. perfect RMSE on trivial data), any non-zero
    # score would yield an infinite reward — return 1.0 (perfect score) instead.
    if abs(baseline_cv) < 1e-9:
        return 1.0

    if metric_direction == "higher_is_better":
        delta = cv_score - baseline_cv
    else:
        delta = baseline_cv - cv_score

    if delta < 0:
        return -0.5

    # 1% improvement → ~0.1 reward, 10% improvement → ~1.0 reward
    return min(delta / (abs(baseline_cv) + 1e-8) * 10, 1.0)


def build_execution_script(
    generated_code: str,
    data_path: str,
    target_column: str,
    metric: str,
    cv_folds: int = 5,
) -> str:
    """
    Wrap generated code in a CV harness that outputs a JSON result line.
    Generated code is dedented and injected at module level — no extra indentation.
    """
    clean_code = textwrap.dedent(generated_code).strip()

    return f"""import warnings
warnings.filterwarnings('ignore')
import json
import sys
import numpy as np
import pandas as pd

# Competition context available to generated code
DATA_PATH = {json.dumps(data_path)}
TARGET = {json.dumps(target_column)}
METRIC = {json.dumps(metric)}
CV_FOLDS = {cv_folds}

# ── BEGIN GENERATED CODE ─────────────────────────────────────────────────────
{clean_code}
# ── END GENERATED CODE ───────────────────────────────────────────────────────

# Capture result
try:
    import math as _math
    _score = float(cv_score)
    if _math.isnan(_score) or _math.isinf(_score):
        print(json.dumps({{"error": f"cv_score is {{_score}} (not finite)", "success": False}}))
        sys.exit(1)
    print(json.dumps({{"cv_score": _score, "success": True}}))
except NameError:
    print(json.dumps({{"error": "cv_score variable not set by generated code", "success": False}}))
    sys.exit(1)
except (TypeError, ValueError) as e:
    print(json.dumps({{"error": f"cv_score is not numeric: {{e}}", "success": False}}))
    sys.exit(1)
"""


def execute_in_docker(
    code: str,
    data_mount: str,
    client: docker.DockerClient,
) -> tuple[bool, float | None, str | None]:
    """
    Execute code string in isolated Docker container.
    Uses detach + wait pattern to support timeout without crashing the SDK.
    Returns (success, cv_score, error_message).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()  # Ensure fully written to disk before Docker reads it
        script_path = f.name

    container = None
    try:
        container = client.containers.run(
            image=DOCKER_IMAGE,
            command="python /workspace/script.py",
            volumes={
                script_path: {"bind": "/workspace/script.py", "mode": "ro"},
                data_mount: {"bind": "/data", "mode": "ro"},
            },
            mem_limit="8g",
            nano_cpus=4_000_000_000,
            network_disabled=True,
            detach=True,      # Run detached so we can apply timeout
            remove=False,     # We remove manually in finally block
        )

        # Wait with timeout
        try:
            exit_result = container.wait(timeout=EXECUTION_TIMEOUT_S)
            exit_code = exit_result.get("StatusCode", 1)
        except Exception:
            return False, None, f"execution timed out after {EXECUTION_TIMEOUT_S}s"

        raw_logs = container.logs(stdout=True, stderr=False)
        output = raw_logs.decode("utf-8", errors="replace").strip()

        if exit_code != 0 or not output:
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            return False, None, (stderr or "non-zero exit, no output")[:500]

        last_line = output.split("\n")[-1]
        parsed = json.loads(last_line)
        if parsed.get("success"):
            return True, float(parsed["cv_score"]), None
        return False, None, parsed.get("error", "unknown error")

    except json.JSONDecodeError:
        return False, None, f"invalid JSON output: {output[:200]}"
    except docker.errors.ImageNotFound:
        return False, None, (
            f"Docker image '{DOCKER_IMAGE}' not found. "
            f"Build it first: docker build -f deploy/Dockerfile.execution -t {DOCKER_IMAGE} ."
        )
    except Exception as e:
        return False, None, f"docker error: {type(e).__name__}: {e}"
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                pass
        Path(script_path).unlink(missing_ok=True)


def score_generated_code(
    generated_code: str,
    data_path: str,
    target_column: str,
    metric: str,
    baseline_cv: float,
    metric_direction: str = "higher_is_better",
) -> ExecutionResult:
    """Full pipeline: wrap code → execute → compute reward."""
    start = time.time()
    client = _DOCKER_CLIENT

    if client is None:
        return ExecutionResult(
            success=False,
            cv_score=None,
            error="Docker client unavailable (daemon not running or not installed)",
            execution_time_s=time.time() - start,
            reward=-1.0,
        )

    # Generated code sees /data (container-internal path), not the host data_path
    harness = build_execution_script(generated_code, "/data", target_column, metric)
    success, cv_score, error = execute_in_docker(harness, data_path, client)
    elapsed = time.time() - start

    reward = compute_reward(cv_score, baseline_cv, success, metric_direction)

    return ExecutionResult(
        success=success,
        cv_score=cv_score,
        error=error,
        execution_time_s=elapsed,
        reward=reward,
    )


def batch_score(
    code_samples: list[str],
    data_path: str,
    target_column: str,
    metric: str,
    baseline_cv: float,
    metric_direction: str = "higher_is_better",
    max_workers: int = 4,
) -> list[ExecutionResult]:
    """Score multiple code samples in parallel (for GRPO batch processing)."""
    if not code_samples:
        return []

    def score_one(code: str) -> ExecutionResult:
        return score_generated_code(
            code, data_path, target_column, metric, baseline_cv, metric_direction
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(score_one, code_samples))

    if results:
        success_rate = sum(r.success for r in results) / len(results)
        avg_reward = sum(r.reward for r in results) / len(results)
        logger.info(f"Batch CV scoring: {success_rate:.0%} success, avg reward {avg_reward:.3f}")
    return results
