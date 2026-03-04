"""
Podium Master Pipeline
Orchestrates the full data → training → evaluation pipeline.
~76 hours total on 18× A6000 + Azure burst for synthesis.

Usage:
  python pipeline.py                          # Full pipeline
  python pipeline.py --stage discovery        # Step 1: collect data
  python pipeline.py --stage synthesis        # Step 2: generate pairs
  python pipeline.py --stage train            # Step 3: 3-stage training
  python pipeline.py --stage eval             # Step 4: PodiumBench
"""

import os
import subprocess
import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()
app = typer.Typer()


STAGES = [
    # ── Discovery ────────────────────────────────────────────────────────
    {
        "name": "check_env",
        "description": "Verify environment and GPU setup",
        "cmd": "bash scripts/check_env.sh",
        "phase": "discovery",
        "estimated_hours": 0.1,
    },
    {
        "name": "discover_competitions",
        "description": "Fetch all Kaggle competition metadata",
        "cmd": "python discovery/kaggle_notebooks.py --output-dir data/raw/notebooks",
        "phase": "discovery",
        "estimated_hours": 1.0,
    },
    {
        "name": "fetch_notebooks",
        "description": "Download 300k+ competition notebooks (30 workers)",
        "cmd": "python discovery/fetch_bulk.py --workers 30",
        "phase": "discovery",
        "estimated_hours": 3.0,
    },
    {
        "name": "collect_writeups",
        "description": "Scrape winning solution writeups from competition forums",
        "cmd": "python discovery/solution_writeups.py",
        "phase": "discovery",
        "estimated_hours": 2.0,
    },
    # ── Synthesis ────────────────────────────────────────────────────────
    {
        "name": "start_vllm",
        "description": "Launch Qwen2.5-72B synthesis servers (Azure burst)",
        "cmd": "bash scripts/start_vllm.sh",
        "phase": "synthesis",
        "estimated_hours": 0.5,
    },
    {
        "name": "synthesize_notebooks",
        "description": "Synthesize (problem, reasoning, code, score) pairs from notebooks",
        "cmd": "python synthesis/synthesize_bulk.py --concurrency 32",
        "phase": "synthesis",
        "estimated_hours": 20.0,
    },
    {
        "name": "synthesize_writeups",
        "description": "Extract structured strategy pairs from winning writeups",
        # synthesize_bulk.py uses --notebook-dir and --index, not --input.
        # Writeup notebooks are stored in data/raw/writeups/files with index at
        # data/raw/writeups/notebook_index.jsonl (produced by solution_writeups.py).
        "cmd": (
            "python synthesis/synthesize_bulk.py"
            " --notebook-dir data/raw/writeups/files"
            " --index data/raw/writeups/notebook_index.jsonl"
            " --output data/synthesized/writeup_pairs.jsonl"
        ),
        "phase": "synthesis",
        "estimated_hours": 12.0,
    },
    {
        "name": "synthesize_dialogues",
        "description": "Generate multi-turn competition lifecycle dialogues",
        "cmd": "python synthesis/multi_turn.py",
        "phase": "synthesis",
        "estimated_hours": 8.0,
    },
    {
        "name": "synthesize_techniques",
        "description": "Synthesize technique → competition application pairs",
        "cmd": "python synthesis/technique_pairs.py",
        "phase": "synthesis",
        "estimated_hours": 6.0,
    },
    # ── Validation ────────────────────────────────────────────────────────
    {
        "name": "validate_pairs",
        "description": "Quality filter and deduplication (MinHash)",
        "cmd": "python validation/validate.py",
        "phase": "validation",
        "estimated_hours": 2.0,
    },
    {
        "name": "prepare_rl_tasks",
        "description": "Build CV execution task set for Stage 2 RL",
        "cmd": "python validation/prepare_rl_tasks.py",
        "phase": "validation",
        "estimated_hours": 4.0,
    },
    {
        "name": "prepare_dpo_pairs",
        "description": "Build strategy preference pairs for Stage 3 DPO",
        "cmd": "python synthesis/dpo_pairs.py",
        "phase": "validation",
        "estimated_hours": 3.0,
    },
    # ── Training ────────────────────────────────────────────────────────
    {
        "name": "train_sft",
        "description": "Stage 1: Supervised Fine-Tuning (6h on 18× A6000)",
        "cmd": "torchrun --nproc_per_node=18 training/train.py",
        "phase": "train",
        "estimated_hours": 6.0,
    },
    {
        "name": "train_rl",
        "description": "Stage 2: CV-Verified RL with GRPO (4h on 18× A6000)",
        "cmd": "torchrun --nproc_per_node=18 training/train_rl.py",
        "phase": "train",
        "estimated_hours": 4.0,
    },
    {
        "name": "train_dpo",
        "description": "Stage 3: DPO on competition strategy preferences (2h)",
        "cmd": "torchrun --nproc_per_node=18 training/train_dpo.py",
        "phase": "train",
        "estimated_hours": 2.0,
    },
    # ── Evaluation ────────────────────────────────────────────────────────
    {
        "name": "podium_bench",
        "description": "PodiumBench evaluation on 75 competitions",
        "cmd": "python evaluation/eval.py",
        "phase": "eval",
        "estimated_hours": 4.0,
    },
    # ── Deploy ────────────────────────────────────────────────────────────
    {
        "name": "deploy",
        "description": "Launch Podium API server (Docker)",
        "cmd": "docker compose -f deploy/docker-compose.yml up -d",
        "phase": "deploy",
        "estimated_hours": 0.2,
    },
]


def run_stage(stage: dict, dry_run: bool = False) -> bool:
    """Execute a pipeline stage. Returns True on success."""
    console.print(f"\n[bold cyan]▶ {stage['name']}[/bold cyan]: {stage['description']}")
    console.print(f"  [dim]{stage['cmd']}[/dim]")

    if dry_run:
        console.print("  [yellow](dry run — skipping)[/yellow]")
        return True

    result = subprocess.run(stage["cmd"], shell=True)
    if result.returncode != 0:
        console.print(f"  [red]✗ Failed (exit {result.returncode})[/red]")
        return False

    console.print(f"  [green]✓ Complete[/green]")
    return True


@app.command()
def main(
    stage: str = typer.Option(
        None,
        help="Run only this phase: discovery | synthesis | validation | train | eval | deploy"
    ),
    from_stage: str = typer.Option(None, help="Resume pipeline from this stage name"),
    dry_run: bool = typer.Option(False, help="Print commands without executing"),
    list_stages: bool = typer.Option(False, "--list", help="List all stages and exit"),
):
    """Podium: full training pipeline from raw data to deployed model."""

    if list_stages:
        table = Table(title="Podium Pipeline Stages")
        table.add_column("Stage", style="cyan")
        table.add_column("Phase")
        table.add_column("Description")
        table.add_column("Est. Hours", justify="right")
        for s in STAGES:
            table.add_row(s["name"], s["phase"], s["description"], str(s["estimated_hours"]))
        console.print(table)

        total = sum(s["estimated_hours"] for s in STAGES)
        console.print(f"\nTotal estimated: {total:.1f} hours")
        return

    # Filter stages
    stages_to_run = STAGES
    if stage:
        stages_to_run = [s for s in STAGES if s["phase"] == stage]
        if not stages_to_run:
            console.print(f"[red]Unknown phase: {stage}[/red]")
            raise typer.Exit(1)
    elif from_stage:
        names = [s["name"] for s in STAGES]
        if from_stage not in names:
            console.print(f"[red]Unknown stage: {from_stage}[/red]")
            raise typer.Exit(1)
        idx = names.index(from_stage)
        stages_to_run = STAGES[idx:]

    total_hours = sum(s["estimated_hours"] for s in stages_to_run)
    console.print(f"\n[bold]Podium Pipeline[/bold] — {len(stages_to_run)} stages, ~{total_hours:.0f}h estimated")
    if dry_run:
        console.print("[yellow]DRY RUN MODE[/yellow]")

    # Run
    for s in stages_to_run:
        success = run_stage(s, dry_run=dry_run)
        if not success:
            console.print(f"\n[red bold]Pipeline failed at stage: {s['name']}[/red bold]")
            console.print(f"To resume: python pipeline.py --from-stage {s['name']}")
            raise typer.Exit(1)

    console.print("\n[green bold]Pipeline complete.[/green bold]")


if __name__ == "__main__":
    app()
