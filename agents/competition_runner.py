"""
Competition Runner — the main entry point for competing in a Kaggle competition.
Orchestrates all specialist agents across the full competition lifecycle.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from agents.eda_agent import EDAAgent
from agents.feature_agent import FeatureAgent
from agents.model_agent import ModelAgent
from agents.ensemble_agent import EnsembleAgent
from knowledge.competition_memory import CompetitionMemory


@dataclass
class CompetitionSession:
    competition_url: str
    data_path: Path
    output_dir: Path
    time_budget_hours: float = 168.0  # 1 week default

    # Set after parsing
    competition_type: str = ""
    metric: str = ""
    target_column: str = ""
    metric_direction: str = "higher_is_better"

    # Tracked during run
    best_cv_score: float = 0.0
    submissions: list[dict] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def hours_elapsed(self) -> float:
        return (datetime.now() - self.start_time).total_seconds() / 3600

    @property
    def hours_remaining(self) -> float:
        return self.time_budget_hours - self.hours_elapsed

    @property
    def time_fraction_used(self) -> float:
        return self.hours_elapsed / self.time_budget_hours


class CompetitionRunner:
    """
    Orchestrates the full Kaggle competition pipeline.
    Time-budget-aware: allocates effort across EDA → features → modeling → ensemble.
    """

    # Phase time allocations (fraction of total budget)
    PHASE_ALLOCATIONS = {
        "eda": 0.15,  # 15% — EDA and data understanding
        "features": 0.30,  # 30% — Feature engineering iteration
        "modeling": 0.35,  # 35% — Model training and CV
        "ensemble": 0.20,  # 20% — Ensembling and final submission
    }

    def __init__(self, model_path: str, memory_path: str = "./data/competition_memory"):
        self.eda_agent = EDAAgent(model_path)
        self.feature_agent = FeatureAgent(model_path)
        self.model_agent = ModelAgent(model_path)
        self.ensemble_agent = EnsembleAgent(model_path)
        self.memory = CompetitionMemory(memory_path)

    def parse_competition(self, session: CompetitionSession) -> dict:
        """Parse competition URL → Universal Competition Description (UCD)."""
        import httpx

        slug = session.competition_url.rstrip("/").split("/")[-1]
        logger.info(f"Parsing competition: {slug}")

        # Fetch competition overview
        try:
            with httpx.Client() as client:
                resp = client.get(
                    f"https://www.kaggle.com/api/v1/competitions/{slug}",
                    headers={"Authorization": f"Bearer {_get_kaggle_key()}"},
                )
                meta = resp.json()
        except Exception as e:
            logger.warning(f"Could not fetch competition API: {e}, using defaults")
            meta = {}

        return {
            "competition_id": slug,
            "title": meta.get("title", slug),
            "description": meta.get("description", ""),
            "evaluation_metric": meta.get("evaluationMetric", ""),
            "deadline": meta.get("deadline"),
            "reward": meta.get("reward"),
        }

    def detect_competition_type(self, data_path: Path, meta: dict) -> str:
        """Heuristic competition type detection from data structure and metadata."""
        files = list(data_path.glob("*"))
        has_images = any(f.suffix in {".jpg", ".png", ".jpeg", ".tif"} for f in files)
        has_audio = any(f.suffix in {".mp3", ".wav", ".ogg", ".flac"} for f in files)

        if has_images and has_audio:
            return "multimodal"
        if has_images:
            return "computer_vision"
        if has_audio:
            return "audio"

        # Check for text columns in train.csv
        try:
            import pandas as pd

            train = pd.read_csv(data_path / "train.csv", nrows=100)
            text_cols = [
                c
                for c in train.columns
                if train[c].dtype == object and _safe_mean_len(train[c]) > 50
            ]
            time_cols = [
                c
                for c in train.columns
                if any(
                    kw in c.lower() for kw in ["date", "time", "year", "month", "week"]
                )
            ]
            if text_cols:
                return "nlp"
            if time_cols:
                return "time_series"
        except Exception:
            pass

        return "tabular"

    def retrieve_similar_competitions(self, session: CompetitionSession) -> list[dict]:
        """Query competition memory for similar historical competitions."""
        similar = self.memory.search_similar(
            competition_type=session.competition_type,
            metric=session.metric,
            top_k=5,
        )
        if similar:
            logger.info(f"Found {len(similar)} similar historical competitions")
        return similar

    async def run_eda_phase(self, session: CompetitionSession) -> dict:
        """Phase 1: EDA — understand the data. Runs CPU-bound work in a thread."""
        phase_budget = self.PHASE_ALLOCATIONS["eda"] * session.time_budget_hours
        logger.info(f"Starting EDA phase ({phase_budget:.1f}h budget)")

        eda_results = await asyncio.to_thread(
            self.eda_agent.analyze,
            data_path=session.data_path,
            competition_type=session.competition_type,
            metric=session.metric,
            time_budget_hours=phase_budget,
        )

        logger.info(
            f"EDA complete. Key findings: {len(eda_results.get('insights', []))}"
        )
        return eda_results

    async def run_feature_phase(
        self, session: CompetitionSession, eda_results: dict
    ) -> dict:
        """Phase 2: Feature engineering — iterative improvement. Runs in a thread."""
        phase_budget = self.PHASE_ALLOCATIONS["features"] * session.time_budget_hours
        logger.info(f"Starting feature engineering phase ({phase_budget:.1f}h budget)")

        feature_results = await asyncio.to_thread(
            self.feature_agent.engineer,
            data_path=session.data_path,
            eda_results=eda_results,
            competition_type=session.competition_type,
            metric=session.metric,
            target_column=session.target_column,
            time_budget_hours=phase_budget,
        )

        best_cv = feature_results.get("best_cv_score", 0)
        logger.info(f"Feature phase complete. Best CV: {best_cv:.5f}")
        session.best_cv_score = best_cv
        return feature_results

    async def run_modeling_phase(
        self,
        session: CompetitionSession,
        feature_results: dict,
    ) -> dict:
        """Phase 3: Model training and cross-validation. Runs in a thread."""
        phase_budget = self.PHASE_ALLOCATIONS["modeling"] * session.time_budget_hours
        logger.info(f"Starting modeling phase ({phase_budget:.1f}h budget)")

        model_results = await asyncio.to_thread(
            self.model_agent.train,
            data_path=session.data_path,
            feature_results=feature_results,
            competition_type=session.competition_type,
            metric=session.metric,
            target_column=session.target_column,
            metric_direction=session.metric_direction,
            time_budget_hours=phase_budget,
        )

        best_cv = model_results.get("best_cv_score", session.best_cv_score)
        logger.info(f"Modeling phase complete. Best CV: {best_cv:.5f}")
        session.best_cv_score = best_cv
        return model_results

    async def run_ensemble_phase(
        self,
        session: CompetitionSession,
        model_results: dict,
    ) -> dict:
        """Phase 4: Ensembling and final submission generation. Runs in a thread."""
        phase_budget = self.PHASE_ALLOCATIONS["ensemble"] * session.time_budget_hours
        logger.info(f"Starting ensemble phase ({phase_budget:.1f}h budget)")

        ensemble_results = await asyncio.to_thread(
            self.ensemble_agent.build,
            model_results=model_results,
            data_path=session.data_path,
            metric=session.metric,
            metric_direction=session.metric_direction,
            time_budget_hours=phase_budget,
        )

        final_cv = ensemble_results.get("ensemble_cv_score", session.best_cv_score)
        logger.info(f"Ensemble phase complete. Final CV: {final_cv:.5f}")
        return ensemble_results

    async def compete(
        self,
        competition_url: str,
        data_path: str,
        output_dir: str,
        time_budget_hours: float = 168.0,
    ) -> dict:
        """
        Full competition run. Returns final results dict with submission path and CV score.
        """
        session = CompetitionSession(
            competition_url=competition_url,
            data_path=Path(data_path),
            output_dir=Path(output_dir),
            time_budget_hours=time_budget_hours,
        )
        session.output_dir.mkdir(parents=True, exist_ok=True)

        train_csv = session.data_path / "train.csv"
        if not train_csv.exists():
            raise FileNotFoundError(
                f"train.csv not found at {train_csv}. Check that data_path is correct."
            )

        logger.info(f"Podium starting competition: {competition_url}")
        logger.info(f"Time budget: {time_budget_hours}h | Data: {data_path}")

        # Parse and classify competition
        meta = self.parse_competition(session)
        session.competition_type = self.detect_competition_type(session.data_path, meta)
        session.metric = meta.get("evaluation_metric", "auc")
        session.metric_direction = _metric_direction(session.metric)
        logger.info(
            f"Competition type: {session.competition_type} | Metric: {session.metric} ({session.metric_direction})"
        )

        # Retrieve similar historical competitions from memory
        self.retrieve_similar_competitions(session)

        # Run pipeline phases
        eda_results = await self.run_eda_phase(session)
        feature_results = await self.run_feature_phase(session, eda_results)
        model_results = await self.run_modeling_phase(session, feature_results)
        ensemble_results = await self.run_ensemble_phase(session, model_results)

        # Select best submission — ensemble returns "best_submission_path",
        # model_agent returns "submission_paths" (dict keyed by model name)
        _model_sub_paths = model_results.get("submission_paths", {})
        _best_single = _model_sub_paths.get(model_results.get("best_model", ""))
        submission_path = ensemble_results.get("best_submission_path") or _best_single

        final_results = {
            "competition": competition_url,
            "competition_type": session.competition_type,
            "metric": session.metric,
            "final_cv_score": ensemble_results.get(
                "ensemble_cv_score", session.best_cv_score
            ),
            "submission_path": submission_path,
            "hours_used": session.hours_elapsed,
            "phases": {
                "eda": eda_results,
                "features": feature_results,
                "modeling": model_results,
                "ensemble": ensemble_results,
            },
        }

        # Save results
        results_path = session.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2, default=str)

        # Update competition memory with what we learned
        self.memory.record_competition(session, final_results)

        logger.info(f"Competition complete. CV: {final_results['final_cv_score']:.5f}")
        logger.info(f"Submission: {submission_path}")
        return final_results


def _metric_direction(metric: str) -> str:
    """Return 'lower_is_better' for error/loss metrics, else 'higher_is_better'."""
    m = metric.lower().replace(" ", "_").replace("-", "_")
    return (
        "lower_is_better"
        if any(
            kw in m
            for kw in ["rmse", "rmsle", "mse", "mae", "error", "loss", "logloss"]
        )
        else "higher_is_better"
    )


def _get_kaggle_key() -> str:
    import os

    return os.environ.get("KAGGLE_KEY", "")


def _safe_mean_len(series) -> float:
    """Compute mean string length, returning 0 if the column has non-string values."""
    try:
        lengths = series.str.len()
        mean_val = lengths.mean()
        return float(mean_val) if pd.notna(mean_val) else 0.0
    except Exception:
        return 0.0


if __name__ == "__main__":
    import typer

    def main(
        competition_url: str,
        data_path: str,
        output_dir: str = "./results/competition",
        model_path: str = "./checkpoints/dpo",
        time_budget_hours: float = 168.0,
    ):
        runner = CompetitionRunner(model_path=model_path)
        results = asyncio.run(
            runner.compete(competition_url, data_path, output_dir, time_budget_hours)
        )
        print(f"\nFinal CV Score: {results['final_cv_score']:.5f}")
        print(f"Submission: {results['submission_path']}")

    typer.run(main)
