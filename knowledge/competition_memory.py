"""
Competition Memory — persistent cross-competition learning.
Every competition Podium runs makes it smarter.

Uses ChromaDB for vector similarity search + JSON for structured pattern storage.
This is Differentiator 3: the compound learning that no other system has.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from loguru import logger
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class CompetitionMemoryEntry:
    competition_id: str
    competition_type: str
    metric: str
    final_cv_score: float
    leaderboard_percentile: float | None

    # What worked
    winning_features: list[str]
    winning_models: list[str]
    ensemble_strategy: str
    key_insight: str

    # What didn't work (equally important)
    failed_approaches: list[str]

    # Metadata
    timestamp: str
    time_used_hours: float


class CompetitionMemory:
    """
    Persistent memory store for cross-competition pattern learning.
    Enables: "this competition looks like X — here's what worked there."
    """

    def __init__(self, memory_path: str = "./data/competition_memory"):
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(parents=True, exist_ok=True)

        # ChromaDB for semantic similarity search
        self.chroma = chromadb.PersistentClient(path=str(self.memory_path / "chroma"))
        self.collection = self.chroma.get_or_create_collection(
            name="competitions",
            metadata={"hnsw:space": "cosine"},
        )

        # Sentence transformer for competition fingerprinting
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Structured records (JSON)
        self.records_path = self.memory_path / "records.jsonl"

        # In-memory index for O(1) record lookups (competition_id → record dict).
        # Populated lazily on first access; invalidated when record_competition() appends.
        self._record_index: dict[str, dict] | None = None

        logger.info(
            f"Competition memory loaded: {self.collection.count()} competitions"
        )

    def _fingerprint(
        self, competition_type: str, metric: str, description: str = ""
    ) -> str:
        """Create a text fingerprint for embedding-based similarity search."""
        return f"{competition_type} {metric} {description}".strip()

    def record_competition(self, session: Any, results: dict) -> None:
        """
        Persist what we learned from a competition.
        Called after every competition run.
        """
        phases = results.get("phases", {})
        feature_phase = phases.get("features", {})
        model_phase = phases.get("modeling", {})
        ensemble_phase = phases.get("ensemble", {})

        entry = CompetitionMemoryEntry(
            competition_id=session.competition_url.split("/")[-1],
            competition_type=session.competition_type,
            metric=session.metric,
            final_cv_score=float(
                results.get("final_cv_score", 0)
            ),  # ensure JSON-serializable
            leaderboard_percentile=results.get("leaderboard_percentile"),
            winning_features=feature_phase.get("top_features", []),
            winning_models=model_phase.get("best_models", []),
            ensemble_strategy=ensemble_phase.get("strategy", ""),
            key_insight=results.get("key_insight", ""),
            failed_approaches=results.get("failed_approaches", []),
            timestamp=datetime.now().isoformat(),
            time_used_hours=float(results.get("hours_used", 0)),
        )

        # Embed and store in ChromaDB
        fingerprint = self._fingerprint(
            entry.competition_type,
            entry.metric,
            entry.key_insight,
        )
        embedding = self.embedder.encode(fingerprint).tolist()

        # Use upsert so re-running the same competition updates rather than crashing
        self.collection.upsert(
            ids=[entry.competition_id],
            embeddings=[embedding],
            metadatas=[
                {
                    "competition_type": entry.competition_type,
                    "metric": entry.metric,
                    "final_cv_score": entry.final_cv_score,
                    "ensemble_strategy": entry.ensemble_strategy,
                }
            ],
            documents=[fingerprint],
        )

        # Append to structured records and update in-memory index
        record_dict = asdict(entry)
        with open(self.records_path, "a") as f:
            f.write(json.dumps(record_dict) + "\n")

        # Keep index consistent: if already loaded, update in place; otherwise invalidate
        # so the next _load_record() call will rebuild from disk.
        if self._record_index is not None:
            self._record_index[entry.competition_id] = record_dict

        logger.info(
            f"Recorded competition {entry.competition_id} → memory ({self.collection.count()} total)"
        )

    def search_similar(
        self,
        competition_type: str,
        metric: str,
        description: str = "",
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find most similar historical competitions.
        Returns list of structured memory entries with strategies that worked.
        """
        if self.collection.count() == 0:
            return []

        fingerprint = self._fingerprint(competition_type, metric, description)
        query_embedding = self.embedder.encode(fingerprint).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["metadatas", "documents", "distances"],
        )

        similar = []
        for i, comp_id in enumerate(results["ids"][0]):
            entry = self._load_record(comp_id)
            if entry:
                entry["similarity_score"] = 1 - results["distances"][0][i]
                similar.append(entry)

        return sorted(similar, key=lambda x: x.get("similarity_score", 0), reverse=True)

    def get_strategy_recommendation(
        self,
        competition_type: str,
        metric: str,
        description: str = "",
    ) -> str:
        """
        Generate a natural language strategy recommendation based on similar competitions.
        Used to prime agents with historical context.
        """
        similar = self.search_similar(competition_type, metric, description, top_k=3)
        if not similar:
            return "No historical competitions found. Starting fresh."

        recommendations = []
        for comp in similar:
            sim_pct = comp.get("similarity_score", 0) * 100
            recommendations.append(
                f"- {comp['competition_id']} ({sim_pct:.0f}% similar, CV {comp['final_cv_score']:.4f}): "
                f"Best features: {', '.join(comp.get('winning_features', [])[:3])} | "
                f"Best models: {', '.join(comp.get('winning_models', [])[:2])} | "
                f"Ensemble: {comp.get('ensemble_strategy', 'N/A')} | "
                f"Key insight: {comp.get('key_insight', 'N/A')}"
            )

        return "Similar competitions from memory:\n" + "\n".join(recommendations)

    def _build_record_index(self) -> dict[str, dict]:
        """Build in-memory index from records.jsonl.  O(n) once, then O(1) per lookup."""
        index: dict[str, dict] = {}
        if not self.records_path.exists():
            return index
        with open(self.records_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping corrupted memory record: {line[:80]}")
                    continue
                # Later entries overwrite earlier ones (same semantics as the old linear scan)
                cid = record.get("competition_id")
                if cid:
                    index[cid] = record
        return index

    def _load_record(self, competition_id: str) -> dict | None:
        """Load a specific competition record from the in-memory index (O(1))."""
        if self._record_index is None:
            self._record_index = self._build_record_index()
        return self._record_index.get(competition_id)

    def stats(self) -> dict:
        """Summary statistics about what's in memory."""
        if not self.records_path.exists():
            return {"total_competitions": 0}

        records = []
        with open(self.records_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping corrupted record in stats(): {line[:80]}")
                    continue

        type_counts = {}
        for r in records:
            t = r.get("competition_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_competitions": len(records),
            "by_type": type_counts,
            "avg_cv_score": sum(r.get("final_cv_score", 0) for r in records)
            / max(len(records), 1),
        }
