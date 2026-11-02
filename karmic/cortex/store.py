"""Cortex store — ChromaDB wrapper for the three named indices (Recall /
Intuition / Caution), plus the full-episode JSONL store for layered
compression Tier 2 (structured arc). Tier 1 is `distilled_learnings`,
read straight off the episode. Tier 3 (raw originals, reversible cache) is
a simplified stub in this MVP — see README.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import chromadb

from .schema import Episode, Outcome, text_key


class CortexStore:
    def __init__(self, persist_dir: str = "./cortex_data"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir / "chroma"))
        self.recall = self.client.get_or_create_collection("recall")
        self.intuition = self.client.get_or_create_collection("intuition")
        self.caution = self.client.get_or_create_collection("caution")
        self.instinct = self.client.get_or_create_collection("instinct")

        self.episodes_path = self.persist_dir / "episodes.jsonl"
        self.episodes_path.touch(exist_ok=True)

    # ---- write-back ----

    def write_episode(self, episode: Episode) -> None:
        meta = {
            "episode_id": episode.episode_id,
            "domain": episode.domain,
            "scope": episode.scope,
            "outcome": episode.outcome.value,
        }
        # Chroma metadata can't hold None, so optional fields are present only when set.
        for key in ("outcome_value", "outcome_metric", "decision"):
            if getattr(episode, key) is not None:
                meta[key] = getattr(episode, key)

        self.recall.add(
            ids=[episode.episode_id],
            documents=[episode.problem_summary],
            metadatas=[meta],
        )

        if episode.abstracted_pattern:
            self.intuition.add(
                ids=[episode.episode_id],
                documents=[episode.abstracted_pattern],
                metadatas=[meta],
            )

        if episode.failure_summary and episode.outcome != Outcome.SUCCESS:
            self.caution.add(
                ids=[episode.episode_id],
                documents=[episode.failure_summary],
                metadatas=[meta],
            )

        # Tier 2: full structured arc, append-only JSONL.
        with open(self.episodes_path, "a") as f:
            f.write(episode.model_dump_json() + "\n")

    def promote_to_instinct(self, pattern_text: str, source_episode_ids: list[str], scope: str = "default") -> str:
        """Called by Replay once a cluster has >= 3 confirmed independent
        episodes. Returns the instinct entry id."""
        instinct_id = f"instinct::{text_key(pattern_text)}"
        self.instinct.upsert(
            ids=[instinct_id],
            documents=[pattern_text],
            metadatas=[{
                "scope": scope,
                "source_count": len(source_episode_ids),
                "source_episode_ids": ",".join(source_episode_ids),
            }],
        )
        return instinct_id

    # ---- retrieval ----

    def _query(self, collection, query_text: str, k: int, scope: Optional[str]) -> list[dict]:
        if collection.count() == 0:
            return []
        where = {"scope": scope} if scope else None
        result = collection.query(
            query_texts=[query_text],
            n_results=min(k, collection.count()),
            where=where,
        )
        hits = []
        for i in range(len(result["ids"][0])):
            hits.append({
                "id": result["ids"][0][i],
                "document": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "distance": result["distances"][0][i],
            })
        return hits

    def query_recall(self, query_text: str, k: int = 3, scope: Optional[str] = None) -> list[dict]:
        return self._query(self.recall, query_text, k, scope)

    def query_intuition(self, query_text: str, k: int = 3, scope: Optional[str] = None) -> list[dict]:
        return self._query(self.intuition, query_text, k, scope)

    def query_caution(self, query_text: str, k: int = 3, scope: Optional[str] = None) -> list[dict]:
        return self._query(self.caution, query_text, k, scope)

    def query_instinct(self, query_text: str, k: int = 3, scope: Optional[str] = None) -> list[dict]:
        """Instinct is meant to be always-included, not similarity-gated —
        callers should generally just list all entries for the scope. This
        query path exists for the rare case of ranking many instinct entries."""
        return self._query(self.instinct, query_text, k, scope)

    def all_instinct(self, scope: Optional[str] = None) -> list[dict]:
        where = {"scope": scope} if scope else None
        result = self.instinct.get(where=where)
        return [
            {"id": result["ids"][i], "document": result["documents"][i], "metadata": result["metadatas"][i]}
            for i in range(len(result["ids"]))
        ]

    # ---- full episode access (Tier 2) ----

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        return next((e for e in self.all_episodes() if e.episode_id == episode_id), None)

    def all_episodes(self) -> list[Episode]:
        episodes = []
        with open(self.episodes_path) as f:
            for line in f:
                if line.strip():
                    episodes.append(Episode.model_validate(json.loads(line)))
        return episodes

    def stats(self) -> dict:
        return {
            "recall_count": self.recall.count(),
            "intuition_count": self.intuition.count(),
            "caution_count": self.caution.count(),
            "instinct_count": self.instinct.count(),
            "total_episodes": len(self.all_episodes()),
        }
