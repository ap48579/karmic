"""Retrieval Flow — query Recall + Intuition + Caution + Instinct together,
then expand via graph spreading activation. See cortex_reasoning.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .graph import MemoryGraph
from .store import CortexStore


@dataclass
class RetrievalContext:
    recall_hits: list[dict] = field(default_factory=list)
    intuition_hits: list[dict] = field(default_factory=list)
    caution_hits: list[dict] = field(default_factory=list)
    instinct_entries: list[dict] = field(default_factory=list)
    graph_expanded_episode_ids: list[str] = field(default_factory=list)
    # episode_ids actually surfaced — used by write-back to reinforce/weaken edges
    all_surfaced_episode_ids: list[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Tier 1 compression: short, distilled, cheap to inject."""
        parts = []

        if self.recall_hits:
            parts.append("## Recall — similar past tasks (literal match)")
            for h in self.recall_hits:
                parts.append(f"- [{h['metadata']['outcome']}] {h['document']}")

        if self.intuition_hits:
            parts.append("\n## Intuition — this resembles a deeper pattern")
            for h in self.intuition_hits:
                parts.append(f"- {h['document']}")

        if self.graph_expanded_episode_ids:
            parts.append(
                f"\n## Intuition (associative) — {len(self.graph_expanded_episode_ids)} "
                "related episode(s) surfaced via memory graph, not direct similarity"
            )

        if self.caution_hits:
            parts.append("\n## Caution — known dead ends, don't retry these")
            for h in self.caution_hits:
                parts.append(f"- {h['document']}")

        if self.instinct_entries:
            parts.append("\n## Instinct — core heuristics (always apply)")
            for e in self.instinct_entries:
                parts.append(f"- {e['document']}")

        if not parts:
            return "(no prior memory — this is effectively a cold start)"
        return "\n".join(parts)


def retrieve(
    store: CortexStore,
    graph: MemoryGraph,
    query_text: str,
    scope: Optional[str] = None,
    k: int = 3,
    graph_hops: int = 2,
) -> RetrievalContext:
    ctx = RetrievalContext()

    ctx.recall_hits = store.query_recall(query_text, k=k, scope=scope)
    ctx.intuition_hits = store.query_intuition(query_text, k=k, scope=scope)
    ctx.caution_hits = store.query_caution(query_text, k=k, scope=scope)
    ctx.instinct_entries = store.all_instinct(scope=scope)

    seed_ids = list({
        h["id"] for h in (ctx.recall_hits + ctx.intuition_hits) if h["id"] in graph.graph
    })
    if seed_ids:
        activated = graph.spreading_activation(seed_ids, hops=graph_hops)
        # Only surface graph-expanded nodes that weren't already a direct hit.
        direct_ids = {h["id"] for h in ctx.recall_hits + ctx.intuition_hits}
        ctx.graph_expanded_episode_ids = [
            node for node in activated if node not in direct_ids
        ][:k]

    ctx.all_surfaced_episode_ids = list({
        h["id"] for h in (ctx.recall_hits + ctx.intuition_hits + ctx.caution_hits)
    } | set(ctx.graph_expanded_episode_ids))

    return ctx
