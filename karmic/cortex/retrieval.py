"""Retrieval Flow — query Recall + Intuition + Caution + Instinct together,
then expand via graph spreading activation. See cortex_reasoning.md."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

from .graph import EvidenceGate, MemoryGraph, pattern_node_id
from .store import CortexStore


def _label(meta: dict) -> str:
    """Declined decisions are labeled so an over-cautious streak is visible."""
    return f"declined · {meta['outcome']}" if meta.get("decision") == "declined" else meta["outcome"]


def _evidence(stats: dict) -> str:
    """Counts computed by code, printed beside a pattern so the model reads evidence,
    not just the pattern's own assertion."""
    if not stats.get("n"):
        return ""
    text = (f" (seen {stats['n']}×: {stats.get('success', 0)} success / "
            f"{stats.get('failure', 0)} failure / {stats.get('partial', 0)} partial")
    if stats.get("outcome_values"):
        values = stats["outcome_values"]
        text += (f"; {stats['outcome_metric']} mean {statistics.fmean(values):+.4g}, "
                 f"median {statistics.median(values):+.4g}")
    return text + ")"


def _hit_sections(recall: list[dict], intuition: list[dict], caution: list[dict], heading: str) -> list[str]:
    parts = []
    if recall:
        parts.append(f"\n{heading} Recall — similar past tasks (literal match)")
        for h in recall:
            parts.append(f"- [{_label(h['metadata'])}] {h['document']}")
    if intuition:
        parts.append(f"\n{heading} Intuition — this resembles a deeper pattern")
        for h in intuition:
            parts.append(f"- {h['document']}{_evidence(h['stats'])}")
    if caution:
        parts.append(f"\n{heading} Caution — known dead ends, don't retry these")
        for h in caution:
            parts.append(f"- {h['document']}")
    return parts


@dataclass
class RetrievalContext:
    recall_hits: list[dict] = field(default_factory=list)
    intuition_hits: list[dict] = field(default_factory=list)
    caution_hits: list[dict] = field(default_factory=list)
    instinct_entries: list[dict] = field(default_factory=list)
    graph_expanded_episode_ids: list[str] = field(default_factory=list)
    # episode_ids actually surfaced — used by write-back to reinforce/weaken edges
    all_surfaced_episode_ids: list[str] = field(default_factory=list)
    # Set by retrieve_many: each hit then carries the "query" it was found for.
    queries: list[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Tier 1 compression: short, distilled, cheap to inject."""
        if self.queries:
            parts = []
            for q in self.queries:
                mine = [[h for h in hits if h["query"] == q]
                        for hits in (self.recall_hits, self.intuition_hits, self.caution_hits)]
                parts.append(f"\n## Item: {q}")
                parts.extend(_hit_sections(*mine, heading="###") or ["(no prior memory for this item)"])
        else:
            parts = _hit_sections(self.recall_hits, self.intuition_hits, self.caution_hits, heading="##")

        if self.graph_expanded_episode_ids:
            parts.append(
                f"\n## Intuition (associative) — {len(self.graph_expanded_episode_ids)} "
                "related episode(s) surfaced via memory graph, not direct similarity"
            )

        if self.instinct_entries:
            parts.append("\n## Instinct — core heuristics (always apply)")
            for e in self.instinct_entries:
                parts.append(f"- {e['document']}{_evidence(e['stats'])}")

        if not parts:
            return "(no prior memory — this is effectively a cold start)"
        return "\n".join(parts).lstrip("\n")


def _direct_hits(
    store: CortexStore,
    graph: MemoryGraph,
    query_text: str,
    k: int,
    scope: Optional[str],
    evidence_gate: Optional[EvidenceGate],
) -> tuple[list[dict], list[dict], list[dict]]:
    recall = store.query_recall(query_text, k=k, scope=scope)
    intuition = store.query_intuition(query_text, k=k, scope=scope)
    for h in intuition:
        h["stats"] = graph.outcome_stats(pattern_node_id(h["document"]))
    if evidence_gate is not None:
        # Process episodes (no `decision`) are exempt: their outcome is known at once.
        intuition = [h for h in intuition
                     if "decision" not in h["metadata"] or h["stats"].get("n", 0) >= evidence_gate.min_n]
    caution = store.query_caution(query_text, k=k, scope=scope)
    return recall, intuition, caution


def _expand(ctx: RetrievalContext, store: CortexStore, graph: MemoryGraph,
            scope: Optional[str], k: int, graph_hops: int) -> RetrievalContext:
    ctx.instinct_entries = store.all_instinct(scope=scope)
    for e in ctx.instinct_entries:
        e["stats"] = graph.outcome_stats(pattern_node_id(e["document"]))

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


def retrieve(
    store: CortexStore,
    graph: MemoryGraph,
    query_text: str,
    scope: Optional[str] = None,
    k: int = 3,
    graph_hops: int = 2,
    evidence_gate: Optional[EvidenceGate] = None,
) -> RetrievalContext:
    ctx = RetrievalContext()
    ctx.recall_hits, ctx.intuition_hits, ctx.caution_hits = _direct_hits(
        store, graph, query_text, k, scope, evidence_gate
    )
    return _expand(ctx, store, graph, scope, k, graph_hops)


def retrieve_many(
    store: CortexStore,
    graph: MemoryGraph,
    queries: list[str],
    k_each: int = 3,
    k_total: int = 10,
    scope: Optional[str] = None,
    graph_hops: int = 2,
    evidence_gate: Optional[EvidenceGate] = None,
) -> RetrievalContext:
    """One query per item rather than one blended query: the embedding of several items
    concatenated matches none of them well. An episode found for several items is kept
    once, under the item it matched most closely; each index then keeps its k_total
    closest hits."""
    best: list[dict[str, dict]] = [{}, {}, {}]  # recall, intuition, caution: id -> hit
    for q in queries:
        for kept, hits in zip(best, _direct_hits(store, graph, q, k_each, scope, evidence_gate)):
            for h in hits:
                if h["id"] not in kept or h["distance"] < kept[h["id"]]["distance"]:
                    kept[h["id"]] = {**h, "query": q}

    ctx = RetrievalContext(queries=list(queries))
    ctx.recall_hits, ctx.intuition_hits, ctx.caution_hits = (
        sorted(kept.values(), key=lambda h: h["distance"])[:k_total] for kept in best
    )
    return _expand(ctx, store, graph, scope, k_total, graph_hops)
