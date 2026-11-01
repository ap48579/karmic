"""Sleep-like Consolidation Replay — per cortex_reasoning.md.

Callable on demand in this MVP (not yet wired to idle-detection scheduling —
that's a deployment concern, not a core-logic one; see README).

Abstraction needs cross-episode comparison, not single-episode generation:
the `compare_fn` is injected so Cortex's core logic stays model-agnostic and
testable without an API key (tests pass a stub; the agent layer passes a
real Claude-backed comparator).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .graph import MemoryGraph, pattern_node_id
from .store import CortexStore

ComparisonResult = tuple[bool, Optional[str]]  # (same_pattern, canonical_label_if_same)
CompareFn = Callable[[list[str]], ComparisonResult]


@dataclass
class ReplaySummary:
    clusters_examined: int = 0
    clusters_merged: int = 0
    clusters_rejected: int = 0
    promoted_to_instinct: int = 0
    edges_pruned: int = 0


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _greedy_cluster(ids: list[str], embeddings: list[list[float]], threshold: float = 0.82) -> list[list[int]]:
    """Single-link clustering via union-find over a similarity threshold.
    Simple on purpose — avoids pulling in scikit-learn for an MVP-scale store."""
    n = len(ids)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    vecs = [np.array(e) for e in embeddings]
    for i in range(n):
        for j in range(i + 1, n):
            if _cosine_sim(vecs[i], vecs[j]) >= threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return [members for members in clusters.values() if len(members) >= 2]


def run_replay(
    store: CortexStore,
    graph: MemoryGraph,
    compare_fn: CompareFn,
    promote_threshold: int = 3,
    similarity_threshold: float = 0.82,
    lambda_by_topic: Optional[dict[str, float]] = None,
) -> ReplaySummary:
    summary = ReplaySummary()

    raw = store.intuition.get(include=["documents", "embeddings", "metadatas"])
    ids, docs, embeddings, metas = raw["ids"], raw["documents"], raw["embeddings"], raw["metadatas"]

    if len(ids) < 2:
        graph.decay_all(lambda_by_topic)
        summary.edges_pruned = len(graph.prune())
        return summary

    clusters = _greedy_cluster(ids, embeddings, threshold=similarity_threshold)
    summary.clusters_examined = len(clusters)

    for member_idxs in clusters:
        cluster_docs = [docs[i] for i in member_idxs]
        cluster_episode_ids = [ids[i] for i in member_idxs]
        cluster_scope = metas[member_idxs[0]].get("scope", "default")

        same_pattern, canonical_label = compare_fn(cluster_docs)

        if not same_pattern:
            summary.clusters_rejected += 1
            # Down-weight edges that formed from surface similarity alone.
            for a in cluster_episode_ids:
                for b in cluster_episode_ids:
                    if a != b and graph.graph.has_edge(a, b, key="co_retrieved_with"):
                        graph.reinforce(a, b, "co_retrieved_with", success=False)
            continue

        summary.clusters_merged += 1
        canonical_label = canonical_label or cluster_docs[0]
        pattern_node = pattern_node_id(canonical_label)
        graph.add_node(pattern_node, node_type="pattern", text=canonical_label)

        for old_doc, episode_id in zip(cluster_docs, cluster_episode_ids):
            old_pattern_node = pattern_node_id(old_doc)
            if old_pattern_node != pattern_node and old_pattern_node in graph.graph:
                graph.add_edge(old_pattern_node, pattern_node, edge_type="supersedes", weight=1.0)
            graph.add_edge(episode_id, pattern_node, edge_type="instance_of", weight=0.7)
            graph.reinforce(episode_id, pattern_node, "instance_of", success=True)

        if len(cluster_episode_ids) >= promote_threshold:
            store.promote_to_instinct(canonical_label, cluster_episode_ids, scope=cluster_scope)
            summary.promoted_to_instinct += 1

    graph.decay_all(lambda_by_topic)
    summary.edges_pruned = len(graph.prune())
    return summary
