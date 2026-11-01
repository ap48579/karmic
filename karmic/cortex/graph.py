"""Memory Graph & Groove Mechanics — per cortex_reasoning.md.

Strength lives on edges, not nodes. Edge weight = "groove depth": strengthened
by Hebbian co-activation + success, decayed by disuse (topic-scoped).

Persistence is SQLite-backed (nodes/edges tables), not a single JSON blob.
Every mutation (add_node/add_edge/reinforce/decay_all/prune) commits
immediately, so a crash mid-mutation loses at most that one change instead
of the whole graph — the previous design re-serialized the entire graph to
one JSON file on every save, which had no such guarantee. SQLite needs no
server and ships with Python's standard library, so this adds no new infra
dependency. The in-memory NetworkX graph is still what every algorithm here
(reinforce, decay, spreading activation) actually operates on; SQLite is
purely the persistence layer underneath it.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Optional

import networkx as nx

from .schema import text_key

DEFAULT_LAMBDA = 0.05  # decay rate per day, used when no topic-specific rate is set
SECONDS_PER_DAY = 86400.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    attrs TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL NOT NULL,
    topic TEXT NOT NULL,
    last_reinforced REAL NOT NULL,
    lambda_topic REAL,
    PRIMARY KEY (source, target, edge_type)
);
"""


def pattern_node_id(pattern_text: str) -> str:
    return f"pattern::{text_key(pattern_text)}"


class MemoryGraph:
    """NetworkX-backed groove graph. Edge types:
    instance_of, co_retrieved_with, supersedes, contradicts.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.graph = nx.MultiDiGraph()
        self.persist_path = persist_path
        self._conn: Optional[sqlite3.Connection] = None
        if persist_path:
            Path(persist_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(persist_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            self.load()

    # ---- node / edge mutation ----

    def add_node(self, node_id: str, node_type: str, **attrs) -> None:
        if node_id in self.graph:
            self.graph.nodes[node_id].update(attrs)
        else:
            self.graph.add_node(node_id, node_type=node_type, **attrs)
        self._persist_node(node_id)

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        weight: float = 0.5,
        topic: str = "default",
    ) -> None:
        self.graph.add_edge(
            source,
            target,
            key=edge_type,
            edge_type=edge_type,
            weight=weight,
            topic=topic,
            last_reinforced=time.time(),
        )
        self._persist_edge(source, target, edge_type)

    def reinforce(
        self,
        source: str,
        target: str,
        edge_type: str,
        success: bool,
        learning_rate: float = 0.3,
    ) -> float:
        """Hebbian update rule from cortex_reasoning.md:
        retrieve + use + success -> strengthen.
        retrieve + ignored, or used + still failed -> flat or weaken.
        Returns the new weight.
        """
        if not self.graph.has_edge(source, target, key=edge_type):
            self.add_edge(source, target, edge_type, weight=0.5)

        data = self.graph[source][target][edge_type]
        current = data["weight"]
        if success:
            new_weight = current + learning_rate * (1.0 - current)
        else:
            new_weight = current - (learning_rate * 0.5) * current
        new_weight = max(0.01, min(1.0, new_weight))

        data["weight"] = new_weight
        data["last_reinforced"] = time.time()
        self._persist_edge(source, target, edge_type)
        return new_weight

    # ---- decay ----

    def effective_weight(self, source: str, target: str, edge_type: str, now: Optional[float] = None) -> float:
        """Live decayed weight without mutating stored state — used at query time
        so freshness reflects elapsed time even between periodic decay passes."""
        if not self.graph.has_edge(source, target, key=edge_type):
            return 0.0
        data = self.graph[source][target][edge_type]
        now = now or time.time()
        days_elapsed = (now - data["last_reinforced"]) / SECONDS_PER_DAY
        lam = data.get("lambda_topic", DEFAULT_LAMBDA)
        return data["weight"] * math.exp(-lam * days_elapsed)

    def decay_all(self, lambda_by_topic: Optional[dict[str, float]] = None, now: Optional[float] = None) -> int:
        """Mutating decay pass — called periodically (piggybacks on Replay's
        decay+prune step). Returns count of edges below the prune threshold."""
        lambda_by_topic = lambda_by_topic or {}
        now = now or time.time()
        prune_candidates = 0
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            lam = lambda_by_topic.get(data.get("topic", "default"), DEFAULT_LAMBDA)
            days_elapsed = (now - data["last_reinforced"]) / SECONDS_PER_DAY
            data["weight"] = data["weight"] * math.exp(-lam * days_elapsed)
            data["last_reinforced"] = now
            data["lambda_topic"] = lam
            if data["weight"] < 0.05:
                prune_candidates += 1
        self._persist_all_edges()
        return prune_candidates

    def prune(self, threshold: float = 0.05) -> list[tuple[str, str, str]]:
        """Archive (caller decides what 'archive' means) edges below threshold.
        Returns the removed (source, target, edge_type) triples."""
        to_remove = [
            (u, v, k)
            for u, v, k, d in self.graph.edges(keys=True, data=True)
            if d["weight"] < threshold
        ]
        for u, v, k in to_remove:
            self.graph.remove_edge(u, v, key=k)
        if self._conn and to_remove:
            self._conn.executemany(
                "DELETE FROM edges WHERE source=? AND target=? AND edge_type=?", to_remove
            )
            self._conn.commit()
        return to_remove

    # ---- retrieval: spreading activation ----

    def spreading_activation(
        self,
        seed_nodes: list[str],
        hops: int = 2,
        decay_per_hop: float = 0.5,
    ) -> dict[str, float]:
        """Walk outward from seed nodes along weighted edges. Activation
        propagates multiplicatively (edge weight * decay_per_hop per hop),
        so high-groove paths carry activation further than shallow ones."""
        activation: dict[str, float] = {n: 1.0 for n in seed_nodes if n in self.graph}
        frontier = dict(activation)

        for _ in range(hops):
            next_frontier: dict[str, float] = {}
            for node, node_activation in frontier.items():
                if node not in self.graph:
                    continue
                for _, target, edge_type in self.graph.out_edges(node, keys=True):
                    w = self.effective_weight(node, target, edge_type)
                    propagated = node_activation * w * decay_per_hop
                    if propagated < 0.01:
                        continue
                    incoming = activation.get(target, 0.0)
                    activation[target] = max(incoming, propagated)
                    next_frontier[target] = max(next_frontier.get(target, 0.0), propagated)
            frontier = next_frontier
            if not frontier:
                break

        for seed in seed_nodes:
            activation.pop(seed, None)
        return dict(sorted(activation.items(), key=lambda kv: kv[1], reverse=True))

    # ---- persistence (SQLite) ----

    def _persist_node(self, node_id: str) -> None:
        if not self._conn:
            return
        data = dict(self.graph.nodes[node_id])
        node_type = data.pop("node_type", "unknown")
        self._conn.execute(
            "INSERT OR REPLACE INTO nodes (node_id, node_type, attrs) VALUES (?, ?, ?)",
            (node_id, node_type, json.dumps(data)),
        )
        self._conn.commit()

    def _persist_edge(self, source: str, target: str, edge_type: str) -> None:
        if not self._conn:
            return
        data = self.graph[source][target][edge_type]
        self._conn.execute(
            "INSERT OR REPLACE INTO edges "
            "(source, target, edge_type, weight, topic, last_reinforced, lambda_topic) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (source, target, edge_type, data["weight"], data["topic"],
             data["last_reinforced"], data.get("lambda_topic")),
        )
        self._conn.commit()

    def _persist_all_edges(self) -> None:
        """Batched version of _persist_edge for decay_all's full-sweep update
        -- one transaction for every edge instead of one commit per edge."""
        if not self._conn:
            return
        rows = [
            (u, v, key, d["weight"], d["topic"], d["last_reinforced"], d.get("lambda_topic"))
            for u, v, key, d in self.graph.edges(keys=True, data=True)
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO edges "
            "(source, target, edge_type, weight, topic, last_reinforced, lambda_topic) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def load(self) -> None:
        """Rebuilds the in-memory NetworkX graph from SQLite."""
        conn = self._conn
        if conn is None:
            return
        self.graph = nx.MultiDiGraph()
        for node_id, node_type, attrs_json in conn.execute("SELECT node_id, node_type, attrs FROM nodes"):
            self.graph.add_node(node_id, node_type=node_type, **json.loads(attrs_json))
        for source, target, edge_type, weight, topic, last_reinforced, lambda_topic in conn.execute(
            "SELECT source, target, edge_type, weight, topic, last_reinforced, lambda_topic FROM edges"
        ):
            kwargs = {"weight": weight, "topic": topic, "last_reinforced": last_reinforced}
            if lambda_topic is not None:
                kwargs["lambda_topic"] = lambda_topic
            self.graph.add_edge(source, target, key=edge_type, edge_type=edge_type, **kwargs)

    def stats(self) -> dict:
        weights = [d["weight"] for _, _, d in self.graph.edges(data=True)]
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "mean_weight": sum(weights) / len(weights) if weights else 0.0,
            "max_weight": max(weights) if weights else 0.0,
            "min_weight": min(weights) if weights else 0.0,
        }
