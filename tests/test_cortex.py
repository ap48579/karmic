"""Standalone Cortex tests — no Claude API key required. Run with:
    python -m tests.test_cortex
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from karmic.cortex import CortexStore, Episode, MemoryGraph, Outcome, retrieve, run_replay, write_back
from karmic.cortex.graph import pattern_node_id

TEST_ROOT = Path("./test_cortex_data")
_counter = {"n": 0}


def fresh_store_and_graph():
    # chromadb's PersistentClient caches state per-path in-process, so reusing
    # one path across tests after rmtree leaves it in a bad state — give each
    # test its own subdirectory instead.
    _counter["n"] += 1
    test_dir = TEST_ROOT / f"run_{_counter['n']}"
    store = CortexStore(persist_dir=str(test_dir))
    graph = MemoryGraph()
    return store, graph


def test_write_and_recall():
    store, graph = fresh_store_and_graph()
    ep = Episode(
        problem_summary="Fix a null pointer exception when config is missing",
        outcome=Outcome.SUCCESS,
        abstracted_pattern="config-driven refactor, not a logic bug",
        distilled_learnings=["Always validate config is loaded before use"],
    )
    write_back(store, graph, ep)

    hits = store.query_recall("null pointer when config missing")
    assert len(hits) == 1, f"expected 1 recall hit, got {len(hits)}"
    assert hits[0]["id"] == ep.episode_id
    print("PASS: test_write_and_recall")


def test_intuition_catches_structural_match():
    store, graph = fresh_store_and_graph()
    ep1 = Episode(
        problem_summary="Crash when YAML settings file is absent on startup",
        outcome=Outcome.SUCCESS,
        abstracted_pattern="config-driven refactor: validate config exists before use",
    )
    write_back(store, graph, ep1)

    # Surface-dissimilar wording, same underlying pattern.
    ctx = retrieve(store, graph, "App throws an error if the .env file isn't present")
    assert len(ctx.intuition_hits) >= 1, "Intuition should surface the structurally similar pattern"
    print("PASS: test_intuition_catches_structural_match")


def test_groove_reinforcement():
    store, graph = fresh_store_and_graph()
    ep1 = Episode(problem_summary="task A", outcome=Outcome.SUCCESS)
    ep2 = Episode(problem_summary="task B", outcome=Outcome.SUCCESS)
    write_back(store, graph, ep1)
    write_back(store, graph, ep2)

    w0 = graph.effective_weight(ep1.episode_id, ep2.episode_id, "co_retrieved_with")
    assert w0 == 0.0, "no edge should exist yet"

    graph.reinforce(ep1.episode_id, ep2.episode_id, "co_retrieved_with", success=True)
    w1 = graph.effective_weight(ep1.episode_id, ep2.episode_id, "co_retrieved_with")
    graph.reinforce(ep1.episode_id, ep2.episode_id, "co_retrieved_with", success=True)
    w2 = graph.effective_weight(ep1.episode_id, ep2.episode_id, "co_retrieved_with")

    assert w1 > 0, "edge should strengthen on success"
    assert w2 > w1, "repeated success should deepen the groove further"
    print(f"PASS: test_groove_reinforcement (weight 0 -> {w1:.3f} -> {w2:.3f})")


def test_decay_weakens_unused_edges():
    store, graph = fresh_store_and_graph()
    graph.add_node("a", "episode")
    graph.add_node("b", "episode")
    graph.add_edge("a", "b", "co_retrieved_with", weight=0.8)

    # Simulate 30 days of no reinforcement.
    import time
    graph.graph["a"]["b"]["co_retrieved_with"]["last_reinforced"] = time.time() - 30 * 86400

    decayed = graph.effective_weight("a", "b", "co_retrieved_with")
    assert decayed < 0.8, "weight should decay after 30 days unused"
    print(f"PASS: test_decay_weakens_unused_edges (0.8 -> {decayed:.3f} after 30d)")


def test_spreading_activation():
    store, graph = fresh_store_and_graph()
    for n in ["seed", "near", "far"]:
        graph.add_node(n, "episode")
    graph.add_edge("seed", "near", "co_retrieved_with", weight=0.9)
    graph.add_edge("near", "far", "co_retrieved_with", weight=0.9)

    activated = graph.spreading_activation(["seed"], hops=2)
    assert "near" in activated and "far" in activated
    assert activated["near"] > activated["far"], "activation should decay with hop distance"
    print(f"PASS: test_spreading_activation ({activated})")


def test_replay_merges_confirmed_cluster():
    store, graph = fresh_store_and_graph()
    for i in range(3):
        ep = Episode(
            problem_summary=f"task variant {i}: missing config crashes the app",
            outcome=Outcome.SUCCESS,
            abstracted_pattern="validate configuration is loaded before use",
        )
        write_back(store, graph, ep)

    def stub_compare(docs: list[str]):
        return True, "Always validate configuration exists before dereferencing it"

    summary = run_replay(store, graph, compare_fn=stub_compare, promote_threshold=3)
    assert summary.clusters_merged >= 1, "should have merged the 3 near-identical patterns"
    assert summary.promoted_to_instinct >= 1, "3 confirmed instances should promote to Instinct"

    instinct_entries = store.all_instinct()
    assert len(instinct_entries) >= 1
    # Replay must find the pattern node write-back created, or the old pattern is never superseded.
    old, canonical = (pattern_node_id("validate configuration is loaded before use"),
                      pattern_node_id("Always validate configuration exists before dereferencing it"))
    assert graph.graph.has_edge(old, canonical, key="supersedes")
    print(f"PASS: test_replay_merges_confirmed_cluster ({summary})")


def test_pattern_ids_are_stable_across_processes():
    # Regression: ids were built from hash(), which Python salts per process, so a pattern
    # written in one run was invisible to Replay or write-back in the next.
    code = "from karmic.cortex.graph import pattern_node_id; print(pattern_node_id('validate config before use'))"
    ids = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                       cwd=str(Path(__file__).parent.parent), env={**os.environ, "PYTHONHASHSEED": seed}).stdout.strip()
        for seed in ("1", "2")
    }
    assert ids == {pattern_node_id("validate config before use")}, ids
    print("PASS: test_pattern_ids_are_stable_across_processes")


def test_replay_rejects_false_merge():
    store, graph = fresh_store_and_graph()
    ep1 = Episode(problem_summary="x", outcome=Outcome.SUCCESS, abstracted_pattern="off-by-one loop bound error")
    ep2 = Episode(problem_summary="y", outcome=Outcome.SUCCESS, abstracted_pattern="off-by-zero index bound error")
    write_back(store, graph, ep1)
    write_back(store, graph, ep2)

    def stub_reject(docs: list[str]):
        return False, None  # model determines these are NOT actually the same

    summary = run_replay(store, graph, compare_fn=stub_reject, similarity_threshold=0.5)
    if summary.clusters_examined > 0:
        assert summary.clusters_rejected == summary.clusters_examined
        assert summary.promoted_to_instinct == 0
    print(f"PASS: test_replay_rejects_false_merge ({summary})")


if __name__ == "__main__":
    test_write_and_recall()
    test_intuition_catches_structural_match()
    test_groove_reinforcement()
    test_decay_weakens_unused_edges()
    test_spreading_activation()
    test_replay_merges_confirmed_cluster()
    test_replay_rejects_false_merge()
    test_pattern_ids_are_stable_across_processes()
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    print("\nAll Cortex tests passed.")
