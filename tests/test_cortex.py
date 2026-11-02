"""Standalone Cortex tests — no Claude API key required. Run with:
    python -m tests.test_cortex
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from karmic.cortex import (CortexStore, Episode, EvidenceGate, MemoryGraph, Outcome, retrieve, retrieve_many,
                           run_replay, write_back)
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


def test_outcome_statistics_on_write_back():
    store, _ = fresh_store_and_graph()
    graph_path = store.persist_dir / "graph.db"
    graph = MemoryGraph(persist_path=str(graph_path))
    pattern = "momentum after an earnings beat fades within a week"
    for outcome, value in [(Outcome.SUCCESS, 0.02), (Outcome.FAILURE, -0.03), (Outcome.FAILURE, -0.01),
                           (Outcome.PARTIAL, 0.001)]:
        write_back(store, graph, Episode(problem_summary="hold through earnings?", outcome=outcome,
                                         abstracted_pattern=pattern, outcome_value=value,
                                         outcome_metric="excess_return_5d", decision="acted"))

    # Counts survive a reload, since consumers keep the graph across runs.
    stats = MemoryGraph(persist_path=str(graph_path)).outcome_stats(pattern_node_id(pattern))
    assert (stats["n"], stats["success"], stats["failure"], stats["partial"]) == (4, 1, 2, 1), stats
    text = retrieve(store, graph, "hold through earnings?").to_prompt_text()
    assert "seen 4×: 1 success / 2 failure / 1 partial; excess_return_5d mean -0.00475, median -0.0045" in text, text
    print("PASS: test_outcome_statistics_on_write_back")


def test_evidence_gate_hides_thin_patterns():
    store, graph = fresh_store_and_graph()
    gate = EvidenceGate(min_n=3)
    write_back(store, graph, Episode(problem_summary="sell before the dividend date",
                                     abstracted_pattern="dividend capture rarely beats the price drop",
                                     decision="acted"))
    # A process episode has no decision, so it surfaces however thin its evidence.
    write_back(store, graph, Episode(problem_summary="validator rejected the order size",
                                     abstracted_pattern="order size must respect the lot limit"))

    surfaced = [h["document"] for h in retrieve(store, graph, "dividend date sale", k=5,
                                                evidence_gate=gate).intuition_hits]
    assert surfaced == ["order size must respect the lot limit"], surfaced

    for _ in range(2):
        write_back(store, graph, Episode(problem_summary="sell before the dividend date",
                                         abstracted_pattern="dividend capture rarely beats the price drop",
                                         decision="acted"))
    surfaced = [h["document"] for h in retrieve(store, graph, "dividend date sale", k=5,
                                                evidence_gate=gate).intuition_hits]
    assert "dividend capture rarely beats the price drop" in surfaced, surfaced
    print("PASS: test_evidence_gate_hides_thin_patterns")


def test_gate_counts_patterns_merged_by_replay():
    # Regression: counts were read from each hit's own wording, so differently worded copies
    # of one pattern stayed at n=1 each and never cleared the gate, even after Replay merged them.
    store, graph = fresh_store_and_graph()
    for pattern in ("validate config before use", "validate the config before use",
                    "always validate config before use"):
        write_back(store, graph, Episode(problem_summary="config missing at startup", abstracted_pattern=pattern,
                                         decision="acted"))
    gate = EvidenceGate(min_n=3)
    assert retrieve(store, graph, "config missing", k=5, evidence_gate=gate).intuition_hits == []

    summary = run_replay(store, graph, compare_fn=lambda docs: (True, "validate configuration before use"),
                         similarity_threshold=0.5)
    assert summary.clusters_merged == 1, summary
    hits = retrieve(store, graph, "config missing", k=5, evidence_gate=gate).intuition_hits
    assert len(hits) == 3 and all(h["stats"]["n"] == 3 for h in hits), hits
    print("PASS: test_gate_counts_patterns_merged_by_replay")


def _replay_with_gate(cluster_outcomes: list[Outcome], other_outcomes: list[Outcome]):
    store, graph = fresh_store_and_graph()
    for outcome in cluster_outcomes:
        write_back(store, graph, Episode(problem_summary="buy the post-split dip", outcome=outcome,
                                         abstracted_pattern="post-split dips recover", decision="acted"))
    # The rest of the domain sets the base rate; no pattern, so nothing else clusters.
    for outcome in other_outcomes:
        write_back(store, graph, Episode(problem_summary="unrelated trade", outcome=outcome, decision="acted"))
    summary = run_replay(store, graph, compare_fn=lambda docs: (True, None), promote_threshold=3,
                         evidence_gate=EvidenceGate(min_n=10, alpha=0.05))
    return summary, graph


def test_evidence_gate_promotes_only_significant_patterns():
    # 10/10 against a 50% base rate: p = 1/1024.
    summary, graph = _replay_with_gate([Outcome.SUCCESS] * 10, [Outcome.FAILURE] * 10)
    assert summary.promoted_to_instinct == 1, summary
    assert graph.outcome_stats(pattern_node_id("post-split dips recover"))["n"] == 10

    # The base rate includes the cluster: 11/20 = 0.55, so 6/10 gives p ~ 0.50 and the
    # pattern stays an Intuition.
    summary, _ = _replay_with_gate([Outcome.SUCCESS] * 6 + [Outcome.FAILURE] * 4,
                                   [Outcome.SUCCESS] * 5 + [Outcome.FAILURE] * 5)
    assert summary.clusters_merged == 1 and summary.promoted_to_instinct == 0, summary

    # Enough for Replay's >=3 episodes, but below min_n.
    summary, _ = _replay_with_gate([Outcome.SUCCESS] * 3, [Outcome.FAILURE] * 10)
    assert summary.promoted_to_instinct == 0, summary
    print("PASS: test_evidence_gate_promotes_only_significant_patterns")


def test_declined_decisions_are_labeled():
    store, graph = fresh_store_and_graph()
    write_back(store, graph, Episode(problem_summary="block the release with a flaky test",
                                     outcome=Outcome.FAILURE, decision="declined"))
    text = retrieve(store, graph, "block the release").to_prompt_text()
    assert "- [declined · failure] block the release with a flaky test" in text, text
    print("PASS: test_declined_decisions_are_labeled")


def test_retrieve_many_dedupes_and_groups_by_item():
    store, graph = fresh_store_and_graph()
    disk = Episode(problem_summary="disk usage alert on the database host")
    cert = Episode(problem_summary="TLS certificate expiring on the load balancer")
    write_back(store, graph, disk)
    write_back(store, graph, cert)

    queries = ["database host disk is filling up", "load balancer certificate renewal"]
    ctx = retrieve_many(store, graph, queries, k_each=2, k_total=10)
    assert sorted(h["id"] for h in ctx.recall_hits) == sorted([disk.episode_id, cert.episode_id])
    assert {h["id"]: h["query"] for h in ctx.recall_hits} == {disk.episode_id: queries[0],
                                                               cert.episode_id: queries[1]}
    text = ctx.to_prompt_text()
    first, second = text.split("## Item: load balancer certificate renewal")
    assert "disk usage alert" in first and "TLS certificate" in second, text
    print("PASS: test_retrieve_many_dedupes_and_groups_by_item")


if __name__ == "__main__":
    test_write_and_recall()
    test_intuition_catches_structural_match()
    test_groove_reinforcement()
    test_decay_weakens_unused_edges()
    test_spreading_activation()
    test_replay_merges_confirmed_cluster()
    test_replay_rejects_false_merge()
    test_pattern_ids_are_stable_across_processes()
    test_outcome_statistics_on_write_back()
    test_evidence_gate_hides_thin_patterns()
    test_evidence_gate_promotes_only_significant_patterns()
    test_gate_counts_patterns_merged_by_replay()
    test_declined_decisions_are_labeled()
    test_retrieve_many_dedupes_and_groups_by_item()
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    print("\nAll Cortex tests passed.")
