#!/usr/bin/env python3
"""CLI entrypoint.

    python cli.py "write a function that returns the nth fibonacci number"
    python cli.py --research "In what year was the Eiffel Tower completed?"
    python cli.py --stats                       # show Cortex memory stats
    python cli.py --self-improve --harness research   # one RSI cycle on a harness (see rsi/)

--research needs QWEN_API_KEY (+ QWEN_BASE_URL for a self-hosted/Ollama
endpoint, + QWEN_MODEL) set; see agents/research/qwen_config.py.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", help="Task description (coding), or question with --research")
    parser.add_argument("--research", action="store_true",
                         help="Run the deep-research pipeline instead of the coding pipeline")
    parser.add_argument("--scope", default="default")
    parser.add_argument("--workspace-dir", default="./workspace")
    parser.add_argument("--cortex-dir", default="./cortex_data")
    parser.add_argument("--stats", action="store_true", help="Print Cortex memory stats and exit")
    parser.add_argument("--self-improve", action="store_true",
                         help="Run self-improvement cycle(s) against a harness's lineage branch and exit")
    parser.add_argument("--harness", default="coding", choices=["coding", "research"],
                         help="Which harness to self-improve via the unified rsi/ engine -- "
                              "'coding' is Claude-backed, 'research' is Qwen/Ollama-backed, see "
                              "docs/05_rsi_harness2_design_spec.md")
    parser.add_argument("--cycles", type=int, default=1, help="Number of cycles with --self-improve")
    args = parser.parse_args()

    if args.stats:
        from cortex import CortexStore
        store = CortexStore(persist_dir=args.cortex_dir)
        print(store.stats())
        return

    if args.self_improve:
        repo_root = Path(__file__).parent
        from rsi import HARNESSES, LLM_CALL_FACTORIES, run_cycle

        harness = HARNESSES[args.harness]
        env_key = "ANTHROPIC_API_KEY" if args.harness == "coding" else "QWEN_API_KEY"
        if not os.environ.get(env_key):
            print(f"WARNING: {env_key} is not set — the proposer will get a dry-run "
                  "placeholder response and every cycle will be skipped.\n", file=sys.stderr)
        llm_call = LLM_CALL_FACTORIES[args.harness]()

        for _ in range(args.cycles):
            result = asyncio.run(run_cycle(repo_root, harness, llm_call, cortex_dir=args.cortex_dir))
            print(f"\n{'='*60}")
            print(f"[{harness.name}] Generation {result.generation}: {result.status.upper()}")
            print(f"Summary: {result.summary}")
            if result.baseline_score is not None:
                print(f"Score: baseline={result.baseline_score:.1f} candidate={result.candidate_score:.1f}")
            if result.applied_edits:
                print(f"Edited: {', '.join(result.applied_edits)}")
            if result.detail:
                print(f"Detail: {result.detail[:500]}")
            print(f"{'='*60}\n")
        return

    if not args.task:
        parser.error("task is required unless --stats or --self-improve is passed")

    if args.research:
        if not os.environ.get("QWEN_API_KEY"):
            print("WARNING: QWEN_API_KEY is not set — running in DRY RUN mode. "
                  "The research harness will return placeholder text instead of real "
                  "completions.\n", file=sys.stderr)
        from research_pipeline import run_research_pipeline
        result = run_research_pipeline(args.task, scope=args.scope,
                                        workspace_dir=args.workspace_dir, cortex_dir=args.cortex_dir)
        print(f"\n{'='*60}")
        print(f"Outcome: {result.episode.outcome.value}")
        print(f"Revisions: {result.episode.healer_attempt_count}")
        print(f"Cost: {result.episode.cost_actual.input_tokens} in / {result.episode.cost_actual.output_tokens} out tokens")
        print(f"{'='*60}\n")
        print("Final report:\n" + result.final_report)
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY is not set — running in DRY RUN mode. "
              "Agents will return placeholder text instead of real completions.\n", file=sys.stderr)

    result = run_pipeline(args.task, scope=args.scope, workspace_dir=args.workspace_dir, cortex_dir=args.cortex_dir)

    print(f"\n{'='*60}")
    print(f"Outcome: {result.episode.outcome.value}")
    print(f"Healer attempts: {result.episode.healer_attempt_count}")
    print(f"Cost: {result.episode.cost_actual.input_tokens} in / {result.episode.cost_actual.output_tokens} out tokens")
    print(f"Workspace: {result.workspace}")
    print(f"{'='*60}\n")
    if result.review:
        print("Reviewer verdict:\n" + result.review)


if __name__ == "__main__":
    main()
