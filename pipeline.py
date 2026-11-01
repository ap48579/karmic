"""Pipeline orchestrator — ties the 6 agents + Cortex together.

Flow (coding_agent_plan.md "Coding Self-Healing Flow"):
  Cortex retrieve -> Context Builder -> Planner -> Coder -> Tester
    -> (fail) Debugger loop, escalating effort, up to MAX_ATTEMPTS
    -> (pass) Reviewer -> Learning Agent -> Cortex write-back
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agents.pipeline_agents import (
    ContextBuilder, Coder, Debugger, LearningAgent, Planner, Reviewer, Tester, extract_code_block,
)
from cortex import CortexStore, Episode, MemoryGraph, Outcome, retrieve, write_back
from cortex.schema import ActionArcStep, CostActual, HealerAttempt

MAX_DEBUGGER_ATTEMPTS = 4


@dataclass
class PipelineResult:
    episode: Episode
    final_code: str
    workspace: Path
    review: str | None
    dry_run: bool


def predict_complexity(task: str) -> str:
    """Crude MVP heuristic — coding_agent_plan.md's Complexity Score is
    meant to also weigh past episode healer-attempt counts; that signal
    isn't wired up yet, so this stays simple on purpose."""
    word_count = len(task.split())
    if word_count < 12:
        return "simple"
    if word_count < 30:
        return "medium"
    return "complex"


def run_pipeline(task: str, scope: str = "default", workspace_dir: str = "./workspace", cortex_dir: str = "./cortex_data") -> PipelineResult:
    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    store = CortexStore(persist_dir=cortex_dir)
    graph = MemoryGraph(persist_path=str(Path(cortex_dir) / "graph.sqlite3"))

    action_arc: list[ActionArcStep] = []
    cost = CostActual()

    def track(result):
        cost.input_tokens += result.input_tokens
        cost.output_tokens += result.output_tokens

    # 1. Retrieve from Cortex
    retrieved = retrieve(store, graph, task, scope=scope)
    memory_context = retrieved.to_prompt_text()
    action_arc.append(ActionArcStep(action="cortex_retrieve", tool_used="cortex.retrieve", output=memory_context[:300]))

    # 2. Context Builder
    context_summary = ContextBuilder().run(workspace)
    action_arc.append(ActionArcStep(action="context_build", tool_used="ContextBuilder", output=context_summary))

    # 3. Planner
    planner_result = Planner().run(task, memory_context + "\n\n" + context_summary)
    track(planner_result)
    action_arc.append(ActionArcStep(action="plan", tool_used="Planner", output=planner_result.text[:500]))

    # 4. Coder
    coder_result = Coder().run(planner_result.text)
    track(coder_result)
    filename, code = extract_code_block(coder_result.text)
    action_arc.append(ActionArcStep(action="code", tool_used="Coder", output=f"wrote {filename} ({len(code)} chars)"))

    # 5. Tester (mechanical — no LLM call)
    tester = Tester()
    test_result = tester.run(workspace, filename, code)
    action_arc.append(ActionArcStep(
        action="test", tool_used="Tester",
        output=f"passed={test_result['passed']} stderr={test_result['stderr'][:200]}",
    ))

    # 6. Debugger loop (self-paced, escalating effort)
    healer_attempts: list[HealerAttempt] = []
    debugger = Debugger()
    attempt_n = 0
    while not test_result["passed"] and attempt_n < MAX_DEBUGGER_ATTEMPTS:
        candidates, effort_used = debugger.diagnose_and_fix(code, test_result, attempt_n)
        for code in candidates:  # first passing candidate wins; otherwise the last one tried stands
            test_result = tester.run(workspace, filename, code)
            if test_result["passed"]:
                break
        healer_attempts.append(HealerAttempt(
            hypothesis=f"attempt {attempt_n + 1} at effort={effort_used}, {len(candidates)} candidate(s)",
            fix=f"{len(code)} char rewrite",
            outcome="passed" if test_result["passed"] else f"failed: {test_result['stderr'][:150]}",
        ))
        attempt_n += 1

    outcome = Outcome.SUCCESS if test_result["passed"] else (Outcome.PARTIAL if attempt_n > 0 else Outcome.FAILURE)

    # 7. Reviewer (only if tests passed)
    review_text = None
    if test_result["passed"]:
        review_result = Reviewer().run(task, code)
        track(review_result)
        review_text = review_result.text
        action_arc.append(ActionArcStep(action="review", tool_used="Reviewer", output=review_text[:300]))

    # 8. Learning Agent -> Cortex write-back
    learning = LearningAgent()
    summary_text = "\n".join(f"{s.action}: {s.output[:150]}" for s in action_arc)
    learned = learning.run(task, outcome.value, summary_text)

    episode = Episode(
        domain="coding",
        scope=scope,
        problem_summary=task,
        initial_plan=planner_result.text,
        action_arc=action_arc,
        healer_attempts=healer_attempts,
        outcome=outcome,
        final_solution=code if test_result["passed"] else None,
        abstracted_pattern=learned["abstracted_pattern"],
        failure_summary=learned["failure_summary"],
        distilled_learnings=learned["distilled_learnings"],
        cost_actual=cost,
        predicted_complexity=predict_complexity(task),
        healer_attempt_count=len(healer_attempts),
    )

    write_back(store, graph, episode, retrieved=retrieved)

    return PipelineResult(episode=episode, final_code=code, workspace=workspace, review=review_text, dry_run=planner_result.dry_run)
