"""Deep Research pipeline orchestrator — the research-domain sibling of
pipeline.py, sharing the exact same Cortex (memory/trace) wiring.

Flow (docs/03_integration_spec.md, docs/04_e2e_research_agent_spec.md):
  Cortex retrieve -> fold memory into the query -> vendored GPTResearcher
  (conduct_research + write_report) -> Grader/Reviser loop (ours, not
  vendored) -> Reflection -> Cortex write-back.

MAX_REVISIONS is this pipeline's analogue of pipeline.py's
MAX_DEBUGGER_ATTEMPTS -- a plain tunable constant, the kind of low-risk
lever the RSI engine (Spec 5) is meant to experiment with.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from karmic.agents.research import configure_qwen
from karmic.agents.research.policies import DEFAULT_POLICIES, Decision, PolicyState, evaluate
from karmic.agents.research.reflection import grade, reflect, revise
from karmic.agents.research.session import EpisodeSession
from karmic.cortex import CortexStore, Episode, MemoryGraph, Outcome, retrieve, write_back
from karmic.cortex.schema import ActionArcStep, CostActual, HealerAttempt

MAX_REVISIONS = 3


@dataclass
class ResearchResult:
    episode: Episode
    final_report: str
    dry_run: bool


async def _run_research(question: str, scope: str, workspace_dir: str, cortex_dir: str) -> ResearchResult:
    provider = configure_qwen()
    dry_run = provider == "dry_run"

    store = CortexStore(persist_dir=cortex_dir)
    graph = MemoryGraph(persist_path=str(Path(cortex_dir) / "graph.sqlite3"))
    session = EpisodeSession.start(workspace_dir)

    cost = CostActual()
    action_arc: list[ActionArcStep] = []

    def track(in_tok: int, out_tok: int) -> None:
        cost.input_tokens += in_tok
        cost.output_tokens += out_tok

    # 1. Retrieve from Cortex
    retrieved = retrieve(store, graph, question, scope=scope)
    memory_context = retrieved.to_prompt_text()
    action_arc.append(ActionArcStep(action="cortex_retrieve", tool_used="cortex.retrieve", output=memory_context[:300]))
    session.checkpoint("cortex_retrieve", {"memory_context_len": len(memory_context)})

    # 2. Fold memory into the query -- GPTResearcher's `context=` kwarg gets
    # overwritten by conduct_research(), confirmed by reading agent.py, so
    # the only injection point that reliably reaches every internal stage
    # (sub-question generation, persona-building) is the query text itself.
    research_query = question
    if "no prior memory" not in memory_context:
        research_query = (
            f"{question}\n\nRelevant context from prior research episodes -- do not "
            f"repeat work already covered, and treat Caution entries as known dead "
            f"ends:\n{memory_context}"
        )
    note = session.check_steer()
    if note:
        research_query += f"\n\nOperator note: {note}"

    if dry_run:
        draft = f"[DRY RUN — no QWEN_API_KEY set. Research would run for: {question}]"
        action_arc.append(ActionArcStep(action="research", tool_used="GPTResearcher", output="dry_run"))
    else:
        from gpt_researcher import GPTResearcher

        researcher = GPTResearcher(query=research_query, report_type="research_report")
        await researcher.conduct_research()
        session.checkpoint("conduct_research", {})
        draft = await researcher.write_report()
        session.checkpoint("write_report", {"draft_len": len(draft)})
        action_arc.append(ActionArcStep(action="research", tool_used="GPTResearcher", output=draft[:300]))

    # 3. Grader/Reviser loop (ours, not vendored -- GPT Researcher has none)
    policy_state = PolicyState()
    healer_attempts: list[HealerAttempt] = []
    revision_n = 0
    if dry_run:
        verdict_value = "revise"
    else:
        verdict, in_tok, out_tok = await grade(question, draft)
        track(in_tok, out_tok)
        policy_state.total_input_tokens, policy_state.total_output_tokens = cost.input_tokens, cost.output_tokens
        while verdict.verdict != "pass" and revision_n < MAX_REVISIONS:
            policy_result = evaluate(DEFAULT_POLICIES, policy_state)
            if policy_result.decision in (Decision.DENY, Decision.ASK):
                session.checkpoint("policy_stop", {"reason": policy_result.reason})
                break
            note = session.check_steer()
            mistakes = verdict.mistakes + ([note] if note else [])
            draft, in_tok, out_tok = await revise(draft, mistakes)
            track(in_tok, out_tok)
            healer_attempts.append(HealerAttempt(
                hypothesis="; ".join(verdict.mistakes) or "unspecified",
                fix=f"revision {revision_n + 1}", outcome=verdict.verdict,
            ))
            session.checkpoint("revise", {"revision": revision_n + 1})
            verdict, in_tok, out_tok = await grade(question, draft)
            track(in_tok, out_tok)
            revision_n += 1
            policy_state.revision_count = revision_n
            policy_state.total_input_tokens, policy_state.total_output_tokens = cost.input_tokens, cost.output_tokens
        verdict_value = verdict.verdict
    action_arc.append(ActionArcStep(action="grade", tool_used="Grader",
                                     output=f"final_verdict={verdict_value} revisions={revision_n}"))

    outcome = Outcome.SUCCESS if verdict_value == "pass" else (
        Outcome.PARTIAL if revision_n > 0 else Outcome.FAILURE
    )

    # 4. Reflection -> Cortex write-back
    summary_text = "\n".join(f"{s.action}: {s.output[:150]}" for s in action_arc)
    reflection, in_tok, out_tok = await reflect(question, outcome.value, summary_text)
    track(in_tok, out_tok)

    episode = Episode(
        domain="research",
        scope=scope,
        problem_summary=question,
        action_arc=action_arc,
        healer_attempts=healer_attempts,
        outcome=outcome,
        final_solution=draft if outcome != Outcome.FAILURE else None,
        abstracted_pattern=reflection.abstracted_pattern,
        failure_summary=reflection.failure_summary,
        distilled_learnings=reflection.distilled_learnings,
        cost_actual=cost,
        healer_attempt_count=revision_n,
    )

    write_back(store, graph, episode, retrieved=retrieved)
    session.checkpoint("write_back", {"episode_id": episode.episode_id})

    return ResearchResult(episode=episode, final_report=draft, dry_run=dry_run)


def run_research_pipeline(
    question: str, scope: str = "default", workspace_dir: str = "./workspace", cortex_dir: str = "./cortex_data",
) -> ResearchResult:
    return asyncio.run(_run_research(question, scope, workspace_dir, cortex_dir))
