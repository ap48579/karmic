"""A6 Reflector: finished episode -> the Episode fields Cortex learns from
(abstracted_pattern / failure_summary / distilled_learnings).

Verbal reflection once an outcome is known, optionally *contrasting* a
failed episode with a successful one on a similar task (pass
`compare_with`). Generalizes agents/research/reflection.py's Reflection
step, which stays as the research pipeline's own instance.

The two checks are what keep a lesson from becoming memory poison:
  - groundedness: every number in a lesson appears in the episode
    record (a lesson quoting a figure that never happened is invented);
  - abstraction: abstracted_pattern names no entity from the denylist
    (service names, customers, people). It feeds Intuition, which matches
    *structure* across tasks -- "checkout-service broke after the cache
    deploy" can't transfer, "a cache layer added without a warm-up step
    caused a cold-start spike" can.

Call.inputs:
  episode: str            -- the finished episode as text, numbers included
  outcome: str            -- success | partial | failure (already decided by the consumer)
  compare_with: str       -- optional: a contrasting episode (ExpeL)
  entity_denylist: [str]  -- names that must not appear in abstracted_pattern
"""

from __future__ import annotations

import re

from .runtime import ArchetypeSpec, Call, Result
from .validators import DEFAULT_NUMBERS, NumberPolicy, Problem, unsupported_numbers

MAX_LEARNINGS = 3

SCHEMA = {
    "type": "object",
    "properties": {
        "abstracted_pattern": {"type": "string"},
        "failure_summary": {"type": "string"},
        "distilled_learnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["abstracted_pattern", "failure_summary", "distilled_learnings"],
}

CONTRACT = (
    "Output rules (checked automatically against the episode after you answer):\n"
    "- abstracted_pattern: the general approach or situation, in one sentence, with NO specific names "
    "(no names of systems, organizations, people) -- it must transfer to other cases. Empty string if nothing general was learned.\n"
    "- failure_summary: if the outcome was not a success, what was wrongly assumed. Empty string on success.\n"
    f"- distilled_learnings: at most {MAX_LEARNINGS} short, actionable lessons.\n"
    "- Every number you write must appear in the episode record. Do not invent figures."
)


def _render(call: Call) -> str:
    i = call.inputs
    parts = [f"Outcome: {i['outcome']}", f"Episode:\n{i['episode']}"]
    if i.get("compare_with"):
        parts.append("A contrasting episode on a similar task -- what did it do differently?\n" + i["compare_with"])
    return "\n\n".join(parts)


def _default(call: Call, reason: str) -> dict:
    return {"abstracted_pattern": "", "failure_summary": "", "distilled_learnings": []}


def _check(numbers: NumberPolicy):
    def check(raw: dict, call: Call) -> tuple[dict, list[Problem]]:
        source = call.inputs["episode"] + "\n" + (call.inputs.get("compare_with") or "")
        problems = []
        pattern = raw.get("abstracted_pattern", "").strip()
        failure = raw.get("failure_summary", "").strip()
        learnings = [l.strip() for l in raw.get("distilled_learnings") or [] if l.strip()]
        if len(learnings) > MAX_LEARNINGS:
            problems.append(Problem("too_many_learnings", f"{len(learnings)} learnings; keep the best {MAX_LEARNINGS}.",
                                    "distilled_learnings"))
            learnings = learnings[:MAX_LEARNINGS]
        for where, text in [("abstracted_pattern", pattern), ("failure_summary", failure),
                            *((f"distilled_learnings[{n}]", l) for n, l in enumerate(learnings))]:
            bad = unsupported_numbers(text, source, numbers)
            if bad:
                problems.append(Problem("numbers_ungrounded", f"{where} states {', '.join(f'{b:g}' for b in bad)}, which "
                                        "does not appear in the episode. Use only figures from the record, or none.", where))
        named = [e for e in call.inputs.get("entity_denylist") or []
                 if re.search(rf"(?<!\w){re.escape(e)}(?!\w)", pattern, re.IGNORECASE)]
        if named:
            problems.append(Problem("entity_in_pattern", f"abstracted_pattern names {', '.join(named)}. Describe the "
                                    "situation without specific names so it applies to other cases.", "abstracted_pattern"))
        return {"abstracted_pattern": pattern, "failure_summary": failure, "distilled_learnings": learnings}, problems
    return check


def _metrics(result: Result, call: Call) -> dict:
    o = result.output
    texts = [t for t in [o["abstracted_pattern"], o["failure_summary"], *o["distilled_learnings"]] if t]
    ungrounded = {p.path for p in result.validation.problems if p.kind == "numbers_ungrounded"}
    return {
        "groundedness": (len(texts) - len(ungrounded)) / len(texts) if texts else 1.0,
        "abstracted": not any(p.kind == "entity_in_pattern" for p in result.validation.problems),
        "heal_attempts": result.attempts - 1,
    }


def to_episode_fields(output: dict) -> dict:
    """Reflector output -> keyword arguments for cortex.schema.Episode."""
    return {"abstracted_pattern": output["abstracted_pattern"] or None,
            "failure_summary": output["failure_summary"] or None,
            "distilled_learnings": list(output["distilled_learnings"])}


def reflector(numbers: NumberPolicy = DEFAULT_NUMBERS, max_heal_attempts: int = 2) -> ArchetypeSpec:
    return ArchetypeSpec(
        name="reflector", output_schema=SCHEMA, contract=CONTRACT, check=_check(numbers), render=_render,
        default_output=_default, max_heal_attempts=max_heal_attempts, metrics=_metrics,
    )
