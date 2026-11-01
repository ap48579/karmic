"""The one entry point every archetype shares: run(spec, role, call, llm).

    memory -> model -> schema check -> archetype validator
      clean                        -> return
      problems, budget left        -> re-ask in the same conversation with the
                                      validator's own complaints (S5)
      no progress                  -> stop early: same problems as the last attempt,
                                      so another re-ask would only repeat it
      budget spent                 -> best attempt (fewest problems)
      never schema-valid           -> spec.default_output, validation.failed=True

It never raises on model failure. A consumer pipeline making 50 calls a
run should degrade per role, not crash.

Split of ownership (S3):
  - ArchetypeSpec (this package): output schema, the contract text that
    states what the validator enforces, the validator, the default. Fixed.
  - RoleConfig (the consumer): persona and judgment guidance, loaded from
    the consumer's own prompt file. RSI-editable on the consumer's
    lineage branch; the runtime just reads it.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .evidence import EvidenceIndex
from .llm import JsonLLM, LLMError
from .validators import Problem, ValidationReport, validate_schema

MAX_PROBLEMS_FED_BACK = 8

CheckFn = Callable[[dict, "Call"], tuple[dict, list[Problem]]]


@dataclass(frozen=True)
class ArchetypeSpec:
    name: str
    output_schema: dict
    contract: str  # appended to the role prompt: the rules the validator enforces, stated up front
    check: CheckFn  # (schema-valid raw output, call) -> (cleaned output, problems). Fixed.
    render: Callable[["Call"], str]  # call -> user message (inputs + evidence block)
    default_output: Callable[["Call", str], dict]  # (call, reason) -> safe no-op output
    max_heal_attempts: int = 2
    metrics: Callable[["Result", "Call"], dict] | None = None  # benchmark-facing numbers


@dataclass
class RoleConfig:
    role: str  # e.g. "latency_analyst"
    system_prompt: str
    temperature: float = 0.2
    extra: dict = field(default_factory=dict)  # tunables the consumer exposes to RSI

    @classmethod
    def from_file(cls, role: str, path: str | Path, **kw) -> "RoleConfig":
        return cls(role=role, system_prompt=Path(path).read_text().strip(), **kw)


@dataclass
class Call:
    inputs: dict
    evidence: EvidenceIndex = field(default_factory=EvidenceIndex)
    allowed_evidence: set[str] | None = None  # None = every id in `evidence`
    memory: Any = None  # cortex RetrievalContext (anything with to_prompt_text()), a str, or None

    @property
    def allowed(self) -> set[str]:
        return set(self.evidence.items) if self.allowed_evidence is None else set(self.allowed_evidence)

    def evidence_block(self) -> str:
        return self.evidence.render(self.allowed) or "(no evidence)"


@dataclass
class Attempt:
    n: int
    raw: dict | None
    output: dict | None  # None = schema-invalid or transport failure
    problems: list[Problem]
    seconds: float

    @property
    def heal_problems(self) -> list[Problem]:
        return [p for p in self.problems if p.heal]


@dataclass
class Result:
    output: dict
    validation: ValidationReport
    attempts: int
    history: list[Attempt] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    @property
    def recurring_problem_kinds(self) -> list[str]:
        """Problem kinds the model kept making across heal attempts (seen on >= 2), most persistent first."""
        seen = Counter(kind for a in self.history for kind in {p.kind for p in a.heal_problems})
        return [k for k, n in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0])) if n >= 2]

    def caution_candidates(self, role: str) -> list[str]:
        """Candidate Cortex Caution text (S5): a mistake that survived validator feedback."""
        out = []
        for kind in self.recurring_problem_kinds:
            example = next(p.detail for a in reversed(self.history) for p in a.heal_problems if p.kind == kind)
            out.append(f"{role}: repeated {kind} despite feedback, e.g. {example[:240]}")
        return out

    def to_episode(self, role: str, domain: str, scope: str = "default", problem_summary: str | None = None):
        """A Cortex process Episode for this call -- its outcome is known immediately (S9.2)."""
        from ...cortex.schema import ActionArcStep, CostActual, Episode, HealerAttempt, Outcome

        arc = [ActionArcStep(action=f"attempt {a.n}", tool_used=role,
                             output=f"{len(a.problems)} problems: {', '.join(sorted({p.kind for p in a.problems})) or 'none'}")
               for a in self.history]
        heals = [HealerAttempt(hypothesis="; ".join(p.detail for p in prev.heal_problems)[:500],
                               fix="re-asked with validator feedback",
                               outcome=f"{len(cur.heal_problems)} problems remaining" if cur.output is not None else "schema-invalid")
                 for prev, cur in zip(self.history, self.history[1:])]
        if self.validation.failed:
            outcome = Outcome.FAILURE
        elif self.validation.heal_problems:
            outcome = Outcome.PARTIAL
        else:
            outcome = Outcome.SUCCESS
        cautions = self.caution_candidates(role)
        return Episode(
            domain=domain, scope=scope, problem_summary=problem_summary or f"{role} call",
            action_arc=arc, healer_attempts=heals, outcome=outcome,
            failure_summary=cautions[0] if cautions else None,
            cost_actual=CostActual(input_tokens=self.meta.get("input_tokens", 0),
                                   output_tokens=self.meta.get("output_tokens", 0)),
            healer_attempt_count=len(heals),
        )


def heal_message(problems: list[Problem]) -> str:
    lines = [f"- {p.detail}" for p in problems[:MAX_PROBLEMS_FED_BACK]]
    if len(problems) > MAX_PROBLEMS_FED_BACK:
        lines.append(f"- ...and {len(problems) - MAX_PROBLEMS_FED_BACK} more of the same kind.")
    return ("Your answer was checked against the evidence and has these problems:\n" + "\n".join(lines) +
            "\nReturn the complete corrected JSON answer. Keep everything that was right; fix or remove what was flagged.")


def _stalled(history: list[Attempt]) -> bool:
    if len(history) < 2 or history[-1].output is None or history[-2].output is None:
        return False
    def key(a: Attempt) -> list[tuple[str, str, str]]:
        return sorted((p.kind, p.path, p.detail) for p in a.heal_problems)
    return key(history[-1]) == key(history[-2])


def _memory_text(memory: Any) -> str:
    if memory is None:
        return ""
    text = memory.to_prompt_text() if hasattr(memory, "to_prompt_text") else str(memory)
    return text.strip()


def run(spec: ArchetypeSpec, role: RoleConfig, call: Call, llm: JsonLLM) -> Result:
    system = role.system_prompt.rstrip() + "\n\n" + spec.contract
    user = spec.render(call)
    memory = _memory_text(call.memory)
    if memory:
        user = ("Lessons from past episodes (advisory -- the evidence below wins on any conflict):\n"
                f"{memory}\n\n{user}")
    messages = [{"role": "user", "content": user}]
    history: list[Attempt] = []
    meta: dict = {"seconds": 0.0, "input_tokens": 0, "output_tokens": 0}

    for n in range(1, spec.max_heal_attempts + 2):
        started = time.monotonic()
        try:
            raw, call_meta = llm(system, messages, spec.output_schema, role.temperature)
        except LLMError as exc:
            history.append(Attempt(n, None, None, [Problem("llm_error", str(exc), heal=False)],
                                   round(time.monotonic() - started, 1)))
            continue  # transport failure: retry the same conversation, nothing to heal from
        meta["model"] = call_meta.get("model")
        for k in ("input_tokens", "output_tokens"):
            meta[k] += call_meta.get(k, 0) or 0

        problems = validate_schema(raw, spec.output_schema)
        output = None
        if not problems:
            output, problems = spec.check(raw, call)
        history.append(Attempt(n, raw, output, problems, round(time.monotonic() - started, 1)))
        heal = [p for p in problems if p.heal]
        if output is not None and not heal:
            break
        if _stalled(history):
            meta["stalled"] = True
            break
        messages += [{"role": "assistant", "content": json.dumps(raw)}, {"role": "user", "content": heal_message(heal)}]

    meta["seconds"] = round(sum(a.seconds for a in history), 1)
    valid = [a for a in history if a.output is not None]
    if valid:
        # Fewest heal problems wins; on a tie the later attempt, which had more feedback.
        best = min(reversed(valid), key=lambda a: len(a.heal_problems))
        result = Result(best.output, ValidationReport(best.problems), len(history), history, meta)
    else:
        reason = history[-1].problems[0].detail if history and history[-1].problems else "no attempts"
        result = Result(spec.default_output(call, reason), ValidationReport(history[-1].problems if history else [], failed=True),
                        len(history), history, meta)
    if spec.metrics:
        result.metrics = spec.metrics(result, call)
    return result
