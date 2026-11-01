"""A3 Adversarial Debater: evidence + analyst digests + a side -> the
strongest honest case for that side.

Sides argue over already-validated findings for a configurable number of
rounds, each answering the other's last case, with claim validation on
every point. The check this adds over A2: a side whose every citation is
the *other* side's evidence isn't arguing its case, it's restating the
opponent's.

Call.inputs:
  side: str                     -- "for", "against", "prosecution", ...
  subject: str
  digest: str                   -- prior findings to argue over (e.g. A2 outputs, see digest())
  opponent_case: str | None     -- set on rebuttal rounds
  opposing_evidence: list[str]  -- optional: ids that support the other side
"""

from __future__ import annotations

from typing import Callable

from .llm import JsonLLM
from .runtime import ArchetypeSpec, Call, Result, RoleConfig, run
from .validators import CLAIM_SCHEMA, DEFAULT_NUMBERS, NumberPolicy, Problem, citation_stats, validate_claims

SCHEMA = {
    "type": "object",
    "properties": {
        "case": {"type": "string"},
        "points": {"type": "array", "items": CLAIM_SCHEMA},
        "strongest_point": CLAIM_SCHEMA,
    },
    "required": ["case", "points", "strongest_point"],
}

CONTRACT = (
    "Output rules (checked automatically against the evidence after you answer):\n"
    "- Use only the findings and evidence given; cite evidence ids for every point.\n"
    "- Copy numbers exactly.\n"
    "- Argue from evidence that supports YOUR side. Citing only evidence for the other side is not a case.\n"
    "- If your side's case is genuinely weak, say so in `case` instead of overstating it.\n"
    "- strongest_point is your single best point, with its citations."
)


def _render(call: Call) -> str:
    i = call.inputs
    parts = [f"Your side: {i['side']}"]
    if i.get("subject"):
        parts.append(f"Subject: {i['subject']}")
    if i.get("digest"):
        parts.append(f"Findings so far:\n{i['digest']}")
    parts.append(f"Evidence:\n{call.evidence_block()}")
    if i.get("opponent_case"):
        parts.append(f"The other side argued:\n{i['opponent_case']}\n\nRespond to their strongest point, then restate your case.")
    return "\n\n".join(parts)


def _default(call: Call, reason: str) -> dict:
    return {"case": f"(no case: {reason[:200]})", "points": [], "strongest_point": None}


def _check(numbers: NumberPolicy):
    def check(raw: dict, call: Call) -> tuple[dict, list[Problem]]:
        allowed = call.allowed
        points, problems = validate_claims(raw.get("points"), allowed, call.evidence, "points", numbers)
        strongest, p2 = validate_claims([raw.get("strongest_point") or {}], allowed, call.evidence, "strongest_point", numbers)
        problems += p2
        opposing = set(call.inputs.get("opposing_evidence") or [])
        cited = {i for c in points + strongest for i in c["evidence_ids"]}
        if opposing and cited and cited <= opposing:
            problems.append(Problem(
                "one_sided_citation",
                f"Every id you cited ({', '.join(sorted(cited))}) is evidence for the other side. "
                f"Build the {call.inputs['side']} case from evidence that supports it, or say the case is weak.",
                "points"))
        return {"case": raw.get("case", ""), "points": points,
                "strongest_point": strongest[0] if strongest else None}, problems
    return check


def _metrics(result: Result, call: Call) -> dict:
    o = result.output
    claims = o.get("points", []) + ([o["strongest_point"]] if o.get("strongest_point") else [])
    return {**citation_stats(claims, result.validation.problems), "heal_attempts": result.attempts - 1,
            "cited_ids": sorted({i for c in claims for i in c["evidence_ids"]}),
            "one_sided": any(p.kind == "one_sided_citation" for p in result.validation.problems)}


def debater(numbers: NumberPolicy = DEFAULT_NUMBERS, max_heal_attempts: int = 2) -> ArchetypeSpec:
    return ArchetypeSpec(
        name="debater", output_schema=SCHEMA, contract=CONTRACT, check=_check(numbers), render=_render,
        default_output=_default, max_heal_attempts=max_heal_attempts, metrics=_metrics,
    )


def digest(findings: dict[str, dict]) -> tuple[str, set[str]]:
    """A2 outputs keyed by analyst name -> (digest text, every id they cited). The cited set is
    the natural `allowed_evidence` for the debate: debaters argue over validated findings."""
    lines, ids = [], set()
    for name, o in findings.items():
        lines.append(f"{name.upper()}: {o.get('stance')} (strength {float(o.get('strength') or 0):.2f}). {o.get('reasoning', '')}")
        for kind, tag in (("key_points", "+"), ("risks", "-")):
            for c in o.get(kind, []):
                ids.update(c["evidence_ids"])
                lines.append(f"  {tag} {c['claim']} [{', '.join(c['evidence_ids'])}]")
        if o.get("data_gaps"):
            lines.append(f"  data gaps: {'; '.join(o['data_gaps'])}")
    return "\n".join(lines), ids


def debate(spec: ArchetypeSpec, roles: dict[str, RoleConfig], make_call: Callable[[str, str | None], Call],
           llm: JsonLLM, rounds: int = 1) -> dict[str, list[Result]]:
    """Two (or more) sides, `rounds` rounds. Round 1 is independent opening cases; each later
    round every side sees the previous round's case of the side before it. make_call(side,
    opponent_case) builds the Call, so the consumer keeps control of evidence and scoping."""
    sides = list(roles)
    out: dict[str, list[Result]] = {s: [] for s in sides}
    for r in range(rounds):
        for k, side in enumerate(sides):
            opponent = sides[k - 1]
            opponent_case = out[opponent][-1].output["case"] if r > 0 and out[opponent] else None
            out[side].append(run(spec, roles[side], make_call(side, opponent_case), llm))
    return out
