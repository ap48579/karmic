"""A2 Evidence Analyst: numbered evidence set -> JSON judgment, every claim
citing evidence ids.

Meant to be narrow: one analyst per evidence slice, each blind to the
others. A claim is only as good as its citation, so citation validity is
measured per claim, not per answer.

Call.inputs:
  subject: str   -- what is being analysed ("checkout-service, release 4.2")
  question: str  -- optional; defaults to "Return your analysis as JSON."
"""

from __future__ import annotations

from .runtime import ArchetypeSpec, Call, Result
from .validators import CLAIM_SCHEMA, DEFAULT_NUMBERS, NumberPolicy, Problem, citation_stats, validate_claims

DEFAULT_STANCES = ("positive", "negative", "neutral")

CONTRACT = (
    "Output rules (checked automatically against the evidence after you answer):\n"
    "- Use ONLY the numbered evidence. No outside knowledge about the subject.\n"
    "- Every key_point and risk cites the evidence ids (like \"ev12\") it is based on. A claim you "
    "cannot cite, leave out.\n"
    "- Copy numbers exactly as they appear in the evidence. Do not compute new figures.\n"
    "- strength is 0 to 1: how strongly the evidence supports your stance. Thin or contradictory "
    "evidence means low strength or a neutral stance -- that is a good answer, not a failure.\n"
    "- List anything you would need but don't have in data_gaps."
)


def schema(stances: tuple[str, ...] = DEFAULT_STANCES) -> dict:
    return {
        "type": "object",
        "properties": {
            "stance": {"type": "string", "enum": list(stances)},
            "strength": {"type": "number"},
            "key_points": {"type": "array", "items": CLAIM_SCHEMA},
            "risks": {"type": "array", "items": CLAIM_SCHEMA},
            "data_gaps": {"type": "array", "items": {"type": "string"}},
            "reasoning": {"type": "string"},
        },
        "required": ["stance", "strength", "key_points", "risks", "data_gaps", "reasoning"],
    }


def _render(call: Call) -> str:
    subject = call.inputs.get("subject", "")
    question = call.inputs.get("question") or "Return your analysis as JSON."
    head = f"Subject: {subject}\n\n" if subject else ""
    return f"{head}Evidence:\n{call.evidence_block()}\n\n{question}"


def _default(neutral: str):
    def default(call: Call, reason: str) -> dict:
        return {"stance": neutral, "strength": 0.0, "key_points": [], "risks": [],
                "data_gaps": [f"analysis failed: {reason[:200]}"], "reasoning": ""}
    return default


def _check(numbers: NumberPolicy):
    def check(raw: dict, call: Call) -> tuple[dict, list[Problem]]:
        allowed = call.allowed
        key_points, p1 = validate_claims(raw.get("key_points"), allowed, call.evidence, "key_points", numbers)
        risks, p2 = validate_claims(raw.get("risks"), allowed, call.evidence, "risks", numbers)
        out = {
            "stance": raw["stance"],
            "strength": max(0.0, min(1.0, float(raw.get("strength") or 0))),
            "key_points": key_points,
            "risks": risks,
            "data_gaps": raw.get("data_gaps") or [],
            "reasoning": raw.get("reasoning", ""),
        }
        return out, p1 + p2
    return check


def _metrics(result: Result, call: Call) -> dict:
    out = result.output
    claims = out.get("key_points", []) + out.get("risks", [])
    problems = result.validation.problems
    cited = {i for c in claims for i in c["evidence_ids"]}
    return {**citation_stats(claims, problems), "heal_attempts": result.attempts - 1,
            "cited_ids": sorted(cited), "stance": out.get("stance")}


def evidence_analyst(stances: tuple[str, ...] = DEFAULT_STANCES, neutral: str = "neutral",
                     numbers: NumberPolicy = DEFAULT_NUMBERS, max_heal_attempts: int = 2) -> ArchetypeSpec:
    if neutral not in stances:
        raise ValueError(f"neutral stance {neutral!r} must be one of {stances}")
    return ArchetypeSpec(
        name="evidence_analyst", output_schema=schema(stances), contract=CONTRACT,
        check=_check(numbers), render=_render, default_output=_default(neutral),
        max_heal_attempts=max_heal_attempts, metrics=_metrics,
    )
