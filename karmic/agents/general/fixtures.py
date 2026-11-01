"""Frozen fixtures -> a ScoreLike, so a consumer's role is RSI-gateable the
moment it has fixtures (docs/09_agent_archetypes_spec.md S7).

A fixture is a saved Call (inputs + evidence + allowed ids; never memory,
so the gate is stationary) plus the mechanical facts a correct answer
must have. Expectations are deliberately few kinds, all checkable without
a model:

    {"path": "stance", "equals": "negative"}               # output or metrics, dotted path
    {"path": "actions.release-4.2", "in": ["rollback"]}
    {"path": "values.severity", "equals": "sev1"}
    {"must_cite": ["ev3"]}                                  # planted-fact recall
    {"must_not_cite": ["ev9"]}

The scalar keeps the shape of the existing benchmarks:

    scalar = 100*fixture_pass_rate + 20*faithfulness + 20*numbers_ok
             - 5*avg_heal_attempts - 30*override_rate

faithfulness is citation validity (A2/A3/A4) or grounded rate (A5);
numbers_ok is number-verified rate (A2-A4) or groundedness (A6).
override_rate is 0 for everything but A4. Slow, noisy downstream
outcomes are never an expectation kind -- only facts checkable from the
fixture itself (S8).

Run the gate the way WRITING_YOUR_OWN_BENCHMARK.md requires: as a
subprocess inside the candidate worktree, so the prompt files being
scored are the candidate's (templates/benchmark_template.py does that).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .evidence import EvidenceIndex
from .llm import JsonLLM
from .runtime import ArchetypeSpec, Call, Result, RoleConfig, run


@dataclass
class Fixture:
    fixture_id: str
    role: str
    inputs: dict
    evidence: dict = field(default_factory=lambda: {"prefix": "ev", "items": []})  # EvidenceIndex.to_dict()
    allowed_evidence: list[str] | None = None
    expect: list[dict] = field(default_factory=list)
    note: str = ""  # why this case is in the set -- helps whoever labels the next one

    def to_call(self) -> Call:
        return Call(inputs=self.inputs, evidence=EvidenceIndex.from_dict(self.evidence),
                    allowed_evidence=set(self.allowed_evidence) if self.allowed_evidence is not None else None)

    @classmethod
    def from_call(cls, fixture_id: str, role: str, call: Call, expect: list[dict], note: str = "") -> "Fixture":
        return cls(fixture_id, role, call.inputs, call.evidence.to_dict(),
                   sorted(call.allowed_evidence) if call.allowed_evidence is not None else None, expect, note)


def save_fixtures(path: str | Path, fixtures: list[Fixture]) -> None:
    with open(path, "w") as f:
        for fx in fixtures:
            f.write(json.dumps(asdict(fx), default=sorted) + "\n")  # default=sorted: sets in inputs


def load_fixtures(path: str | Path) -> list[Fixture]:
    return [Fixture(**json.loads(line)) for line in open(path) if line.strip()]


def _lookup(path: str, result: Result):
    for root in (result.output, result.metrics):
        node, ok = root, True
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            else:
                ok = False
                break
        if ok:
            return node, True
    return None, False


def check_expectation(exp: dict, result: Result) -> tuple[bool, str]:
    cited = set(result.metrics.get("cited_ids", []))
    if "must_cite" in exp:
        missing = set(exp["must_cite"]) - cited
        return not missing, f"did not cite {sorted(missing)}" if missing else ""
    if "must_not_cite" in exp:
        bad = set(exp["must_not_cite"]) & cited
        return not bad, f"cited {sorted(bad)}" if bad else ""
    value, found = _lookup(exp["path"], result)
    if not found:
        return False, f"{exp['path']} not in output"
    if "equals" in exp:
        return value == exp["equals"], f"{exp['path']}={value!r}, expected {exp['equals']!r}"
    if "in" in exp:
        return value in exp["in"], f"{exp['path']}={value!r}, expected one of {exp['in']!r}"
    raise ValueError(f"unknown expectation {exp}")


@dataclass
class FixtureRun:
    fixture_id: str
    passed: bool
    failures: list[str]
    metrics: dict


@dataclass
class FixtureScore:
    runs: list[FixtureRun]
    pass_rate: float
    faithfulness: float
    numbers_ok: float
    avg_heal_attempts: float
    override_rate: float
    scalar_score: float

    def to_prompt_text(self) -> str:
        lines = [f"scalar_score={self.scalar_score:.1f} pass_rate={self.pass_rate:.0%} faithfulness={self.faithfulness:.2f} "
                 f"numbers_ok={self.numbers_ok:.2f} avg_heal_attempts={self.avg_heal_attempts:.2f} "
                 f"override_rate={self.override_rate:.2f}"]
        for r in self.runs:
            lines.append(f"  - {r.fixture_id}: {'pass' if r.passed else 'FAIL ' + '; '.join(r.failures)}")
        return "\n".join(lines)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def score_fixtures(spec: ArchetypeSpec, roles: dict[str, RoleConfig], fixtures: list[Fixture], llm: JsonLLM) -> FixtureScore:
    runs = []
    for fx in fixtures:
        result = run(spec, roles[fx.role], fx.to_call(), llm)
        checks = [check_expectation(e, result) for e in fx.expect]
        failures = [msg for ok, msg in checks if not ok]
        if result.validation.failed:
            failures.insert(0, "call failed: no schema-valid output")
        runs.append(FixtureRun(fx.fixture_id, not failures, failures, result.metrics))
    m = [r.metrics for r in runs]
    pass_rate = _mean([float(r.passed) for r in runs])
    faith = _mean([x.get("citation_validity", x.get("grounded_rate", 1.0)) for x in m])
    nums = _mean([x.get("number_verified_rate", x.get("groundedness", 1.0)) for x in m])
    heal = _mean([x.get("heal_attempts", 0) for x in m])
    override = _mean([x.get("override_rate", 0.0) for x in m])
    scalar = 100 * pass_rate + 20 * faith + 20 * nums - 5 * heal - 30 * override
    return FixtureScore(runs, pass_rate, faith, nums, heal, override, scalar)
