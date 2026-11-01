"""Fixed, deterministic checks on model output -- policy, not prompt
(docs/09_agent_archetypes_spec.md S4). Same role as
agents/research/policies.py, extended from run-level caps to per-claim
checks. Never RSI-editable: this module is outside every HarnessSpec
component tree by construction, because the benchmark that gates an edit
is built from these same checks.

Nothing here knows about any domain. Domain knowledge (which numerals in
a claim are names rather than figures, how strict a match must be) comes
in through NumberPolicy, supplied by the consumer.

Every problem carries the text the self-healing loop feeds back to the
model, written to be actionable ("ev12 contains 3, 2024 -- not 4"),
since that specificity is the difference between a heal attempt and a
blind retry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .evidence import EvidenceIndex

_NUM = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class NumberPolicy:
    """How strictly a number in a claim must match a number in its cited evidence
    (docs/09_agent_archetypes_spec.md S15 Q1 leaves per-archetype tuning open)."""

    rel_tol: float = 0.015
    abs_tol: float = 0.06
    # Small integers ("3 retries", "2 rounds") are usually counts the model derived, not
    # figures it could misquote -- checking them causes more false flags than it catches.
    small_int_max: int = 10
    # Unit changes models commonly make: a fraction quoted as a percent, a dollar figure
    # quoted in thousands/millions/billions.
    scales: tuple[float, ...] = (100, 0.01, 1e3, 1e6, 1e9)
    # Regexes for numerals that are part of a name, not a figure ("HTTP 500", "Python 3.12",
    # "ISO 8601"). Which names occur is domain knowledge, so the default is empty and the
    # consumer supplies its own. Without them a name-numeral is flagged as a misquote the
    # model can't fix; the runtime's no-progress stop bounds what that costs. Claim side only.
    ignore: tuple[str, ...] = ()


DEFAULT_NUMBERS = NumberPolicy()


@dataclass
class Problem:
    kind: str  # schema | llm_error | no_citation | unknown_citation | numbers_unverified | ...
    detail: str  # fed back to the model verbatim on a heal attempt
    path: str = ""  # where in the output, e.g. "key_points[2]"
    heal: bool = True  # False = recorded as a metric only (A4 overrides), never sent back

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "path": self.path, "heal": self.heal}


@dataclass
class ValidationReport:
    problems: list[Problem] = field(default_factory=list)
    failed: bool = False  # no attempt ever produced schema-valid output

    @property
    def heal_problems(self) -> list[Problem]:
        return [p for p in self.problems if p.heal]

    def to_dict(self) -> dict:
        return {"problems": [p.to_dict() for p in self.problems], "failed": self.failed}


# --- numbers -------------------------------------------------------------------------------

def numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text or ""):
        try:
            out.append(float(m.replace("$", "").replace(",", "").replace("+", "")))
        except ValueError:
            pass
    return out


def _close(a: float, b: float, policy: NumberPolicy) -> bool:
    return abs(a - b) <= max(policy.rel_tol * abs(a), policy.abs_tol)


def number_supported(value: float, pool: list[float], policy: NumberPolicy = DEFAULT_NUMBERS) -> bool:
    if value.is_integer() and abs(value) <= policy.small_int_max:
        return True
    for v in pool:
        if _close(v, value, policy):
            return True
        for scale in policy.scales:
            if _close(v * scale, value, policy) or _close(v / scale, value, policy):
                return True
    return False


def strip_name_numerals(text: str, policy: NumberPolicy = DEFAULT_NUMBERS) -> str:
    for pattern in policy.ignore:
        text = re.sub(pattern, " ", text)
    return text


def unsupported_numbers(text: str, source_text: str, policy: NumberPolicy = DEFAULT_NUMBERS) -> list[float]:
    pool = numbers(source_text)
    return [n for n in numbers(strip_name_numerals(text, policy)) if not number_supported(n, pool, policy)]


def _fmt(n: float) -> str:
    return str(int(n)) if n.is_integer() else f"{n:g}"


# --- citations -------------------------------------------------------------------------------

def validate_claims(
    claims: list[dict] | None, allowed_ids: set[str], evidence: EvidenceIndex, path: str = "claims",
    policy: NumberPolicy = DEFAULT_NUMBERS,
) -> tuple[list[dict], list[Problem]]:
    """Returns (kept claims, problems). A claim is {"claim": str, "evidence_ids": [str]}.

    Claims citing nothing, or citing ids the role wasn't given, are dropped. Claims with
    numbers not found in their cited evidence are kept but flagged `numbers_unverified`:
    dropping them loses too much signal, flagging lets downstream roles and benchmarks
    count them.
    """
    kept, problems = [], []
    for n, c in enumerate(claims or []):
        if not isinstance(c, dict):
            continue
        claim = (c.get("claim") or "").strip()
        ids = [i for i in (c.get("evidence_ids") or []) if isinstance(i, str)]
        where = f"{path}[{n}]"
        if not claim:
            continue
        if not ids:
            problems.append(Problem("no_citation", f'{where} "{claim[:120]}" cites no evidence. '
                                    "Cite the evidence ids it is based on, or remove it.", where))
            continue
        unknown = [i for i in ids if i not in allowed_ids or i not in evidence.items]
        if unknown:
            problems.append(Problem("unknown_citation", f'{where} "{claim[:120]}" cites {", ".join(unknown)}, '
                                    "which you were not given. Cite only ids from the evidence list.", where))
            continue
        cited_text = " ".join(evidence.text(i) for i in ids)
        bad = unsupported_numbers(claim, cited_text, policy)
        entry = {"claim": claim, "evidence_ids": ids}
        if bad:
            entry["numbers_unverified"] = bad
            found = sorted({_fmt(x) for x in numbers(" ".join(evidence.items[i].label for i in ids))})[:12]
            problems.append(Problem(
                "numbers_unverified",
                f'{where} "{claim[:120]}" states {", ".join(_fmt(b) for b in bad)}, which does not appear in '
                f'{", ".join(ids)} (those items contain: {", ".join(found) or "no numbers"}). '
                "Copy the number exactly from the evidence, or cite the item it comes from.", where))
        kept.append(entry)
    return kept, problems


def citation_stats(claims: list[dict], problems: list[Problem]) -> dict:
    """For benchmarks: fraction of emitted claims that survived, and that were number-verified."""
    dropped = sum(1 for p in problems if p.kind in ("no_citation", "unknown_citation"))
    emitted = len(claims) + dropped
    unverified = sum(1 for c in claims if c.get("numbers_unverified"))
    return {
        "emitted": emitted,
        "citation_validity": len(claims) / emitted if emitted else 1.0,
        "number_verified_rate": (len(claims) - unverified) / len(claims) if claims else 1.0,
    }


# --- grounding (A5) ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def find_span(quote: str, document: str) -> tuple[int, int] | None:
    """Character interval of `quote` in `document`, whitespace- and case-insensitive.
    Exact containment only: a paraphrase is not grounding."""
    q = _normalize(quote)
    if not q:
        return None
    # Build the normalized document alongside a map back to original offsets.
    norm_chars, offsets, prev_space = [], [], False
    for i, ch in enumerate(document):
        if ch.isspace():
            if prev_space or not norm_chars:
                continue
            norm_chars.append(" ")
            prev_space = True
        else:
            norm_chars.append(ch.lower())
            prev_space = False
        offsets.append(i)
    at = "".join(norm_chars).find(q)
    if at < 0:
        return None
    return offsets[at], offsets[at + len(q) - 1] + 1


# --- schema -----------------------------------------------------------------------------------

_JSON_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def validate_schema(value, schema: dict, path: str = "$") -> list[Problem]:
    """The JSON-schema subset the archetype schemas use: type, properties, required, items,
    enum, minimum/maximum. Ollama's `format` and Anthropic's tool_use usually enforce
    this already; checking again keeps the runtime correct for any llm callable."""
    problems: list[Problem] = []
    t = schema.get("type")
    if t == "number":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif t == "integer":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif t in _JSON_TYPES:
        ok = isinstance(value, _JSON_TYPES[t])
    else:
        ok = True
    if not ok:
        return [Problem("schema", f"{path} must be of type {t}, got {type(value).__name__}.", path)]
    if "enum" in schema and value not in schema["enum"]:
        problems.append(Problem("schema", f"{path} must be one of {schema['enum']}, got {value!r}.", path))
    if t in ("number", "integer"):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(Problem("schema", f"{path} must be >= {schema['minimum']}, got {value}.", path))
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(Problem("schema", f"{path} must be <= {schema['maximum']}, got {value}.", path))
    if t == "object":
        for key in schema.get("required", []):
            if key not in value:
                problems.append(Problem("schema", f"{path} is missing required field {key!r}.", path))
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                problems += validate_schema(value[key], sub, f"{path}.{key}")
    if t == "array" and "items" in schema:
        for i, v in enumerate(value):
            problems += validate_schema(v, schema["items"], f"{path}[{i}]")
    return problems


CLAIM_SCHEMA = {
    "type": "object",
    "properties": {"claim": {"type": "string"}, "evidence_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["claim", "evidence_ids"],
}
