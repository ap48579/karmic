"""A5 Extractor / Classifier: document -> labeled fields, each grounded in
an exact quote from the document.

Lineage: google/langextract -- every extraction maps back to a character
interval of its source, so a value can be checked against the text
instead of trusted. Here that's enforced, not just reported: a quote that
isn't verbatim in the document (whitespace- and case-insensitive) is a
problem the model must fix, and a closed label set makes classification
mechanically checkable.

Fields are declared by the consumer:

    extractor({
        "severity": Field(kind="label", labels=("sev1", "sev2", "sev3")),
        "affected_users": Field(kind="number", required=False),
        "root_cause": Field(kind="text", required=False),
    })

Call.inputs:
  document: str
  instructions: str  -- optional extra guidance for this document type
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .runtime import ArchetypeSpec, Call, Result
from .validators import DEFAULT_NUMBERS, NumberPolicy, Problem, find_span, unsupported_numbers

NOT_FOUND = "NOT_FOUND"
# A number field's value is the number alone, optionally with a currency sign / unit word.
# Found live: a model put the whole quoted sentence in `value`, which a "number appears in
# quote" check alone happily accepts.
_NUMBER_VALUE = re.compile(r"^[-+]?\$?\d[\d,]*(?:\.\d+)?\s*(?:%|[kKmMbB]|thousand|million|billion|bn|mm)?$")


@dataclass(frozen=True)
class Field:
    kind: str = "text"  # text | number | label
    labels: tuple[str, ...] = ()  # closed set, kind="label" only
    required: bool = True
    multiple: bool = False  # may appear more than once (e.g. several counterparties)
    description: str = ""


CONTRACT = (
    "Output rules (checked automatically against the document after you answer):\n"
    "- One extraction per value found. `quote` is copied VERBATIM from the document (a short span, "
    "not a paraphrase) and must contain the evidence for `value`.\n"
    "- Label fields: `value` must be one of the listed labels exactly.\n"
    "- Number fields: the number in `value` must appear in `quote`.\n"
    f"- If a required field is not in the document, return value \"{NOT_FOUND}\" with an empty quote. "
    "Never guess."
)


def schema(fields: dict[str, Field]) -> dict:
    return {
        "type": "object",
        "properties": {"extractions": {"type": "array", "items": {
            "type": "object",
            "properties": {"field": {"type": "string", "enum": list(fields)},
                           "value": {"type": "string"}, "quote": {"type": "string"}},
            "required": ["field", "value", "quote"],
        }}},
        "required": ["extractions"],
    }


def _render_fields(fields: dict[str, Field]) -> str:
    lines = []
    for name, f in fields.items():
        desc = f" -- {f.description}" if f.description else ""
        kind = f"one of: {', '.join(f.labels)}" if f.kind == "label" else f.kind
        req = "required" if f.required else "optional"
        many = ", may repeat" if f.multiple else ""
        lines.append(f"- {name} ({kind}; {req}{many}){desc}")
    return "\n".join(lines)


def _check(fields: dict[str, Field], numbers: NumberPolicy):
    def check(raw: dict, call: Call) -> tuple[dict, list[Problem]]:
        doc = call.inputs["document"]
        kept, problems, counts = [], [], {name: 0 for name in fields}
        for n, e in enumerate(raw.get("extractions") or []):
            where = f"extractions[{n}]"
            name, value, quote = e["field"], (e.get("value") or "").strip(), e.get("quote") or ""
            f = fields[name]
            if value == NOT_FOUND:  # explicit absence: fine, nothing to ground
                kept.append({"field": name, "value": None, "quote": "", "span": None})
                continue
            if f.kind == "label" and value not in f.labels:
                problems.append(Problem("label_not_allowed", f"{where} {name}={value!r} is not an allowed label. "
                                        f"Use one of: {', '.join(f.labels)}.", where))
                continue
            span = find_span(quote, doc)
            if span is None:
                problems.append(Problem("not_grounded", f'{where} quote "{quote[:160]}" for {name} does not appear '
                                        "verbatim in the document. Copy the exact text.", where))
                continue
            if f.kind == "number":
                if not _NUMBER_VALUE.match(value):
                    problems.append(Problem("not_a_number", f"{where} {name}={value[:80]!r}: value must be just the number "
                                            '(e.g. "250"), with the supporting text in quote.', where))
                    continue
                if unsupported_numbers(value, quote, numbers):
                    problems.append(Problem("numbers_unverified", f"{where} {name}={value!r} is not a number stated in its "
                                            f'quote "{quote[:160]}".', where))
                    continue
            if counts[name] and not f.multiple:
                problems.append(Problem("duplicate_field", f"{where} {name} was extracted more than once; it takes a single value.", where))
                continue
            counts[name] += 1
            kept.append({"field": name, "value": value, "quote": doc[span[0]:span[1]], "span": list(span)})
        given = {e["field"] for e in kept}
        missing = [name for name, f in fields.items() if f.required and name not in given]
        if missing:
            problems.append(Problem("missing_field", f"No extraction for required field(s) {', '.join(missing)}. "
                                    f"Extract them, or return value \"{NOT_FOUND}\" if the document doesn't contain them.",
                                    "extractions"))
        values: dict = {}
        for e in kept:
            if fields[e["field"]].multiple:
                values.setdefault(e["field"], []).append(e["value"])
            else:
                values[e["field"]] = e["value"]
        return {"extractions": kept, "values": values}, problems
    return check


def _metrics(result: Result, call: Call) -> dict:
    kept = [e for e in result.output.get("extractions", []) if e["value"] is not None]
    rejected = [p for p in result.validation.problems
                if p.kind in ("not_grounded", "label_not_allowed", "numbers_unverified", "not_a_number")]
    emitted = len(kept) + len(rejected)
    return {"grounded_rate": len(kept) / emitted if emitted else 1.0, "values": result.output.get("values", {}),
            "heal_attempts": result.attempts - 1}


def extractor(fields: dict[str, Field], numbers: NumberPolicy = DEFAULT_NUMBERS, max_heal_attempts: int = 2) -> ArchetypeSpec:
    for name, f in fields.items():
        if f.kind == "label" and not f.labels:
            raise ValueError(f"label field {name!r} needs a closed label set")

    def render(call: Call) -> str:
        extra = f"\n\n{call.inputs['instructions']}" if call.inputs.get("instructions") else ""
        return f"Fields to extract:\n{_render_fields(fields)}{extra}\n\nDocument:\n<<<\n{call.inputs['document']}\n>>>"

    def default(call: Call, reason: str) -> dict:
        return {"extractions": [], "values": {}, "error": reason[:200]}

    return ArchetypeSpec(
        name="extractor", output_schema=schema(fields), contract=CONTRACT, check=_check(fields, numbers),
        render=render, default_output=default, max_heal_attempts=max_heal_attempts, metrics=_metrics,
    )
