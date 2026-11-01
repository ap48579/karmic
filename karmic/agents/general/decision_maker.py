"""A4 Decision Maker: candidates + state + constraints -> one action per
candidate, under hard constraints enforced in code.

Limits are computed in code; the model may only choose within them:
allowed actions per item, a citation for any non-default action, the
consumer's own caps via constrain(), and a no-op default for anything the
model skipped. The prompt states the rules, code enforces them.

Overrides are recorded, not hidden: every time code had to correct the
model is an entry in output["overrides"] and counts toward
metrics["override_rate"]. They are *not* fed back to the model
(Problem.heal=False) -- healing them away would hide exactly the number
the benchmark needs (S7), and code has already made the decision safe.
What IS healed: decisions for unknown or duplicate items, missing
decisions, and a non-default action with no valid citation.

Call.inputs:
  items: [{"id", "allowed_actions": [...], "default_action", "description",
           "allowed_evidence": [ids] (optional, else Call.allowed)}]
  state: str            -- current state the decision applies to, shown to the model
  constraints: str      -- human-readable statement of the hard limits
  state_obj: Any        -- optional, passed through to constrain()
"""

from __future__ import annotations

from typing import Callable

from .runtime import ArchetypeSpec, Call, Result
from .validators import DEFAULT_NUMBERS, NumberPolicy, Problem, citation_stats, validate_claims

# (decision, item, call) -> (possibly corrected decision, human-readable overrides)
ConstrainFn = Callable[[dict, dict, Call], tuple[dict, list[str]]]

CONTRACT = (
    "Output rules (checked automatically after you answer):\n"
    "- Return exactly one decision per candidate, using its id as item_id.\n"
    "- Choose only from that candidate's allowed actions. Its default (no-op) action is a normal, "
    "often correct answer.\n"
    "- Any non-default action needs a rationale citing evidence ids in evidence_ids; copy numbers exactly.\n"
    "- Hard limits are enforced in code after you answer; anything outside them is corrected and logged."
)


def schema(actions: tuple[str, ...], extra_properties: dict | None = None, extra_required: list[str] | None = None) -> dict:
    decision = {
        "type": "object",
        "properties": {
            "item_id": {"type": "string"},
            "action": {"type": "string", "enum": list(actions)},
            "rationale": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
            **(extra_properties or {}),
        },
        "required": ["item_id", "action", "rationale", "evidence_ids", *(extra_required or [])],
    }
    return {"type": "object", "properties": {"decisions": {"type": "array", "items": decision},
                                             "note": {"type": "string"}},
            "required": ["decisions", "note"]}


def _render(call: Call) -> str:
    i = call.inputs
    blocks = []
    for item in i["items"]:
        blocks.append(f"### {item['id']}\nAllowed actions: {', '.join(item['allowed_actions'])} "
                      f"(default: {item['default_action']})\n{item.get('description', '').strip()}")
    parts = []
    if i.get("state"):
        parts.append(f"Current state:\n{i['state']}")
    if i.get("constraints"):
        parts.append(f"Hard limits:\n{i['constraints']}")
    parts.append("Candidates:\n\n" + "\n\n".join(blocks))
    parts.append(f"Evidence:\n{call.evidence_block()}")
    parts.append("Return one decision per candidate.")
    return "\n\n".join(parts)


def _no_op(item: dict, reason: str) -> dict:
    return {"item_id": item["id"], "action": item["default_action"], "rationale": reason, "evidence_ids": [],
            "overrides": [reason]}


def _default(call: Call, reason: str) -> dict:
    decisions = [_no_op(item, f"decision call failed: {reason[:160]}") for item in call.inputs["items"]]
    return {"decisions": decisions, "note": "", "overrides": [d["overrides"][0] for d in decisions]}


def _check(constrain: ConstrainFn | None, numbers: NumberPolicy):
    def check(raw: dict, call: Call) -> tuple[dict, list[Problem]]:
        items = {it["id"]: it for it in call.inputs["items"]}
        decisions, problems, seen = [], [], set()
        for n, d in enumerate(raw.get("decisions") or []):
            where = f"decisions[{n}]"
            item = items.get(d.get("item_id"))
            if item is None or item["id"] in seen:
                problems.append(Problem("unknown_item", f"{where} item_id {d.get('item_id')!r} is not a candidate or is "
                                        f"a duplicate. Candidates: {', '.join(items)}.", where))
                continue
            seen.add(item["id"])
            d = dict(d)
            overrides: list[str] = []
            if d["action"] not in item["allowed_actions"]:
                overrides.append(f"{item['id']}: action {d['action']!r} not allowed here, set to {item['default_action']!r}")
                d["action"] = item["default_action"]
            if d["action"] != item["default_action"]:
                allowed = set(item.get("allowed_evidence") or call.allowed)
                kept, p = validate_claims([{"claim": d.get("rationale", ""), "evidence_ids": d.get("evidence_ids") or []}],
                                          allowed, call.evidence, f"{where}.rationale", numbers)
                problems += p
                if not kept:
                    overrides.append(f"{item['id']}: {d['action']!r} had no valid citation, set to {item['default_action']!r}")
                    d["action"] = item["default_action"]
                else:
                    d["evidence_ids"] = kept[0]["evidence_ids"]
            if constrain is not None:
                d, extra = constrain(d, item, call)
                overrides += extra
            d["overrides"] = overrides
            problems += [Problem("override", o, where, heal=False) for o in overrides]
            decisions.append(d)
        missing = [i for i in items if i not in seen]
        if missing:
            problems.append(Problem("missing_decision", f"No decision for {', '.join(missing)}. "
                                    "Return one decision for every candidate.", "decisions"))
            decisions += [_no_op(items[i], "no decision returned; defaulted to no action") for i in missing]
        return {"decisions": decisions, "note": raw.get("note", ""),
                "overrides": [o for d in decisions for o in d["overrides"]]}, problems
    return check


def _metrics(result: Result, call: Call) -> dict:
    decisions = result.output["decisions"]
    items = {it["id"]: it for it in call.inputs["items"]}
    acted = [d for d in decisions if d["action"] != items[d["item_id"]]["default_action"]]
    claims = [{"claim": d["rationale"], "evidence_ids": d["evidence_ids"]} for d in acted]
    return {
        **citation_stats(claims, [p for p in result.validation.problems if p.path.endswith(".rationale")]),
        "override_rate": sum(1 for d in decisions if d["overrides"]) / len(decisions) if decisions else 0.0,
        "action_rate": len(acted) / len(decisions) if decisions else 0.0,
        "actions": {d["item_id"]: d["action"] for d in decisions},
        "heal_attempts": result.attempts - 1,
    }


def decision_maker(actions: tuple[str, ...], constrain: ConstrainFn | None = None,
                   extra_properties: dict | None = None, extra_required: list[str] | None = None,
                   numbers: NumberPolicy = DEFAULT_NUMBERS, max_heal_attempts: int = 2) -> ArchetypeSpec:
    return ArchetypeSpec(
        name="decision_maker", output_schema=schema(actions, extra_properties, extra_required), contract=CONTRACT,
        check=_check(constrain, numbers), render=_render, default_output=_default,
        max_heal_attempts=max_heal_attempts, metrics=_metrics,
    )
