"""Archetype tests -- no model or API key required (ScriptedLLM). Run with:
    python -m tests.test_general_agents

Fixtures use one neutral example domain throughout (evaluating a service
release); nothing here is specific to it.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from karmic.agents.general import (
    NOT_FOUND, Call, EvidenceIndex, Field, Fixture, LLMError, NumberPolicy, RoleConfig, ScriptedLLM, debate,
    debater, decision_maker, evidence_analyst, extractor, load_fixtures, reflector, run, save_fixtures,
    score_fixtures,
)
from karmic.agents.general.validators import find_span, number_supported, unsupported_numbers, validate_claims, validate_schema

ROLE = RoleConfig(role="test_role", system_prompt="You are a test analyst.")


def _evidence() -> EvidenceIndex:
    ev = EvidenceIndex()
    ev.add("metric", "load test release 4.2: 1,250,000 requests, p95 latency 412 ms", {"requests": 1250000, "p95_ms": 412})  # ev1
    ev.add("metric", "release 4.2 canary error_rate = 0.084", {"value": 0.084})  # ev2
    ev.add("incident", "incident 2026-09-10: 3,400,000 failed requests during the 4.2 canary", {"failed": 3400000})  # ev3
    return ev


def _analysis(**over) -> dict:
    out = {"stance": "positive", "strength": 0.6, "key_points": [], "risks": [], "data_gaps": [], "reasoning": "r"}
    out.update(over)
    return out


def test_number_check_accepts_unit_changes():
    assert number_supported(8.4, [0.084])  # fraction quoted as percent
    assert number_supported(1.25, [1250000])  # quoted in millions
    assert number_supported(3, [])  # small integer counts are exempt
    assert not number_supported(4.1, [0.084, 1250000])
    print("PASS: test_number_check_accepts_unit_changes")


def test_name_numerals_come_from_the_consumer():
    # A numeral inside a name isn't a figure, but which names occur is domain knowledge:
    # the default policy knows none, and the consumer adds its own.
    text = "Returned HTTP 500 on 12% of requests, timestamps in ISO 8601"
    assert unsupported_numbers(text, "error_rate = 0.084") == [500.0, 12.0, 8601.0]
    policy = NumberPolicy(ignore=(r"\bHTTP\s*\d{3}\b", r"\bISO\s*\d+\b"))
    assert unsupported_numbers(text, "error_rate = 0.084", policy) == [12.0]
    print("PASS: test_name_numerals_come_from_the_consumer")


def test_heal_loop_stops_when_no_progress():
    stuck = _analysis(key_points=[{"claim": "Errors up 12%", "evidence_ids": ["ev2"]}])
    llm = ScriptedLLM([stuck, stuck, stuck])
    result = run(evidence_analyst(), ROLE, Call(inputs={}, evidence=_evidence()), llm)
    assert result.attempts == 2 and result.meta.get("stalled") and len(llm.calls) == 2
    print("PASS: test_heal_loop_stops_when_no_progress")


def test_validate_claims_drops_and_flags():
    ev = _evidence()
    kept, problems = validate_claims([
        {"claim": "Handled 1.25 million requests", "evidence_ids": ["ev1"]},
        {"claim": "No citation here", "evidence_ids": []},
        {"claim": "Cites a ghost", "evidence_ids": ["ev99"]},
        {"claim": "Canary errors were 12%", "evidence_ids": ["ev2"]},
    ], {"ev1", "ev2", "ev3"}, ev)
    assert [c["claim"] for c in kept] == ["Handled 1.25 million requests", "Canary errors were 12%"]
    assert kept[1]["numbers_unverified"] == [12.0]
    assert sorted(p.kind for p in problems) == ["no_citation", "numbers_unverified", "unknown_citation"]
    assert "0.084" in next(p.detail for p in problems if p.kind == "numbers_unverified")  # tells it what IS there
    print("PASS: test_validate_claims_drops_and_flags")


def test_schema_validation():
    schema = evidence_analyst().output_schema
    assert validate_schema(_analysis(), schema) == []
    bad = validate_schema({"stance": "great", "strength": "high"}, schema)
    details = " ".join(p.detail for p in bad)
    assert "missing required field" in details and "one of" in details and "type number" in details
    print("PASS: test_schema_validation")


def test_heal_loop_feeds_validator_problems_back():
    llm = ScriptedLLM([
        _analysis(key_points=[{"claim": "Canary errors were 12%", "evidence_ids": ["ev2"]}]),
        _analysis(key_points=[{"claim": "Canary errors were 8.4%", "evidence_ids": ["ev2"]}]),
    ])
    result = run(evidence_analyst(), ROLE, Call(inputs={"subject": "release 4.2"}, evidence=_evidence()), llm)
    assert result.attempts == 2
    assert result.validation.problems == []
    assert result.output["key_points"][0]["claim"] == "Canary errors were 8.4%"
    second = llm.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert "12" in second[2]["content"] and "ev2" in second[2]["content"]  # specific, not a blind retry
    assert "Output rules" in llm.calls[0]["system"] and llm.calls[0]["system"].startswith("You are a test analyst.")
    print("PASS: test_heal_loop_feeds_validator_problems_back")


def test_heal_budget_exhausted_returns_best_attempt():
    bad = _analysis(key_points=[{"claim": "Up 50%", "evidence_ids": ["ev2"]}, {"claim": "x", "evidence_ids": []}])
    better = _analysis(key_points=[{"claim": "Up 50%", "evidence_ids": ["ev2"]}])
    llm = ScriptedLLM([bad, better, bad])
    result = run(evidence_analyst(), ROLE, Call(inputs={}, evidence=_evidence()), llm)
    assert result.attempts == 3 and not result.validation.failed
    assert len(result.validation.heal_problems) == 1  # the `better` attempt
    assert result.recurring_problem_kinds == ["numbers_unverified", "no_citation"]  # seen 3x, 2x
    assert result.caution_candidates("test_role")[0].startswith("test_role: repeated numbers_unverified")
    print("PASS: test_heal_budget_exhausted_returns_best_attempt")


def test_never_raises_on_model_failure():
    llm = ScriptedLLM([LLMError("connection refused"), {"stance": "nope"}, LLMError("timeout")])
    result = run(evidence_analyst(), ROLE, Call(inputs={}, evidence=_evidence()), llm)
    assert result.validation.failed
    assert result.output["stance"] == "neutral" and result.output["strength"] == 0.0
    assert "timeout" in result.output["data_gaps"][0]
    print("PASS: test_never_raises_on_model_failure")


def test_memory_is_injected_as_advisory():
    llm = ScriptedLLM([_analysis()])
    run(evidence_analyst(), ROLE, Call(inputs={}, evidence=_evidence(), memory="Canary error spikes under 1% were noise."), llm)
    first = llm.calls[0]["messages"][0]["content"]
    assert first.startswith("Lessons from past episodes") and "evidence below wins" in first
    print("PASS: test_memory_is_injected_as_advisory")


def test_debater_flags_one_sided_citation_and_runs_rounds():
    ev = _evidence()
    one_sided = {"case": "c", "points": [{"claim": "Canary failed", "evidence_ids": ["ev3"]}],
                 "strongest_point": {"claim": "Canary failed", "evidence_ids": ["ev3"]}}
    for_case = {"case": "case for shipping", "points": [{"claim": "p95 412 ms", "evidence_ids": ["ev1"]}],
                "strongest_point": {"claim": "p95 412 ms", "evidence_ids": ["ev1"]}}
    against = {"case": "case against shipping", "points": [{"claim": "Canary failed", "evidence_ids": ["ev3"]}],
               "strongest_point": {"claim": "Canary failed", "evidence_ids": ["ev3"]}}
    llm = ScriptedLLM([one_sided, for_case, against, for_case, against])
    opposing = {"for": ["ev3"], "against": ["ev1"]}

    def make_call(side, opponent_case):
        return Call(inputs={"side": side, "subject": "ship release 4.2", "opponent_case": opponent_case,
                            "opposing_evidence": opposing[side]}, evidence=ev)

    roles = {"for": RoleConfig("for", "Argue for."), "against": RoleConfig("against", "Argue against.")}
    out = debate(debater(), roles, make_call, llm, rounds=2)
    assert out["for"][0].attempts == 2 and out["for"][0].history[0].problems[0].kind == "one_sided_citation"
    assert out["for"][0].output["case"] == "case for shipping"
    rebuttal_prompt = llm.calls[3]["messages"][0]["content"]  # "for", round 2
    assert "The other side argued:\ncase against shipping" in rebuttal_prompt
    print("PASS: test_debater_flags_one_sided_citation_and_runs_rounds")


def _release_call(ev):
    return Call(inputs={
        "items": [
            {"id": "release-4.2", "allowed_actions": ["keep", "rollback"], "default_action": "keep",
             "description": "deployed to 10% of traffic"},
            {"id": "release-4.3", "allowed_actions": ["ship", "hold"], "default_action": "hold", "description": "candidate"},
            {"id": "flag-new-cache", "allowed_actions": ["ship", "hold"], "default_action": "hold", "description": "candidate"},
        ],
        "state": "4.1 serves 90% of traffic", "constraints": "max 25% rollout per decision",
    }, evidence=ev)


def test_decision_maker_enforces_and_logs_overrides():
    def cap_rollout(d, item, call):
        if d.get("rollout_pct", 0) > 25:
            return {**d, "rollout_pct": 25}, [f"{item['id']}: rollout {d['rollout_pct']}% capped to 25%"]
        return d, []

    spec = decision_maker(("ship", "hold", "keep", "rollback"), constrain=cap_rollout,
                          extra_properties={"rollout_pct": {"type": "number"}}, extra_required=["rollout_pct"])
    llm = ScriptedLLM([{"note": "", "decisions": [
        {"item_id": "release-4.2", "action": "ship", "rationale": "more", "evidence_ids": ["ev1"], "rollout_pct": 0},
        {"item_id": "release-4.3", "action": "ship", "rationale": "p95 was 412 ms", "evidence_ids": ["ev1"], "rollout_pct": 60},
        {"item_id": "flag-new-cache", "action": "ship", "rationale": "vibes", "evidence_ids": [], "rollout_pct": 5},
    ]}] * 3)
    result = run(spec, ROLE, _release_call(_evidence()), llm)
    assert result.metrics["actions"] == {"release-4.2": "keep", "release-4.3": "ship", "flag-new-cache": "hold"}
    shipped = next(d for d in result.output["decisions"] if d["item_id"] == "release-4.3")
    assert shipped["rollout_pct"] == 25 and shipped["overrides"]
    assert abs(result.metrics["override_rate"] - 1.0) < 1e-9  # all three corrected by code
    # Overrides aren't healed; the uncited "ship" is, so the model was re-asked -- once,
    # since the identical second answer made no progress.
    assert result.attempts == 2 and result.meta["stalled"]
    assert all(p.kind == "no_citation" for p in result.validation.heal_problems)
    print("PASS: test_decision_maker_enforces_and_logs_overrides")


def test_decision_maker_fills_missing_decisions():
    llm = ScriptedLLM([{"note": "", "decisions": [
        {"item_id": "release-4.2", "action": "rollback", "rationale": "3,400,000 failed requests", "evidence_ids": ["ev3"]}]}] * 3)
    result = run(decision_maker(("ship", "hold", "keep", "rollback")), ROLE, _release_call(_evidence()), llm)
    assert result.metrics["actions"] == {"release-4.2": "rollback", "release-4.3": "hold", "flag-new-cache": "hold"}
    assert any(p.kind == "missing_decision" for p in result.history[0].problems)
    print("PASS: test_decision_maker_fills_missing_decisions")


DOC = ("Incident report (sev2).\n"
       "On September 3, 2026, checkout latency rose to 2.4 seconds for  about 18,000 users.\n"
       "Root cause: the cache warm-up step was skipped during the 4.2 deploy. We rolled back to 4.1.")


def test_find_span_is_whitespace_and_case_insensitive():
    span = find_span("2.4 seconds for about 18,000 USERS", DOC)
    assert span is not None and DOC[span[0]:span[1]] == "2.4 seconds for  about 18,000 users"
    assert find_span("the database ran out of disk", DOC) is None
    print("PASS: test_find_span_is_whitespace_and_case_insensitive")


def test_extractor_grounds_labels_and_numbers():
    spec = extractor({
        "remediation": Field(kind="label", labels=("rollback", "hotfix", "config_change"), multiple=True),
        "affected_users": Field(kind="number", required=False),
        "owner": Field(kind="text"),
    })
    llm = ScriptedLLM([
        {"extractions": [
            {"field": "remediation", "value": "revert", "quote": "We rolled back to 4.1"},
            {"field": "affected_users", "value": "20000", "quote": "about 18,000 users"},
            {"field": "affected_users", "value": "about 18,000 users were affected", "quote": "about 18,000 users"},
            {"field": "remediation", "value": "rollback", "quote": "we reverted the deploy"},
        ]},
        {"extractions": [
            {"field": "remediation", "value": "rollback", "quote": "We rolled back to 4.1"},
            {"field": "affected_users", "value": "18,000", "quote": "about 18,000 users"},
            {"field": "owner", "value": NOT_FOUND, "quote": ""},
        ]},
    ])
    result = run(spec, ROLE, Call(inputs={"document": DOC}), llm)
    first_kinds = sorted(p.kind for p in result.history[0].problems)
    assert first_kinds == ["label_not_allowed", "missing_field", "not_a_number", "not_grounded", "numbers_unverified"], first_kinds
    assert result.attempts == 2 and result.validation.problems == []
    assert result.output["values"] == {"remediation": ["rollback"], "affected_users": "18,000", "owner": None}
    assert result.metrics["grounded_rate"] == 1.0
    print("PASS: test_extractor_grounds_labels_and_numbers")


def test_reflector_groundedness_and_abstraction():
    episode = ("Shipped release 4.2 of checkout-service to 100% after a 412 ms p95 load test; error rate rose to 8.4% "
               "within 12 minutes and it was rolled back. 3,400,000 requests failed.")
    llm = ScriptedLLM([
        {"abstracted_pattern": "checkout-service deploys are risky", "failure_summary": "Lost 19.5 hours of uptime",
         "distilled_learnings": ["Use a canary"]},
        {"abstracted_pattern": "A full rollout gated only on a load test missed failures that real traffic exposed",
         "failure_summary": "Assumed a 412 ms p95 load test predicted production behaviour",
         "distilled_learnings": ["Gate full rollouts on a canary's error rate, not on load tests alone"]},
    ])
    result = run(reflector(), ROLE, Call(inputs={"episode": episode, "outcome": "failure",
                                                 "entity_denylist": ["checkout-service"]}), llm)
    assert sorted(p.kind for p in result.history[0].problems) == ["entity_in_pattern", "numbers_ungrounded"]
    assert result.attempts == 2 and result.metrics == {"groundedness": 1.0, "abstracted": True, "heal_attempts": 1}
    ep = result.to_episode("reflector", domain="release_reviews")
    assert ep.healer_attempt_count == 1 and ep.outcome.value == "success"
    print("PASS: test_reflector_groundedness_and_abstraction")


def test_fixtures_roundtrip_and_score():
    ev = _evidence()
    call = Call(inputs={"subject": "release 4.2"}, evidence=ev, allowed_evidence={"ev1", "ev3"})
    fixtures = [
        Fixture.from_call("planted_canary_failure", "analyst", call,
                          expect=[{"path": "stance", "equals": "negative"}, {"must_cite": ["ev3"]}]),
        Fixture.from_call("clear_pass", "analyst", call, expect=[{"path": "stance", "in": ["positive"]}]),
    ]
    with tempfile.TemporaryDirectory() as d:
        save_fixtures(Path(d) / "fx.jsonl", fixtures)
        loaded = load_fixtures(Path(d) / "fx.jsonl")
    assert loaded[0].to_call().allowed == {"ev1", "ev3"}
    llm = ScriptedLLM([
        _analysis(stance="negative", risks=[{"claim": "3,400,000 requests failed", "evidence_ids": ["ev3"]}]),
        _analysis(stance="negative"),
    ])
    score = score_fixtures(evidence_analyst(), {"analyst": ROLE}, loaded, llm)
    assert [r.passed for r in score.runs] == [True, False]
    assert score.pass_rate == 0.5 and abs(score.scalar_score - (50 + 20 + 20)) < 1e-9
    assert "clear_pass: FAIL" in score.to_prompt_text()
    print("PASS: test_fixtures_roundtrip_and_score")


if __name__ == "__main__":
    test_number_check_accepts_unit_changes()
    test_name_numerals_come_from_the_consumer()
    test_heal_loop_stops_when_no_progress()
    test_validate_claims_drops_and_flags()
    test_schema_validation()
    test_heal_loop_feeds_validator_problems_back()
    test_heal_budget_exhausted_returns_best_attempt()
    test_never_raises_on_model_failure()
    test_memory_is_injected_as_advisory()
    test_debater_flags_one_sided_citation_and_runs_rounds()
    test_decision_maker_enforces_and_logs_overrides()
    test_decision_maker_fills_missing_decisions()
    test_find_span_is_whitespace_and_case_insensitive()
    test_extractor_grounds_labels_and_numbers()
    test_reflector_groundedness_and_abstraction()
    test_fixtures_roundtrip_and_score()
    print("\nAll general agent tests passed.")
