"""Agent archetypes: reusable single-call roles, each shipping a runtime, a
fixed validator, a self-healing loop, Cortex hooks and a benchmark shape.
The consumer supplies prompts, evidence and fixtures; orchestration stays
with the consumer. See docs/09_agent_archetypes_spec.md.

    from agents.general import Call, EvidenceIndex, RoleConfig, evidence_analyst, ollama_json, run

    ev = EvidenceIndex()
    ev.add("metric", "load test 2026-08-01: p95 latency 412 ms at 1,200 req/s", {"p95_ms": 412, "rps": 1200})
    role = RoleConfig.from_file("latency_analyst", "prompts/latency_analyst.md")
    result = run(evidence_analyst(), role, Call(inputs={"subject": "release 4.2"}, evidence=ev), ollama_json("qwen3:14b"))
    result.output, result.validation.problems, result.metrics
"""

from .debater import debate, debater, digest
from .decision_maker import ConstrainFn, decision_maker
from .evidence import EvidenceIndex, Item
from .evidence_analyst import evidence_analyst
from .extractor import NOT_FOUND, Field, extractor
from .fixtures import Fixture, FixtureScore, load_fixtures, save_fixtures, score_fixtures
from .llm import JsonLLM, LLMError, ScriptedLLM, anthropic_json, ollama_json
from .reflector import reflector, to_episode_fields
from .runtime import ArchetypeSpec, Attempt, Call, Result, RoleConfig, run
from .validators import NumberPolicy, Problem, ValidationReport

__all__ = [
    "ArchetypeSpec", "Attempt", "Call", "Result", "RoleConfig", "run",
    "EvidenceIndex", "Item", "NumberPolicy", "Problem", "ValidationReport",
    "JsonLLM", "LLMError", "ScriptedLLM", "anthropic_json", "ollama_json",
    "evidence_analyst", "debater", "debate", "digest", "decision_maker", "ConstrainFn",
    "extractor", "Field", "NOT_FOUND", "reflector", "to_episode_fields",
    "Fixture", "FixtureScore", "load_fixtures", "save_fixtures", "score_fixtures",
]
