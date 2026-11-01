"""Policy layer — Omnigent-inspired (docs/01_harness_spec.md S6): guardrails
enforced as deterministic checks outside the model's own reasoning, not as
prompt text the model can ignore. This is a fixed safety boundary, same
role as agents/base.py's client wiring or llm_provider/ -- it is NEVER part
of the RSI-editable component tree (rsi/harnesses.py's RESEARCH_HARNESS
excludes it by construction, not by convention).

Omnigent's ALLOW/ASK/DENY model is adopted as-is. ASK has no meaning in a
batch/non-interactive pipeline (there is no approval UI), so it is treated
as DENY-with-a-logged-reason here -- documented, not silently downgraded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Decision(Enum):
    ALLOW = "allow"
    ASK = "ask"  # degrades to DENY in this batch pipeline -- see module docstring
    DENY = "deny"


@dataclass
class PolicyState:
    started_at: float = field(default_factory=time.time)
    revision_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


@dataclass
class PolicyResult:
    decision: Decision
    reason: str = ""


Policy = Callable[[PolicyState], PolicyResult]


def hard_revision_cap(limit: int = 6) -> Policy:
    """Independent of MAX_REVISIONS in research_pipeline.py -- that constant
    is RSI-editable; this cap is not. A bad self-edit that inflates
    MAX_REVISIONS still can't exceed this."""
    def _policy(state: PolicyState) -> PolicyResult:
        if state.revision_count >= limit:
            return PolicyResult(Decision.DENY, f"revision_count {state.revision_count} >= hard cap {limit}")
        return PolicyResult(Decision.ALLOW)
    return _policy


def episode_timeout(seconds: int = 900) -> Policy:
    def _policy(state: PolicyState) -> PolicyResult:
        if state.elapsed_seconds >= seconds:
            return PolicyResult(Decision.DENY, f"elapsed {state.elapsed_seconds:.0f}s >= timeout {seconds}s")
        return PolicyResult(Decision.ALLOW)
    return _policy


def token_budget(max_tokens: int = 500_000) -> Policy:
    def _policy(state: PolicyState) -> PolicyResult:
        if state.total_tokens >= max_tokens:
            return PolicyResult(Decision.DENY, f"total_tokens {state.total_tokens} >= budget {max_tokens}")
        return PolicyResult(Decision.ALLOW)
    return _policy


DEFAULT_POLICIES: list[Policy] = [hard_revision_cap(), episode_timeout(), token_budget()]


def evaluate(policies: list[Policy], state: PolicyState) -> PolicyResult:
    """DENY wins over ASK wins over ALLOW -- evaluated in declared order,
    same precedence rule as Omnigent's policy engine."""
    worst = PolicyResult(Decision.ALLOW)
    for policy in policies:
        result = policy(state)
        if result.decision == Decision.DENY:
            return result
        if result.decision == Decision.ASK and worst.decision == Decision.ALLOW:
            worst = result
    return worst
