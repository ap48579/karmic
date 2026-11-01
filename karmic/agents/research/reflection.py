"""The Grader/Reviser loop plus the Reflection stage that produces Cortex's
write-back fields -- the two things confirmed missing from vendored GPT
Researcher (docs/01_harness_spec.md S1/S2). Neither is vendored; both call
through GenericLLMProvider directly, reusing whatever Qwen configuration
agents/research/qwen_config.configure_qwen() already set, so there's no
second, parallel client implementation.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from gpt_researcher.llm_provider.generic import GenericLLMProvider

GRADER_SYSTEM_PROMPT = (
    "You are the Grader agent -- independent of whatever produced the draft. Check the "
    "draft report against the original question. Flag: (1) any claim not obviously "
    "supported by a citation, (2) any part of the question left unaddressed, (3) any "
    "citation whose URL doesn't plausibly support the claim next to it, (4) any "
    "contradiction the draft silently resolved instead of flagging. "
    "Respond in exactly this format:\n"
    "VERDICT: pass|revise|fail\n"
    "MISTAKES:\n- <category>: <specific description>\n- ...\n"
    "(VERDICT=pass only if there are zero mistakes. VERDICT=fail only if the draft is "
    "unsalvageable and should be regenerated from scratch, not revised.)"
)

REVISER_SYSTEM_PROMPT = (
    "You are the Reviser agent. Given a draft research report and a specific list of "
    "mistakes found in it, produce a corrected draft that fixes every listed mistake "
    "while preserving everything that wasn't flagged. Do not introduce new unsupported "
    "claims while fixing old ones. Output the full corrected report, nothing else."
)

REFLECTION_SYSTEM_PROMPT = (
    "You distill a completed research episode into Cortex memory fields. "
    "If it succeeded, produce a short 'abstracted_pattern' label describing the general "
    "research strategy that worked (not the specific facts found). If it failed, produce "
    "a short 'failure_summary' describing what was wrongly assumed or missed. "
    "Then produce 1-3 short 'distilled_learnings' bullets. "
    "Respond in exactly this format:\n"
    "PATTERN: <text or NONE>\nFAILURE: <text or NONE>\nLEARNINGS:\n- ...\n- ..."
)


@dataclass
class GraderVerdict:
    verdict: str  # pass | revise | fail
    mistakes: list[str] = field(default_factory=list)
    dry_run: bool = False


@dataclass
class ReflectionResult:
    abstracted_pattern: str | None = None
    failure_summary: str | None = None
    distilled_learnings: list[str] = field(default_factory=list)


async def _call(tier: str, system: str, user: str) -> tuple[str, int, int]:
    """Returns (text, input_tokens, output_tokens). Empty text + zero tokens
    signals dry-run (configure_qwen() never set this tier -- no QWEN_API_KEY)."""
    llm_str = os.environ.get(tier)
    if not llm_str:
        return "", 0, 0
    provider, model = llm_str.split(":", 1)
    llm = GenericLLMProvider.from_provider(provider, model=model, temperature=0.3, verbose=False)
    text = await llm.get_chat_response(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        stream=False,
    )
    usage = llm.last_usage_metadata or {}
    return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def _parse_verdict(text: str) -> GraderVerdict:
    verdict_m = re.search(r"VERDICT:\s*(\w+)", text)
    mistakes = re.findall(r"^- (.+)$", text, re.MULTILINE)
    verdict = verdict_m.group(1).strip().lower() if verdict_m else "revise"
    if verdict not in ("pass", "revise", "fail"):
        verdict = "revise"
    return GraderVerdict(verdict=verdict, mistakes=mistakes)


async def grade(question: str, draft: str) -> tuple[GraderVerdict, int, int]:
    prompt = f"Research question: {question}\n\nDraft report:\n{draft}\n\nGrade it."
    text, in_tok, out_tok = await _call("SMART_LLM", GRADER_SYSTEM_PROMPT, prompt)
    if not text:
        return GraderVerdict(verdict="revise", mistakes=["dry-run: no QWEN_API_KEY configured"], dry_run=True), 0, 0
    return _parse_verdict(text), in_tok, out_tok


async def revise(draft: str, mistakes: list[str]) -> tuple[str, int, int]:
    mistakes_text = "\n".join(f"- {m}" for m in mistakes)
    prompt = f"Draft report:\n{draft}\n\nMistakes to fix:\n{mistakes_text}\n\nProduce the corrected draft."
    text, in_tok, out_tok = await _call("SMART_LLM", REVISER_SYSTEM_PROMPT, prompt)
    return (text or draft), in_tok, out_tok


def _parse_reflection(text: str) -> ReflectionResult:
    pattern_m = re.search(r"PATTERN:\s*(.+)", text)
    failure_m = re.search(r"FAILURE:\s*(.+)", text)
    learnings = re.findall(r"^- (.+)$", text, re.MULTILINE)
    pattern = pattern_m.group(1).strip() if pattern_m else None
    failure = failure_m.group(1).strip() if failure_m else None
    return ReflectionResult(
        abstracted_pattern=None if pattern in (None, "NONE") else pattern,
        failure_summary=None if failure in (None, "NONE") else failure,
        distilled_learnings=learnings,
    )


async def reflect(question: str, outcome: str, summary_of_what_happened: str) -> tuple[ReflectionResult, int, int]:
    prompt = f"Task: {question}\nOutcome: {outcome}\nWhat happened:\n{summary_of_what_happened}"
    text, in_tok, out_tok = await _call("FAST_LLM", REFLECTION_SYSTEM_PROMPT, prompt)
    if not text:
        return ReflectionResult(), 0, 0
    return _parse_reflection(text), in_tok, out_tok
