"""Proposer -- both RSI tiers' edit-proposal logic (docs/05_rsi_harness2_design_spec.md
S3/S4). Originally two separate implementations: selfmod/meta_agent.py
(Claude-only, coding harness) and this module (Qwen-only, research harness).
Unified around an injected `llm_call` (rsi/llm_adapters.py) so this module
has no provider-specific knowledge at all -- a HarnessSpec plus any
llm_call works, for any repo.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .harness_spec import Component, HarnessSpec
from .llm_adapters import LLMCall

MAX_EDITS_PER_CYCLE = 3
MAX_EDIT_CHARS = 4000
MAX_FILE_CHARS = 1200  # see docs/01_harness_spec.md S6 Q4 / docs/05 S6 Q4 -- binary-
# searched live against a local Ollama/qwen3:14b setup: an ~8KB prompt returns valid
# JSON, an ~20KB prompt returns a coherent but wrong-format free-form essay instead
# (the model stays on-topic, it just drops the JSON-only instruction -- not garbling).
# Kept conservative so the engine works out of the box against small local models; a
# consumer with a larger-context setup (hosted API, or a bigger Ollama num_ctx) can
# raise this safely. Truncation is always safe regardless: apply_edits() validates
# old_string against the REAL, untruncated file at apply time, so a model proposing
# an edit against content it couldn't see just gets a clean rejection, never a
# silently-wrong edit.


@dataclass
class ProposedEdit:
    file: str
    rationale: str
    old_string: str
    new_string: str


@dataclass
class SelfEditPlan:
    summary: str
    edits: list[ProposedEdit] = field(default_factory=list)
    parse_error: str | None = None


_RULES = f"""Rules:
- Propose at most {MAX_EDITS_PER_CYCLE} edits per cycle. Fewer, sharper edits beat many.
- Each edit is an exact (old_string, new_string) replacement. old_string must \
appear VERBATIM and EXACTLY ONCE in the target file (including whitespace/indentation) \
-- copy it character-for-character from the file content you were given.
- You may ONLY propose edits to files in the allowlist given below. Never propose \
edits to tests, benchmarks, or the self-improvement engine itself.
- Ground every edit in something specific from the report/trajectory you were given \
-- state the hypothesis ("this should reduce X because Y") in `rationale`.
- Prefer the smallest change that tests the hypothesis: a tuned constant, a sentence \
added to a prompt -- not a rewrite of a whole function.
- If nothing is worth changing, return an empty edits list rather than inventing busywork.

Respond with ONLY a single JSON object, no prose before or after, no markdown fence:
{{"summary": "<one sentence>", "edits": [{{"file": "...", "rationale": "...", \
"old_string": "...", "new_string": "..."}}]}}
"""


def _harnessing_system_prompt(harness: HarnessSpec) -> str:
    files = "\n".join(f"- {f}" for f in harness.allowed_files)
    return (
        f"You are the Harnessing-level Proposer for {harness.description}. "
        "You read a trace report of its recent real-world performance across many "
        "episodes and propose small, targeted edits to improve it going forward.\n\n"
        f"You may ONLY propose edits to these files:\n{files}\n\n" + _RULES
    )


def _task_system_prompt(component: Component) -> str:
    files = "\n".join(f"- {f}" for f in component.files)
    return (
        "You are the Task-level Proposer. A single episode keeps running into the same "
        "category of mistake. Propose a small edit, scoped to fixing that one mistake "
        "pattern for the rest of THIS episode -- not a general improvement.\n\n"
        f"You may ONLY propose edits to these files:\n{files}\n\n" + _RULES
    )


def _extract_json(text: str) -> dict | None:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if not brace:
            return None
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            return None


def _build_plan(text: str, allowed_files: list[str]) -> SelfEditPlan:
    data = _extract_json(text)
    if data is None:
        return SelfEditPlan(summary="skipped: planner response was not valid JSON",
                             edits=[], parse_error="unparseable_response")
    raw_edits = data.get("edits", [])[:MAX_EDITS_PER_CYCLE]
    edits: list[ProposedEdit] = []
    for e in raw_edits:
        try:
            file, rationale, old, new = e["file"], e["rationale"], e["old_string"], e["new_string"]
        except (KeyError, TypeError):
            continue
        if file not in allowed_files:
            continue
        if not old.strip() or len(old) > MAX_EDIT_CHARS or len(new) > MAX_EDIT_CHARS:
            continue
        edits.append(ProposedEdit(file=file, rationale=rationale, old_string=old, new_string=new))
    return SelfEditPlan(summary=data.get("summary", ""), edits=edits)


def _render_files(repo_root: Path, files: list[str]) -> str:
    sections = []
    for rel_path in files:
        content = (repo_root / rel_path).read_text()
        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS] + f"\n# ... TRUNCATED at {MAX_FILE_CHARS} chars ..."
        sections.append(f"### {rel_path}\n```python\n{content}\n```")
    return "\n\n".join(sections)


async def propose_from_report(
    repo_root: Path, harness: HarnessSpec, llm_call: LLMCall, report_text: str, lineage_history_text: str,
) -> SelfEditPlan:
    """Harnessing-level: cross-episode Cortex trace report -> candidate edit."""
    file_sections = _render_files(repo_root, harness.allowed_files)
    prompt = (
        f"## Trace report (real production episodes, domain={harness.cortex_domain})\n{report_text}\n\n"
        f"## Lineage history (prior self-improvement cycles for this harness)\n{lineage_history_text}\n\n"
        f"## Allowlisted files you may edit\n{', '.join(harness.allowed_files)}\n\n"
        f"## Current source\n{file_sections}\n\nPropose this cycle's edit plan."
    )
    text = await llm_call(_harnessing_system_prompt(harness), prompt)
    if not text:
        return SelfEditPlan(summary="skipped: dry-run (no LLM credentials configured)",
                             edits=[], parse_error="dry_run")
    return _build_plan(text, harness.allowed_files)


async def propose_from_trajectory(
    repo_root: Path, harness: HarnessSpec, llm_call: LLMCall, mistakes: list[str], component_id: str | None = None,
) -> SelfEditPlan:
    """Task-level: this episode's repeated mistakes -> candidate edit, scoped
    to one component (defaults to the harness's first declared component)."""
    component = next((c for c in harness.component_tree if c.id == component_id), harness.component_tree[0])
    file_sections = _render_files(repo_root, component.files)
    mistakes_text = "\n".join(f"- {m}" for m in mistakes)
    prompt = (
        f"## Repeated mistakes this episode\n{mistakes_text}\n\n"
        f"## Allowlisted files you may edit\n{', '.join(component.files)}\n\n"
        f"## Current source\n{file_sections}\n\nPropose a scoped fix for this episode."
    )
    text = await llm_call(_task_system_prompt(component), prompt)
    if not text:
        return SelfEditPlan(summary="skipped: dry-run (no LLM credentials configured)",
                             edits=[], parse_error="dry_run")
    return _build_plan(text, component.files)
