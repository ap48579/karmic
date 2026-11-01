"""A HarnessSpec is the one thing you need to supply to point the generic
self-improvement engine at a harness -- any harness, in any repo, calling
any LLM. This is the seam that lets the same engine serve self_healing_ai's
own coding harness, its research harness, and -- once exported via
agents.research.scaffold.export_strategy() -- any third-party consumer's
own copy of the research strategy surface, each with its own repo, its own
git lineage, and its own benchmark.

Component groups let a HarnessSpec organize its editable files by role
(Harness²'s c1-c8 taxonomy, docs/05_rsi_harness2_design_spec.md) -- this is
optional structure, not a forced one. A consumer who doesn't care just
declares a single Component holding every editable file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol


class ScoreLike(Protocol):
    scalar_score: float

    def to_prompt_text(self) -> str: ...


BenchmarkFn = Callable[[Path, str], ScoreLike]
# (worktree, python_bin) -> (ok, output tail). Run on the candidate before any benchmark;
# a consumer repo points this at its own test subset (docs/09_agent_archetypes_spec.md S7).
SanityFn = Callable[[Path, str], tuple[bool, str]]


@dataclass(frozen=True)
class Component:
    id: str
    name: str
    files: list[str]
    description: str = ""


@dataclass(frozen=True)
class HarnessSpec:
    name: str  # used as the lineage branch suffix and the on-disk data subdir
    description: str  # shown to the proposer for framing
    component_tree: list[Component]  # the editable surface, optionally grouped by role
    benchmark_fn: BenchmarkFn  # (repo_root, python_bin) -> ScoreLike, run inside a worktree
    cortex_domain: str  # Episode.domain to filter traces by when building the report
    sanity_fn: SanityFn | None = None  # None = this repo's Cortex suite (probe.run_sanity_suite)

    @property
    def allowed_files(self) -> list[str]:
        return [f for c in self.component_tree for f in c.files]
