"""Fixed eval suite for the research harness's harnessing-level gate --
grades by whether the final report actually contains the required facts,
not by self-report. Same empirical-gate philosophy as the coding harness's
exact-stdout-match benchmark: ground truth, kept deliberately even though
Harness² itself doesn't require one (docs/03_integration_spec.md S4).

Shells out to the real `cli.py --research` path built in Spec 3/4, which
inherits whatever QWEN_* env vars are already set in this process -- a
worktree's cli.py is a different source tree, so this has to be a real
subprocess, not an in-process call.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

TIMEOUT_SECONDS = 600  # local models are slow; each task is a full research episode


@dataclass
class ResearchTask:
    name: str
    question: str
    required_keywords: list[str]


RESEARCH_TASKS: list[ResearchTask] = [
    ResearchTask("chernobyl_year",
                 "In what year did the Chernobyl nuclear disaster occur, and in which country (at the time)?",
                 ["1986", "Soviet"]),
    ResearchTask("versailles_treaty",
                 "In what year was the Treaty of Versailles signed, and which conference produced it?",
                 ["1919", "Paris Peace"]),
    ResearchTask("www_inventor",
                 "Who is credited with inventing the World Wide Web, and in what year did they propose it?",
                 ["Berners-Lee", "1989"]),
    ResearchTask("python_creator",
                 "Who created the Python programming language, and in what year was it first released?",
                 ["Guido van Rossum", "1991"]),
    ResearchTask("boiling_point",
                 "What is the boiling point of water in Celsius at sea level under standard atmospheric pressure?",
                 ["100"]),
]

_OUTCOME_RE = re.compile(r"Outcome:\s*(\w+)")
_REVISIONS_RE = re.compile(r"Revisions:\s*(\d+)")
_COST_RE = re.compile(r"Cost:\s*(\d+) in / (\d+) out tokens")


@dataclass
class TaskRun:
    name: str
    outcome: str | None
    revisions: int
    cost_tokens: int
    correct: bool
    notes: str = ""


@dataclass
class BenchmarkScore:
    task_runs: list[TaskRun] = field(default_factory=list)
    success_rate: float = 0.0
    avg_revisions: float = 0.0
    total_cost_tokens: int = 0
    scalar_score: float = 0.0

    def to_prompt_text(self) -> str:
        lines = [f"scalar_score={self.scalar_score:.1f} success_rate={self.success_rate:.0%} "
                 f"avg_revisions={self.avg_revisions:.2f} total_cost_tokens={self.total_cost_tokens}"]
        for r in self.task_runs:
            lines.append(f"  - {r.name}: correct={r.correct} outcome={r.outcome} "
                          f"revisions={r.revisions} {r.notes}")
        return "\n".join(lines)


def _run_one_task(repo_root: Path, task: ResearchTask, python_bin: str, scope: str) -> TaskRun:
    with tempfile.TemporaryDirectory(prefix="rsi_research_ws_") as ws, \
         tempfile.TemporaryDirectory(prefix="rsi_research_cortex_") as cortex_dir:
        try:
            proc = subprocess.run(
                [python_bin, "cli.py", "--research", task.question,
                 "--scope", scope, "--workspace-dir", ws, "--cortex-dir", cortex_dir],
                capture_output=True, text=True, timeout=TIMEOUT_SECONDS, cwd=str(repo_root),
            )
        except subprocess.TimeoutExpired:
            return TaskRun(task.name, None, 0, 0, False, notes="pipeline timed out")

        stdout = proc.stdout
        outcome_m, rev_m, cost_m = _OUTCOME_RE.search(stdout), _REVISIONS_RE.search(stdout), _COST_RE.search(stdout)
        outcome = outcome_m.group(1) if outcome_m else None
        revisions = int(rev_m.group(1)) if rev_m else 0
        cost = (int(cost_m.group(1)) + int(cost_m.group(2))) if cost_m else 0

        correct = outcome is not None and all(kw.lower() in stdout.lower() for kw in task.required_keywords)
        notes = "" if correct else "missing required keyword(s) in report"
        return TaskRun(task.name, outcome, revisions, cost, correct, notes=notes)


def run_benchmark(repo_root: Path, python_bin: str = "python3", scope: str = "rsi_eval") -> BenchmarkScore:
    runs = [_run_one_task(repo_root, t, python_bin, scope) for t in RESEARCH_TASKS]
    n = len(runs)
    success_rate = sum(1 for r in runs if r.correct) / n if n else 0.0
    avg_revisions = sum(r.revisions for r in runs) / n if n else 0.0
    total_cost = sum(r.cost_tokens for r in runs)
    scalar = success_rate * 100 - avg_revisions * 5
    return BenchmarkScore(task_runs=runs, success_rate=success_rate, avg_revisions=avg_revisions,
                           total_cost_tokens=total_cost, scalar_score=scalar)
