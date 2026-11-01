"""Probe -- runs a candidate edit and reports what happened. Never decides
accept/reject itself (that's composer.py).

Harnessing-level: apply the edit in an isolated git worktree, run the
harness's sanity check (HarnessSpec.sanity_fn, defaulting to this repo's
Cortex suite), run the harness's own ground-truth benchmark
(HarnessSpec.benchmark_fn) on baseline vs candidate.

Task-level: deliberately not implemented -- see task_probe_not_implemented().
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .harness_spec import BenchmarkFn, SanityFn, ScoreLike
from .proposer import ProposedEdit


def _run_git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {cwd}:\n{proc.stderr}")
    return proc


def add_worktree(repo_root: Path, path: Path, branch: str, start_point: str, new_branch: bool) -> None:
    if path.exists():
        shutil.rmtree(path)
    args = ["worktree", "add"]
    args += ["-b", branch, str(path), start_point] if new_branch else [str(path), branch]
    _run_git(args, repo_root)


def remove_worktree(repo_root: Path, path: Path) -> None:
    _run_git(["worktree", "remove", "--force", str(path)], repo_root, check=False)
    shutil.rmtree(path, ignore_errors=True)


def apply_edits(worktree: Path, edits: list[ProposedEdit]) -> tuple[list[str], list[str]]:
    applied, skipped = [], []
    for edit in edits:
        target = worktree / edit.file
        if not target.exists():
            skipped.append(f"{edit.file}: file not found in worktree")
            continue
        text = target.read_text()
        count = text.count(edit.old_string)
        if count != 1:
            skipped.append(f"{edit.file}: old_string occurs {count} times (need exactly 1)")
            continue
        target.write_text(text.replace(edit.old_string, edit.new_string, 1))
        applied.append(edit.file)
    return applied, skipped


def run_sanity_suite(worktree: Path, python_bin: str) -> tuple[bool, str]:
    """Default sanity check -- only meaningful for this repo's own harnesses. A consumer
    repo has no tests.test_cortex, so without its own HarnessSpec.sanity_fn every cycle
    there would be rejected as "test suite failed" before its benchmark ever ran."""
    return run_command_sanity([python_bin, "-m", "tests.test_cortex"])(worktree, python_bin)


def run_command_sanity(cmd: list[str], timeout: int = 300) -> SanityFn:
    """SanityFn from a command run in the worktree, e.g. ["pytest", "-q", "tests/unit"]."""
    def _sanity(worktree: Path, python_bin: str) -> tuple[bool, str]:
        try:
            proc = subprocess.run(cmd, cwd=str(worktree), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"sanity command timed out after {timeout}s: {' '.join(cmd)}"
        return proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]
    return _sanity


@dataclass
class HarnessingProbeResult:
    tests_ok: bool
    test_output: str
    baseline_score: ScoreLike | None = None
    candidate_score: ScoreLike | None = None


def harnessing_probe(
    baseline_wt: Path, candidate_wt: Path, python_bin: str, benchmark_fn: BenchmarkFn,
    sanity_fn: SanityFn | None = None,
) -> HarnessingProbeResult:
    """Caller is responsible for creating/tearing down the two worktrees --
    this just applies tests+benchmark to whatever is already staged in each."""
    tests_ok, test_output = (sanity_fn or run_sanity_suite)(candidate_wt, python_bin)
    if not tests_ok:
        return HarnessingProbeResult(tests_ok=False, test_output=test_output)
    baseline_score = benchmark_fn(baseline_wt, python_bin)
    candidate_score = benchmark_fn(candidate_wt, python_bin)
    return HarnessingProbeResult(tests_ok=True, test_output="",
                                  baseline_score=baseline_score, candidate_score=candidate_score)


@dataclass
class TaskProbeResult:
    baseline_mistake_count: int
    candidate_mistake_count: int

    @property
    def candidate_is_better(self) -> bool:
        return self.candidate_mistake_count < self.baseline_mistake_count


def task_probe_not_implemented() -> None:
    """Deliberately not built -- see docs/05_rsi_harness2_design_spec.md S6
    open question 1. A real task-level probe needs to re-run write_report()
    (not just re-grade a fixed draft) with the candidate c1 edit actually
    active in the vendored prompts.py, then grade THAT new draft and
    compare mistake counts against a baseline re-run -- which means
    mutating a live vendored file mid-episode and reverting it, something
    this project hasn't yet worked out how to do safely (a crash between
    mutate and revert would corrupt the harness for every other episode
    using the same checkout). Returning a fabricated comparison here would
    be actively wrong, not just incomplete, so this stays an explicit stub
    until there's a safe mechanism and real mistake-pattern data to design
    the contrast judge against. propose_from_trajectory() (proposer.py)
    works standalone if you want to see what it would propose, without
    anything deciding accept/reject on it yet.
    """
    raise NotImplementedError(
        "task-level probe is intentionally not implemented -- see this function's docstring "
        "and docs/05_rsi_harness2_design_spec.md S6 open question 1"
    )
