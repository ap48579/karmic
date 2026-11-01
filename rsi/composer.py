"""Composer -- decides accept/reject and owns the lineage branch + generation
log, for any HarnessSpec. This is the one generic engine: self_healing_ai's
own coding harness, its research harness, and any third-party consumer's
own exported strategy surface are all just different (HarnessSpec,
llm_call) pairs passed into the same run_cycle() -- nothing here is
specific to either harness, or to this repo.

Safety model:
  - `main` is never touched. All compounding happens on
    `rsi/lineage-<harness.name>`, a separate branch -- you decide if/when
    to merge it.
  - Edits only ever land in a throwaway git worktree, never in this
    process's own working directory, and never in the files currently
    being imported to run this very cycle.
  - Acceptance requires both: the harness's sanity check still passing
    (HarnessSpec.sanity_fn; this repo's Cortex suite by default), AND a
    strictly better score on the harness's own benchmark than the current
    lineage tip. No human judgment call and no self-reported "this is
    better."
  - Every cycle (accepted, rejected, or skipped) is appended to
    rsi_data/<harness.name>/generations.jsonl with the diff saved
    alongside it, so the whole lineage is auditable and any single
    generation is revertible via plain git.

Task-level composition has nothing to call yet (probe.py's task probe is
an explicit stub), so there is no task-level accept path here either.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .harness_spec import HarnessSpec
from .llm_adapters import LLMCall
from .probe import add_worktree, apply_edits, harnessing_probe, remove_worktree
from .proposer import SelfEditPlan, propose_from_report

DATA_DIR = "rsi_data"


def lineage_branch(harness: HarnessSpec) -> str:
    return f"rsi/lineage-{harness.name}"


@dataclass
class CycleResult:
    generation: int
    status: str  # accepted | rejected | skipped
    summary: str
    baseline_score: float | None = None
    candidate_score: float | None = None
    detail: str = ""
    applied_edits: list[str] = field(default_factory=list)


def _run_git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {cwd}:\n{proc.stderr}")
    return proc


def _data_dir(repo_root: Path, harness: HarnessSpec) -> Path:
    d = repo_root / DATA_DIR / harness.name
    (d / "proposals").mkdir(parents=True, exist_ok=True)
    (d / "worktrees").mkdir(parents=True, exist_ok=True)
    return d


def _generations_log(repo_root: Path, harness: HarnessSpec) -> Path:
    return _data_dir(repo_root, harness) / "generations.jsonl"


def _next_generation(repo_root: Path, harness: HarnessSpec) -> int:
    log = _generations_log(repo_root, harness)
    if not log.exists():
        return 1
    with open(log) as f:
        return sum(1 for line in f if line.strip()) + 1


def _append_generation(repo_root: Path, harness: HarnessSpec, record: dict) -> None:
    with open(_generations_log(repo_root, harness), "a") as f:
        f.write(json.dumps(record) + "\n")


def _lineage_history(repo_root: Path, harness: HarnessSpec, n: int = 3) -> str:
    log = _generations_log(repo_root, harness)
    if not log.exists():
        return "(no prior self-improvement cycles)"
    records = [json.loads(line) for line in open(log) if line.strip()]
    recent = records[-n:]
    if not recent:
        return "(no prior self-improvement cycles)"
    return "\n".join(
        f"gen {r['generation']} [{r['status']}]: {r['summary']} "
        f"(baseline={r.get('baseline_score')}, candidate={r.get('candidate_score')})"
        for r in recent
    )


def _ensure_lineage_branch(repo_root: Path, harness: HarnessSpec) -> None:
    branch = lineage_branch(harness)
    if _run_git(["rev-parse", "--verify", branch], repo_root, check=False).returncode != 0:
        _run_git(["branch", branch, "HEAD"], repo_root)


async def run_cycle(
    repo_root: Path, harness: HarnessSpec, llm_call: LLMCall, cortex_dir: str = "./cortex_data",
    python_bin: str = "python3",
) -> CycleResult:
    """One Propose -> Probe -> Compose step, harnessing-level. Never touches
    `main` -- all compounding happens on this harness's own lineage branch;
    the caller decides if/when to merge it."""
    from .tracer import build_report

    repo_root = repo_root.resolve()
    branch = lineage_branch(harness)
    _ensure_lineage_branch(repo_root, harness)
    gen_n = _next_generation(repo_root, harness)
    data_dir = _data_dir(repo_root, harness)

    report = build_report(cortex_dir, domain=harness.cortex_domain)
    lineage_history = _lineage_history(repo_root, harness)
    plan: SelfEditPlan = await propose_from_report(
        repo_root, harness, llm_call, report.to_prompt_text(), lineage_history
    )

    def finish(status: str, summary: str, **kw) -> CycleResult:
        _append_generation(repo_root, harness, {"generation": gen_n, "timestamp": time.time(),
                                                  "status": status, "summary": summary, **kw})
        return CycleResult(generation=gen_n, status=status, summary=summary,
                            baseline_score=kw.get("baseline_score"), candidate_score=kw.get("candidate_score"),
                            detail=kw.get("detail", ""), applied_edits=kw.get("applied_edits", []))

    if not plan.edits:
        return finish("skipped", plan.summary or "no edits proposed", detail=plan.parse_error or "")

    baseline_wt = data_dir / "worktrees" / f"gen{gen_n}_baseline"
    candidate_branch = f"rsi/{harness.name}-gen{gen_n}-candidate"
    candidate_wt = data_dir / "worktrees" / f"gen{gen_n}_candidate"
    keep_candidate_branch = False

    try:
        add_worktree(repo_root, baseline_wt, branch, branch, new_branch=False)
        add_worktree(repo_root, candidate_wt, candidate_branch, branch, new_branch=True)

        applied, skipped = apply_edits(candidate_wt, plan.edits)
        if not applied:
            return finish("rejected", plan.summary, detail="no edits applied: " + "; ".join(skipped))

        _run_git(["add", *applied], candidate_wt)
        commit_msg = f"rsi[{harness.name}] gen{gen_n}: {plan.summary}\n\n" + "\n".join(
            f"- {e.file}: {e.rationale}" for e in plan.edits if e.file in applied
        )
        _run_git(["commit", "-m", commit_msg], candidate_wt)
        patch_path = data_dir / "proposals" / f"gen{gen_n}.patch"
        patch_path.write_text(_run_git(["show", "HEAD"], candidate_wt).stdout)

        result = harnessing_probe(baseline_wt, candidate_wt, python_bin, harness.benchmark_fn, harness.sanity_fn)
        if not result.tests_ok:
            return finish("rejected", plan.summary, applied_edits=applied,
                           detail=f"test suite failed:\n{result.test_output}")

        # Must drop the baseline worktree before force-updating the lineage branch --
        # git refuses to move a branch ref that's checked out in another worktree.
        remove_worktree(repo_root, baseline_wt)

        if result.candidate_score.scalar_score > result.baseline_score.scalar_score:
            _run_git(["branch", "-f", branch, candidate_branch], repo_root)
            keep_candidate_branch = True
            return finish(
                "accepted", plan.summary, applied_edits=applied,
                baseline_score=result.baseline_score.scalar_score, candidate_score=result.candidate_score.scalar_score,
                detail=f"baseline:\n{result.baseline_score.to_prompt_text()}\n"
                       f"candidate:\n{result.candidate_score.to_prompt_text()}",
            )
        return finish(
            "rejected", plan.summary, applied_edits=applied,
            baseline_score=result.baseline_score.scalar_score, candidate_score=result.candidate_score.scalar_score,
            detail="no improvement over lineage tip",
        )
    finally:
        remove_worktree(repo_root, baseline_wt)
        remove_worktree(repo_root, candidate_wt)
        if not keep_candidate_branch:
            _run_git(["branch", "-D", candidate_branch], repo_root, check=False)
