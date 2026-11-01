"""Fixed eval suite — the thing that actually proves a self-edit is better,
instead of trusting the editing model's own opinion of its rewrite.

Each task is graded by exact stdout match, independent of the pipeline's own
(weak — exit-code-only) Tester. Scoring runs the *worktree's* cli.py via
subprocess, so it genuinely exercises whatever source the candidate generation
modified, not whatever happens to be already imported in this process.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

TIMEOUT_SECONDS = 180


@dataclass
class BenchmarkTask:
    name: str
    prompt: str
    expected_stdout: str


BENCHMARK_TASKS: list[BenchmarkTask] = [
    BenchmarkTask(
        name="fibonacci_10",
        prompt=(
            "Write a Python script with no input that computes the 10th Fibonacci "
            "number using the convention fib(0)=0, fib(1)=1, and prints just the "
            "integer result on its own line, nothing else."
        ),
        expected_stdout="55",
    ),
    BenchmarkTask(
        name="sum_first_100",
        prompt=(
            "Write a Python script with no input that prints the sum of the integers "
            "from 1 to 100 inclusive, as a bare integer on its own line, nothing else."
        ),
        expected_stdout="5050",
    ),
    BenchmarkTask(
        name="palindrome_check",
        prompt=(
            "Write a Python script with no input that checks whether the string "
            "'racecar' is a palindrome and prints exactly 'True' or 'False' "
            "(Python bool str()), nothing else."
        ),
        expected_stdout="True",
    ),
    BenchmarkTask(
        name="primes_above_100",
        prompt=(
            "Write a Python script with no input that finds the 5 smallest prime "
            "numbers strictly greater than 100 and prints them space-separated, "
            "ascending, on one line, nothing else."
        ),
        expected_stdout="101 103 107 109 113",
    ),
    BenchmarkTask(
        name="fizzbuzz_15",
        prompt=(
            "Write a Python script with no input that applies standard FizzBuzz "
            "rules to the number 15 (Fizz for multiples of 3, Buzz for multiples of "
            "5, FizzBuzz for multiples of both) and prints only the result for 15, "
            "nothing else."
        ),
        expected_stdout="FizzBuzz",
    ),
]

_OUTCOME_RE = re.compile(r"Outcome:\s*(\w+)")
_ATTEMPTS_RE = re.compile(r"Healer attempts:\s*(\d+)")
_COST_RE = re.compile(r"Cost:\s*(\d+) in / (\d+) out tokens")


@dataclass
class TaskRun:
    name: str
    pipeline_outcome: str | None
    healer_attempts: int
    cost_tokens: int
    correct_output: bool
    notes: str = ""


@dataclass
class BenchmarkScore:
    task_runs: list[TaskRun] = field(default_factory=list)
    success_rate: float = 0.0
    avg_healer_attempts: float = 0.0
    total_cost_tokens: int = 0
    scalar_score: float = 0.0

    def to_prompt_text(self) -> str:
        lines = [f"scalar_score={self.scalar_score:.1f} success_rate={self.success_rate:.0%} "
                 f"avg_healer_attempts={self.avg_healer_attempts:.2f} total_cost_tokens={self.total_cost_tokens}"]
        for r in self.task_runs:
            lines.append(f"  - {r.name}: correct={r.correct_output} outcome={r.pipeline_outcome} "
                          f"attempts={r.healer_attempts} {r.notes}")
        return "\n".join(lines)


def _run_one_task(repo_root: Path, task: BenchmarkTask, python_bin: str, scope: str) -> TaskRun:
    with tempfile.TemporaryDirectory(prefix="selfmod_ws_") as ws, \
         tempfile.TemporaryDirectory(prefix="selfmod_cortex_") as cortex_dir:
        try:
            proc = subprocess.run(
                [python_bin, "cli.py", task.prompt,
                 "--scope", scope, "--workspace-dir", ws, "--cortex-dir", cortex_dir],
                capture_output=True, text=True, timeout=TIMEOUT_SECONDS, cwd=str(repo_root),
            )
        except subprocess.TimeoutExpired:
            return TaskRun(task.name, None, 0, 0, False, notes="pipeline timed out")

        stdout = proc.stdout
        outcome_m = _OUTCOME_RE.search(stdout)
        attempts_m = _ATTEMPTS_RE.search(stdout)
        cost_m = _COST_RE.search(stdout)
        outcome = outcome_m.group(1) if outcome_m else None
        attempts = int(attempts_m.group(1)) if attempts_m else 0
        cost = (int(cost_m.group(1)) + int(cost_m.group(2))) if cost_m else 0

        solution_path = Path(ws) / "solution.py"
        if not solution_path.exists():
            return TaskRun(task.name, outcome, attempts, cost, False, notes="no solution.py produced")

        try:
            run_proc = subprocess.run(
                [python_bin, str(solution_path)],
                capture_output=True, text=True, timeout=10, cwd=ws,
            )
        except subprocess.TimeoutExpired:
            return TaskRun(task.name, outcome, attempts, cost, False, notes="solution.py timed out")

        actual = run_proc.stdout.strip()
        correct = run_proc.returncode == 0 and actual == task.expected_stdout.strip()
        notes = "" if correct else f"got stdout={actual!r} rc={run_proc.returncode}"
        return TaskRun(task.name, outcome, attempts, cost, correct, notes=notes)


def run_benchmark(repo_root: Path, python_bin: str = "python3", scope: str = "selfmod_eval") -> BenchmarkScore:
    runs = [_run_one_task(repo_root, t, python_bin, scope) for t in BENCHMARK_TASKS]
    n = len(runs)
    success_rate = sum(1 for r in runs if r.correct_output) / n if n else 0.0
    avg_attempts = sum(r.healer_attempts for r in runs) / n if n else 0.0
    total_cost = sum(r.cost_tokens for r in runs)
    scalar = success_rate * 100 - avg_attempts * 5
    return BenchmarkScore(
        task_runs=runs, success_rate=success_rate, avg_healer_attempts=avg_attempts,
        total_cost_tokens=total_cost, scalar_score=scalar,
    )
