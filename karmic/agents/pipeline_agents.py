"""The 6 pipeline agents from coding_agent_plan.md: Planner, Coder, Tester,
Debugger, Reviewer, Learning Agent. Context Builder is a minimal stub here
(real codebase-reading is out of scope for this MVP — see README).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .base import Agent, AgentResult


def extract_code_block(text: str, filename_hint: str = "solution.py") -> tuple[str, str]:
    """Pulls the first fenced code block out of a response. Falls back to
    treating the whole response as code if no fence is found."""
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    code = match.group(1) if match else text
    return filename_hint, code.strip()


class ContextBuilder:
    """Minimal stub: lists the workspace directory. Real Context Builder
    (reading an existing codebase) is the first thing to extend post-MVP."""

    def run(self, workspace: Path) -> str:
        files = [f.name for f in workspace.glob("*") if f.is_file()]
        if not files:
            return "Empty workspace — this is a greenfield task, no existing code to read."
        return f"Existing files in workspace: {', '.join(files)}"


class Planner:
    def __init__(self):
        self.agent = Agent(
            name="planner",
            system_prompt=(
                "You are the Planner agent in a self-healing coding pipeline. "
                "Given a task and prior-memory context (Recall/Intuition/Caution/Instinct), "
                "produce a short, concrete, numbered implementation plan. "
                "Be concrete about what file(s) to write and what they should do. "
                "Do not write code yet — just the plan."
            ),
            thinking_budget="high",
        )

    def run(self, task: str, memory_context: str) -> AgentResult:
        prompt = f"Task: {task}\n\nPrior memory context:\n{memory_context}\n\nProduce the plan."
        return self.agent.run(prompt)


class Coder:
    def __init__(self):
        self.agent = Agent(
            name="coder",
            system_prompt=(
                "You are the Coder agent. Given a plan, write the complete, runnable code "
                "to implement it. Output exactly one fenced Python code block. "
                "The code must run standalone with `python3 <file>` and should print its "
                "result or raise on failure — there is no test framework, just direct execution."
            ),
            thinking_budget="medium",
        )

    def run(self, plan: str, fix_instruction: str | None = None) -> AgentResult:
        prompt = f"Plan:\n{plan}"
        if fix_instruction:
            prompt += f"\n\nA previous attempt failed. Fix it:\n{fix_instruction}"
        return self.agent.run(prompt)


class Tester:
    """Zero thinking budget — purely mechanical. Does not call the LLM at all."""

    def run(self, workspace: Path, filename: str, code: str, timeout: int = 10) -> dict:
        path = workspace / filename
        path.write_text(code)
        try:
            result = subprocess.run(
                ["python3", str(path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(workspace),
            )
            return {
                "passed": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "stdout": "", "stderr": "Timed out", "returncode": -1}


class Debugger:
    """Self-paced healing loop: escalating thinking budget across attempts.
    From attempt 3 on it proposes one fix per hypothesis instead of one fix,
    and the caller tests them in order (sequential calls here -- running
    them concurrently is an upgrade, not a logic change)."""

    ESCALATION = ["medium", "high", "max"]
    HYPOTHESIS_HINTS = ["focus on the literal error message", "consider an off-by-one or type mismatch"]

    def _agent_for_effort(self, effort: str) -> Agent:
        return Agent(
            name="debugger",
            system_prompt=(
                "You are the Debugger agent. Given failing code and its error output, "
                "diagnose the root cause and produce a corrected, complete, runnable version. "
                "Output exactly one fenced Python code block."
            ),
            thinking_budget=effort,
        )

    def diagnose_and_fix(self, code: str, error: dict, attempt_number: int) -> tuple[list[str], str]:
        """Returns (candidate rewrites to test in order, effort used)."""
        effort = self.ESCALATION[min(attempt_number, len(self.ESCALATION) - 1)]
        hints = [None] if attempt_number < 2 else self.HYPOTHESIS_HINTS
        agent = self._agent_for_effort(effort)
        candidates = []
        for hint in hints:
            prompt = (
                f"This code failed:\n```python\n{code}\n```\n\n"
                f"stderr:\n{error['stderr']}\nstdout:\n{error['stdout']}\n\n"
                "Diagnose and fix it." + (f" Hint: {hint}" if hint else "")
            )
            _, fixed_code = extract_code_block(agent.run(prompt).text)
            candidates.append(fixed_code)
        return candidates, effort


class Reviewer:
    def __init__(self):
        self.agent = Agent(
            name="reviewer",
            system_prompt=(
                "You are the Reviewer agent — independent of the Coder, grading the final "
                "code on 4 criteria: Correctness, Robustness, Security, Maintainability. "
                "Give a short verdict per criterion, not just pass/fail."
            ),
            thinking_budget="medium",
        )

    def run(self, task: str, final_code: str) -> AgentResult:
        prompt = f"Task: {task}\n\nFinal code:\n```python\n{final_code}\n```\n\nGrade it."
        return self.agent.run(prompt)


class LearningAgent:
    """Generates abstracted_pattern / failure_summary, then hands off to
    cortex.write_back. Stays thin — Cortex's core logic doesn't know about LLMs."""

    def __init__(self):
        self.agent = Agent(
            name="learning_agent",
            system_prompt=(
                "You distill a completed coding episode into Cortex memory fields. "
                "If it succeeded, produce a short 'abstracted_pattern' label describing "
                "the general approach (not the specific code). If it failed, produce a "
                "short 'failure_summary' describing what was wrongly assumed. "
                "Then produce 1-3 short 'distilled_learnings' bullets. "
                "Respond in exactly this format:\n"
                "PATTERN: <text or NONE>\nFAILURE: <text or NONE>\nLEARNINGS:\n- ...\n- ..."
            ),
            thinking_budget="low",
        )

    def run(self, task: str, outcome: str, summary_of_what_happened: str) -> dict:
        prompt = f"Task: {task}\nOutcome: {outcome}\nWhat happened:\n{summary_of_what_happened}"
        result = self.agent.run(prompt)
        return self._parse(result.text)

    @staticmethod
    def _parse(text: str) -> dict:
        pattern_match = re.search(r"PATTERN:\s*(.+)", text)
        failure_match = re.search(r"FAILURE:\s*(.+)", text)
        learnings = re.findall(r"^- (.+)$", text, re.MULTILINE)

        pattern = pattern_match.group(1).strip() if pattern_match else None
        failure = failure_match.group(1).strip() if failure_match else None
        return {
            "abstracted_pattern": None if pattern in (None, "NONE") else pattern,
            "failure_summary": None if failure in (None, "NONE") else failure,
            "distilled_learnings": learnings,
        }
