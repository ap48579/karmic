"""Coding-pipeline agent tests -- no API key required (stub agents). Run with:
    python -m tests.test_coding_pipeline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from karmic.agents.base import AgentResult
from karmic.agents.pipeline_agents import Debugger


class _StubAgent:
    def __init__(self, prompts: list[str]):
        self.prompts = prompts

    def run(self, prompt: str) -> AgentResult:
        self.prompts.append(prompt)
        return AgentResult(text=f"```python\nprint({len(self.prompts)})\n```")


def _debugger_with_stub() -> tuple[Debugger, list[str]]:
    prompts: list[str] = []
    debugger = Debugger()
    debugger._agent_for_effort = lambda effort: _StubAgent(prompts)
    return debugger, prompts


def test_debugger_returns_every_hypothesis():
    # Regression: from attempt 3 the Debugger paid for one fix per hypothesis but returned
    # only the first, so the second was generated and thrown away untested.
    error = {"stderr": "IndexError", "stdout": ""}
    debugger, prompts = _debugger_with_stub()
    candidates, effort = debugger.diagnose_and_fix("x = []\nx[0]", error, attempt_number=0)
    assert candidates == ["print(1)"] and effort == "medium" and "Hint:" not in prompts[0]

    debugger, prompts = _debugger_with_stub()
    candidates, effort = debugger.diagnose_and_fix("x = []\nx[0]", error, attempt_number=2)
    assert candidates == ["print(1)", "print(2)"] and effort == "max"
    assert all("Hint:" in p for p in prompts)
    print("PASS: test_debugger_returns_every_hypothesis")


if __name__ == "__main__":
    test_debugger_returns_every_hypothesis()
    print("\nAll coding pipeline tests passed.")
