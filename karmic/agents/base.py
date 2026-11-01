"""Agent base wrapper — Claude API calls with per-agent thinking budgets.

Per coding_agent_plan.md's "Thinking Budget Per Agent" table:
    Context Builder: low | Planner: high | Coder: medium->high
    Tester: none (mechanical) | Debugger: medium->escalates | Reviewer: medium

Falls back to a DryRunClient when ANTHROPIC_API_KEY isn't set, so the whole
pipeline's control flow and Cortex integration can be verified without a key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

MODEL = "claude-opus-4-8"

THINKING_BUDGET_TO_EFFORT = {
    "none": None,
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "max",
}


@dataclass
class AgentResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    dry_run: bool = False
    sources: list[dict] = field(default_factory=list)  # [{"url": ..., "title": ...}] from server-side tool use


class DryRunClient:
    """Stands in for the Anthropic client when no API key is configured.
    Lets the pipeline's wiring be verified end-to-end without live calls."""

    def __init__(self, canned_responses: dict[str, str] | None = None):
        self.canned = canned_responses or {}
        self.call_log: list[dict] = []

    def call(self, agent_name: str, system_prompt: str, user_message: str, effort: str | None,
              tools: list[dict] | None = None) -> AgentResult:
        self.call_log.append({
            "agent": agent_name,
            "effort": effort,
            "system_prompt_len": len(system_prompt),
            "user_message_len": len(user_message),
            "tools": [t.get("type") for t in (tools or [])],
        })
        text = self.canned.get(agent_name, f"[DRY RUN — no ANTHROPIC_API_KEY set. "
                                            f"{agent_name} would respond here based on:\n{user_message[:200]}...]")
        return AgentResult(text=text, dry_run=True)


class LiveClient:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()

    def call(self, agent_name: str, system_prompt: str, user_message: str, effort: str | None,
              tools: list[dict] | None = None) -> AgentResult:
        kwargs: dict = {}
        if effort is not None:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": effort}
        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=8000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            **kwargs,
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        sources = []
        for block in response.content:
            if block.type == "web_search_tool_result":
                for item in getattr(block, "content", []) or []:
                    url = getattr(item, "url", None)
                    if url:
                        sources.append({"url": url, "title": getattr(item, "title", "")})
        usage = response.usage
        return AgentResult(
            text=text,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            thinking_tokens=0,  # not separately reported by the API
            sources=sources,
        )


def make_client():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return LiveClient()
    return DryRunClient()


@dataclass
class Agent:
    name: str
    system_prompt: str
    thinking_budget: str = "medium"  # none|low|medium|high|max
    tools: list[dict] | None = None  # server-side tools, e.g. web_search_20250305
    client: object = field(default_factory=make_client)

    def run(self, user_message: str) -> AgentResult:
        effort = THINKING_BUDGET_TO_EFFORT[self.thinking_budget]
        return self.client.call(self.name, self.system_prompt, user_message, effort, tools=self.tools)
