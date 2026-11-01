"""Ready-made llm_call adapters for the proposer -- the engine itself
(proposer.py / probe.py / composer.py) doesn't know or care how a consumer
talks to their LLM; it just awaits whatever `llm_call(system, user) -> str`
coroutine it's given, and treats an empty string as the dry-run signal.

These two adapters are what Karmic's own coding and research
harnesses pass in -- proving the interface works for two genuinely
different providers (Anthropic's Messages API vs. any Ollama/DashScope
model via an OpenAI-compatible endpoint) before any other consumer touches
it. Writing a third one for a different provider means implementing this
same two-argument-in, string-out shape -- nothing elsewhere needs to change.
"""

from __future__ import annotations

import os
from typing import Awaitable, Callable

LLMCall = Callable[[str, str], Awaitable[str]]


def qwen_ollama_llm_call(tier: str = "STRATEGIC_LLM") -> LLMCall:
    """Reads FAST_LLM/SMART_LLM/STRATEGIC_LLM (set once here via
    agents.research.qwen_config.configure_qwen()) and calls through
    GenericLLMProvider -- same pattern as agents/research/reflection.py.
    The underlying call returns "" (the engine's dry-run signal) if no
    QWEN_API_KEY/QWEN_BASE_URL was configured.
    """
    from ..agents import research  # side effect: puts gpt_researcher on sys.path

    research.configure_qwen()  # called once at adapter construction, not per-request

    async def _call(system: str, user: str) -> str:
        from gpt_researcher.llm_provider.generic import GenericLLMProvider

        llm_str = os.environ.get(tier)
        if not llm_str:
            return ""
        provider, model = llm_str.split(":", 1)
        llm = GenericLLMProvider.from_provider(provider, model=model, temperature=0.3, verbose=False)
        return await llm.get_chat_response(
            [{"role": "system", "content": system}, {"role": "user", "content": user}], stream=False,
        )

    return _call


def ollama_json_call(model: str, base_url: str = "http://localhost:11434", num_ctx: int = 16384,
                     temperature: float = 0.3, timeout: float = 900) -> LLMCall:
    """Direct Ollama /api/chat with `format="json"` -- no GPT Researcher / LangChain, so it
    works on the core install (docs/09_agent_archetypes_spec.md S10). Returns the raw JSON
    text, which proposer.py parses; "" (the dry-run signal) if Ollama isn't reachable."""
    async def _call(system: str, user: str) -> str:
        import httpx

        body = {"model": model, "stream": False, "format": "json",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": temperature, "num_ctx": num_ctx}}
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{base_url.rstrip('/')}/api/chat", json=body)
                resp.raise_for_status()
        except httpx.HTTPError:
            return ""
        return resp.json().get("message", {}).get("content") or ""

    return _call


def claude_llm_call(thinking_budget: str = "max") -> LLMCall:
    """Wraps agents.base.Agent (Claude via the Anthropic SDK, or
    DryRunClient if ANTHROPIC_API_KEY isn't set) -- the pattern
    selfmod/meta_agent.py used before this engine was unified.
    """
    async def _call(system: str, user: str) -> str:
        from ..agents.base import Agent

        agent = Agent(name="self_improvement_planner", system_prompt=system, thinking_budget=thinking_budget)
        result = agent.run(user)  # synchronous call -- fine here, this is a single-cycle CLI tool, not a server
        return "" if result.dry_run else result.text

    return _call
