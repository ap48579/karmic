"""Structured-output model calls for archetypes. A JsonLLM is
`(system, messages, schema, temperature) -> (parsed dict, meta)` and
raises LLMError on transport failure or unparseable output -- the runtime
turns that into a recorded problem, never an exception for the consumer.

`messages` rather than a single user string because the self-healing
loop re-asks in the same conversation: the model sees its own previous
answer followed by the validator's complaints (the reask pattern from
jxnl/instructor, with deterministic validators in place of pydantic ones).

ollama_json() talks to /api/chat directly with `format=<json schema>`,
so a light install (no GPT Researcher / LangChain) can drive a local model.
"""

from __future__ import annotations

import json
import time
from typing import Callable

JsonLLM = Callable[[str, list[dict], dict, float], tuple[dict, dict]]


class LLMError(Exception):
    pass


def ollama_json(model: str, base_url: str = "http://localhost:11434", num_ctx: int = 16384,
                timeout: float = 600, think: bool | None = None) -> JsonLLM:
    import httpx

    def _call(system: str, messages: list[dict], schema: dict, temperature: float) -> tuple[dict, dict]:
        started = time.monotonic()
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "format": schema,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        if think is not None:
            body["think"] = think
        try:
            resp = httpx.post(f"{base_url.rstrip('/')}/api/chat", json=body, timeout=timeout)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc
        data = resp.json()
        content = data.get("message", {}).get("content") or ""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"unparseable model output: {content[:200]}") from exc
        return parsed, {"model": model, "seconds": round(time.monotonic() - started, 1),
                        "input_tokens": data.get("prompt_eval_count", 0), "output_tokens": data.get("eval_count", 0)}

    return _call


def anthropic_json(model: str = "claude-sonnet-5-5", max_tokens: int = 8000) -> JsonLLM:
    """Forced tool_use, so the schema is enforced server-side. Needs the [claude] extra."""

    def _call(system: str, messages: list[dict], schema: dict, temperature: float) -> tuple[dict, dict]:
        import anthropic

        started = time.monotonic()
        try:
            msg = anthropic.Anthropic().messages.create(
                model=model, max_tokens=max_tokens, system=system, temperature=temperature,
                tools=[{"name": "submit", "description": "Submit the structured result.", "input_schema": schema}],
                tool_choice={"type": "tool", "name": "submit"},
                messages=messages,
            )
        except anthropic.APIError as exc:
            raise LLMError(f"anthropic request failed: {exc}") from exc
        for block in msg.content:
            if block.type == "tool_use":
                return block.input, {"model": model, "seconds": round(time.monotonic() - started, 1),
                                     "input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
        raise LLMError("anthropic response contained no tool_use block")

    return _call


class ScriptedLLM:
    """Test double: returns queued outputs in order (an Exception instance is raised instead),
    and records every call so tests can assert on what the heal loop sent back."""

    def __init__(self, outputs: list):
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def __call__(self, system: str, messages: list[dict], schema: dict, temperature: float) -> tuple[dict, dict]:
        self.calls.append({"system": system, "messages": [dict(m) for m in messages]})
        if not self.outputs:
            raise LLMError("ScriptedLLM ran out of outputs")
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out, {"model": "scripted", "seconds": 0.0}
