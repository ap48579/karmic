"""Translates our own QWEN_* env vars into the env vars the vendored
`gpt_researcher` package actually reads for model selection.

Confirmed against vendor/gpt_researcher/config/config.py (FAST_LLM/SMART_LLM/
STRATEGIC_LLM as "<provider>:<model>" strings, split on ":") and
vendor/gpt_researcher/llm_provider/generic/base.py (the "dashscope" and
"vllm_openai" provider branches) -- not assumed from the README's generic
OPENAI_BASE_URL mention, which turned out to be unnecessary: GPT Researcher
has native Qwen support via Alibaba's DashScope.

Two modes, auto-selected by which of our own env vars are set:
  - QWEN_BASE_URL set  -> self-hosted/any OpenAI-compatible Qwen endpoint
    (vLLM, Ollama, etc.), mapped to gpt_researcher's "vllm_openai" provider.
  - QWEN_BASE_URL unset, QWEN_API_KEY set -> Alibaba Cloud DashScope's
    hosted Qwen API, mapped to gpt_researcher's "dashscope" provider.
"""

from __future__ import annotations

import os


def configure_qwen(model: str | None = None) -> str:
    """Sets FAST_LLM/SMART_LLM/STRATEGIC_LLM and the matching provider
    API key/base env var from our own QWEN_* vars.

    Must be called before constructing Config/GPTResearcher -- gpt_researcher
    reads these env vars at construction time, not lazily.

    Returns the provider name actually selected ("vllm_openai" or
    "dashscope"), or "dry_run" if QWEN_API_KEY isn't set at all -- callers
    should treat "dry_run" as "the vendored tree will fail on its first real
    LLM call", the same signal DryRunProvider gave in the earlier
    from-scratch harness design (now dropped, but the signal is still
    useful for a no-key dry pass through our own wrapper code).
    """
    api_key = os.environ.get("QWEN_API_KEY")
    base_url = os.environ.get("QWEN_BASE_URL")
    model = model or os.environ.get("QWEN_MODEL", "qwen-plus")

    if not api_key:
        return "dry_run"

    if base_url:
        provider = "vllm_openai"
        os.environ["VLLM_OPENAI_API_KEY"] = api_key
        os.environ["VLLM_OPENAI_API_BASE"] = base_url
    else:
        provider = "dashscope"
        os.environ["DASHSCOPE_API_KEY"] = api_key

    llm_str = f"{provider}:{model}"
    os.environ["FAST_LLM"] = llm_str
    os.environ["SMART_LLM"] = llm_str
    os.environ["STRATEGIC_LLM"] = llm_str
    return provider
