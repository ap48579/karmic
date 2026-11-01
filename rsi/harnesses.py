"""Registry of HarnessSpecs this repo's own CLI drives -- the first two
consumers of the generic engine, proving it works for two genuinely
different LLM providers (Claude vs. any Ollama/DashScope model) and two
genuinely different file sets before any third-party consumer touches it.

To add a third: export your own copy of the editable surface into your own
repo (agents.research.scaffold.export_strategy() for the research
harness's shape, or just point at your own files for anything else), write
your own ground-truth benchmark, build your own HarnessSpec and llm_call,
and call rsi.run_cycle(). Nothing in proposer.py, probe.py, or composer.py
needs to change.
"""

from __future__ import annotations

from . import coding_benchmark, research_benchmark
from .harness_spec import Component, HarnessSpec
from .llm_adapters import claude_llm_call, qwen_ollama_llm_call

CODING_HARNESS = HarnessSpec(
    name="coding",
    description="a self-healing coding agent (Planner/Coder/Tester/Debugger/Reviewer)",
    component_tree=[
        Component(
            id="c1", name="agents",
            files=["agents/pipeline_agents.py", "pipeline.py"],
            description="Planner/Coder/Debugger/Reviewer prompts and the orchestration constants "
                        "(MAX_DEBUGGER_ATTEMPTS).",
        ),
    ],
    benchmark_fn=coding_benchmark.run_benchmark,
    cortex_domain="coding",
)
# cortex/ is deliberately absent from every harness's component_tree, for both harnesses
# here and any third-party one -- it's shared, fixed substrate (same role as
# agents/base.py / llm_provider/), never RSI-editable. Cortex is used by every harness
# at once; an edit benchmarked against only one harness's benchmark could silently
# change memory behavior for all the others with nothing to catch it -- same class of
# cross-consumer blast-radius problem a consumer integration surfaced for the vendored
# harness files, so the fix is the same: don't let it be edited at all.

RESEARCH_HARNESS = HarnessSpec(
    name="research",
    description=(
        "an autonomous deep-research agent (Planner/Investigator/Synthesizer/Grader/Reviser) "
        "that answers research questions via web search, cites sources, and revises itself "
        "when its own Grader finds unsupported claims or missed sub-questions"
    ),
    component_tree=[
        Component(
            id="c1", name="system_prompt",
            files=["agents/research/gpt_researcher/prompts.py",
                   "agents/research/gpt_researcher/actions/agent_creator.py"],
            description="Prompt templates and the dynamic per-query persona. Guardrail instructions "
                        "(soft) live here too -- the hard guardrail layer (agents/research/policies.py) "
                        "is never RSI-editable.",
        ),
        Component(
            id="c5", name="tool_guidance",
            files=["agents/research/gpt_researcher/actions/retriever.py",
                   "agents/research/gpt_researcher/actions/web_scraping.py"],
            description="How retriever/scraper results get used -- not the fetch mechanics themselves.",
        ),
        Component(
            id="c6", name="subagents",
            files=["agents/research/gpt_researcher/skills/researcher.py",
                   "agents/research/gpt_researcher/skills/writer.py",
                   "agents/research/gpt_researcher/skills/curator.py"],
            description="The investigate/write/curate role implementations.",
        ),
        Component(
            id="c7", name="scripts",
            files=["agents/research/gpt_researcher/actions/query_processing.py",
                   "agents/research/gpt_researcher/actions/report_generation.py",
                   "research_pipeline.py"],
            description="Sub-question generation, report-writing glue, and our own orchestration "
                        "constants (MAX_REVISIONS).",
        ),
    ],
    benchmark_fn=research_benchmark.run_benchmark,
    cortex_domain="research",
)

HARNESSES = {h.name: h for h in (CODING_HARNESS, RESEARCH_HARNESS)}

# Factories, not already-constructed callables -- constructing qwen_ollama_llm_call()
# has the side effect of calling configure_qwen() once, so build it lazily, only for
# whichever harness is actually being run.
LLM_CALL_FACTORIES = {
    "coding": claude_llm_call,
    "research": qwen_ollama_llm_call,
}
