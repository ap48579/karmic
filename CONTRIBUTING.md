# Contributing

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Running tests

```bash
python -m tests.test_cortex            # no API key needed, covers karmic/cortex/ in isolation
python -m tests.test_general_agents    # no model needed, covers karmic/agents/general/ via ScriptedLLM
python -m tests.test_coding_pipeline   # no API key needed, covers the coding pipeline's agents
python -m tests.test_rsi               # no API key needed, covers RSI harness wiring
```

Beyond these, `karmic/rsi/` and the end-to-end pipelines are exercised via
dry-run mode (omit `ANTHROPIC_API_KEY`/`QWEN_API_KEY` and the pipelines
run with placeholder LLM responses,
exercising the real control flow) and the RSI engine's own benchmark-gated
mechanics tests (see `docs/05_rsi_harness2_design_spec.md` for how those
were validated during development).

## The fixed-substrate vs. strategy-surface boundary

This is the one thing to understand before touching `karmic/agents/research/`:

- **Fixed substrate** — `karmic/agents/research/gpt_researcher/`'s infrastructure
  (`llm_provider/`, `retrievers/`, `scraper/`, `document/`, `vector_store/`,
  etc.), `karmic/agents/research/policies.py` (the hard guardrail layer),
  `karmic/agents/base.py`, **all of `karmic/cortex/`**, and **all of `karmic/agents/general/`**
  (an archetype's benchmark is built from the same validators that guard
  production calls, so an edit must never be able to change the checks
  it's graded by; consumers' role *prompts* are the editable surface, and
  they live in the consumer's own repo). Never proposed as an RSI
  edit, by construction — these files aren't in any
  `HarnessSpec.allowed_files`, for any harness. `karmic/cortex/` specifically is
  excluded from *every* harness's component tree on purpose, not just left
  off by oversight: it's shared across every harness at once, so an edit
  benchmarked against only one harness's benchmark could silently change
  memory behavior for the others with nothing to catch it. If you're
  adding a new retriever, fixing a vendoring bug, or tuning Cortex's decay/
  reinforcement constants, you're working here, and that's a normal PR —
  just not something RSI proposes on its own.
- **Strategy surface** — `karmic/agents/research/gpt_researcher/prompts.py`,
  `query_processing.py`, `researcher.py`, `writer.py`, `curator.py`,
  `report_generation.py`, `agent_creator.py`, `retriever.py`,
  `web_scraping.py`, plus `karmic/research_pipeline.py`'s orchestration
  constants. These are what `karmic/rsi/` is allowed to edit. A PR touching these
  files for this repo's own harness is fine too — just know that any
  third-party consumer who's exported their own copy (`karmic.agents.
  research.scaffold.export_strategy()`) won't see your change until they re-export
  or manually port it; see `karmic/agents/research/VENDOR.md` for the update
  process.

Don't add a new file to the strategy surface list (`karmic/rsi/harnesses.py`'s
`RESEARCH_HARNESS.component_tree`) without a real reason RSI should be
allowed to edit it — the allowlist is deliberately small.

## Updating the vendored GPT Researcher pin

See `karmic/agents/research/VENDOR.md`. Re-clone at the new tag, diff against the
current vendored tree, re-apply any edits listed there that upstream
hasn't since incorporated, and update the pinned commit/tag in that file.

## Git safety for RSI's own branches

`rsi/lineage-<harness>` branches are the self-improvement engine's own
accepted-edit history. Don't force-push over them or rebase them away —
treat them the same as any other branch with real history you'd want to
keep, even though nothing merges them into `main` automatically.
