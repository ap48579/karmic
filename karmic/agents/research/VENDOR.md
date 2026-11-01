# Vendored: gpt_researcher

- **Source**: https://github.com/assafelovic/gpt-researcher
- **Pinned tag**: `v3.7.0`
- **Pinned commit**: `0957c301ed06c2a5857b834358c7227c739041d4`
- **Date vendored**: 2026-10-28
- **License**: Apache License 2.0 — full text in `gpt_researcher_LICENSE` in
  this directory, unchanged from upstream.
- **What was vendored**: only the `gpt_researcher/` Python package itself
  (the installable library), not the repo's `backend/` FastAPI server,
  `multi_agents/` LangGraph variant, frontend, docs, or Dockerfiles — per
  `docs/01_harness_spec.md` §5 open question 3. `gpt_researcher_requirements.txt`
  is the upstream package's own dependency list, kept for reference; see
  this project's root `requirements.txt` for what's actually installed
  (a trimmed subset — fastapi/uvicorn/mcp/langgraph are upstream monorepo
  dependencies not imported anywhere inside the vendored `gpt_researcher/`
  package itself, confirmed by grepping its actual imports).

## Edits applied on top of upstream

None yet. Any future edit (manual, or via the RSI engine per
`docs/05_rsi_harness2_design_spec.md`) should be recorded here with a short
note and the commit that made it, so `git log --follow karmic/agents/research/gpt_researcher/<file>`
plus this file together tell the full story of how our copy has diverged
from `v3.7.0`.

## Re-vendoring / updating the pin

To pick up a newer upstream release: re-clone at the new tag, diff against
this directory, and re-apply any edits listed above that upstream hasn't
since incorporated. Don't blindly overwrite — anything the RSI engine has
adopted into the research lineage branch needs to survive the update or be
deliberately re-evaluated against the new upstream version.
