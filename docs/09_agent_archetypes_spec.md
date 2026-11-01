# Spec 9 — Agent Archetypes: a Catalog of RSI-Ready Agent Types

Status: **build steps 1–3 done, plus the §7 engine seam** (see §16).
`agents/general/` ships A2–A6, the runtime with its heal loop, validators,
fixtures and templates. fintrack has not been switched over yet (step 4).
First consumer is fintrack's paper-trading investment agent
(`fintrack/investment-agent-spec.md`), whose six LLM roles were built
there as one-off code and are the evidence for which archetypes are
actually needed. Spec 8 is parked "until fintrack has been tested with
base RSI + Cortex" — this spec is how that test happens.

## 0. The idea in one paragraph

Today this repo ships two *whole pipelines* (coding, research), and a
consumer can only reuse one by exporting it wholesale. fintrack's fund
needs something different: several small **roles** — analysts that judge
a fixed evidence set, a bull and a bear, a portfolio manager deciding under
hard constraints, a reflector writing lessons. None of those is a research
pipeline, and rewriting them on GPT Researcher would lose the thing that
makes them trustworthy (claim → evidence-id → number check against a
database row). So: add a catalog of **archetypes** — reusable agent types,
each shipping a runtime, a fixed validator, a self-healing loop, Cortex
hooks, and a benchmark template. A consumer supplies only its prompts,
its evidence, and its fixtures. Every archetype is RSI-ready the moment a
consumer writes fixtures, because the validator that guards production
calls is the same thing that grades benchmark runs.

## 1. Why archetypes and not more pipelines

| | Pipeline (today: coding, research) | Archetype (this spec) |
|---|---|---|
| Unit | A whole task, end to end | One role inside someone else's pipeline |
| Who orchestrates | This repo (`pipeline.py`, `research_pipeline.py`) | The consumer (fintrack's `runner.py`/`agent.py`) |
| Reuse story | `export_strategy()` copies the whole tree | `pip install`, import a runtime, pass prompts + evidence |
| Ground truth for RSI | Exact stdout / required keywords | The archetype's own deterministic validator + consumer fixtures |
| Task-level RSI | Blocked (Spec 5 §6 Q1) | Buildable — see §6 |

The two existing pipelines stay as they are and appear in the catalog as
archetypes A1 and A7. Nothing in `rsi/proposer.py` or `rsi/composer.py`
changes for any of this; `rsi/probe.py` gets one seam (§7).

## 2. The catalog

| Id | Archetype | Contract | Fixed validator (never RSI-editable) | Benchmark shape (mechanical) | First fintrack use |
|---|---|---|---|---|---|
| A1 | **Deep Researcher** *(exists)* | question → cited report | `agents/research/policies.py` | required keywords in report | news / qualitative analyst on finalists, retriever = local SearXNG (`retrievers/searx`) |
| A2 | **Evidence Analyst** | numbered evidence set → JSON judgment, every claim cites evidence ids | citation ids exist and are allowed; numbers in a claim match the cited item (tolerance + unit scaling); schema | stance on clear-cut fixtures; % claims with valid citations; % numbers verified; planted-fact recall | disclosure, market, fundamentals analysts |
| A3 | **Adversarial Debater** | evidence + analyst digests + side → strongest case for that side | same as A2, plus "must not cite only the other side's evidence" | fixtures with one planted red/green flag: did the right side find and cite it; citation validity | bull and bear |
| A4 | **Decision Maker** | candidates + state + constraints → actions JSON | hard constraints in code (allowed actions per item, caps, required citation for any non-default action); overrides are logged, not hidden | clear-answer fixtures (e.g. stop hit + negative evidence → exit); **override rate** (how often code had to correct it); thesis citation validity. **Never realized returns** — §8 | portfolio manager |
| A5 | **Extractor / Classifier** | document → labeled fields | schema + closed label set | accuracy / F1 on labeled examples | 8-K item content → event type (fintrack's news analyst today sees titles only) |
| A6 | **Reflector** | finished episode → `abstracted_pattern` / `failure_summary` / `distilled_learnings` | every number in a lesson appears in the episode; abstracted_pattern contains no entity names (configurable denylist, e.g. tickers) | groundedness rate; abstraction rate | writes fund lessons when a paper trade closes |
| A7 | **Coder** *(exists)* | task → working code | sandbox | exact output (`rsi/coding_benchmark.py`) | — |

A2–A6 are the new work. A6 generalizes `agents/research/reflection.py`
(which stays, as the research pipeline's own instance).

### 2.1 Borrowed from: one reference repo per archetype

A1 vendors GPT Researcher because RSI edits its strategy files in place.
A2–A6 are single-call roles whose editable surface is a prompt file
(§3), so there is nothing to vendor. Each takes its **pattern** from an
existing repo and adds the one thing those repos don't have: a
deterministic validator that both heals the call and grades the
benchmark. No code was copied, and none of these repos is a dependency.

| Id | Reference repo (license) | What's borrowed | What's added here |
|---|---|---|---|
| A1 | [assafelovic/gpt-researcher](https://github.com/assafelovic/gpt-researcher) (Apache-2.0) | vendored, see `agents/research/VENDOR.md` | — |
| A2 | [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) (Apache-2.0) analyst team; [princeton-nlp/ALCE](https://github.com/princeton-nlp/ALCE) | one narrow analyst per evidence slice; ALCE's per-claim citation measurement | ids must exist and be allowed; numbers must match the cited item |
| A3 | TradingAgents bull/bear researcher debate | configurable rounds, each side answering the other's last case | `one_sided_citation`: a side citing only the opponent's evidence isn't arguing its case |
| A4 | [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) (MIT) | risk manager computes limits in code, PM may only choose within them | overrides logged as a metric (`override_rate`), never healed away |
| A5 | [google/langextract](https://github.com/google/langextract) (Apache-2.0) | every extraction maps to a char interval of its source | grounding *enforced*: a non-verbatim quote is a problem the model must fix |
| A6 | [noahshinn/reflexion](https://github.com/noahshinn/reflexion), [LeapLabTHU/ExpeL](https://github.com/LeapLabTHU/ExpeL) (Apache-2.0) | verbal reflection after outcome; ExpeL's failed-vs-successful contrast (`compare_with`) | groundedness (no invented figures) + abstraction (no entity names in `abstracted_pattern`) |
| heal loop | [jxnl/instructor](https://github.com/jxnl/instructor) | re-ask in the same conversation with the validation error | deterministic validators instead of pydantic ones; stop when a re-ask makes no progress |

Not adopted, and why:
- **TradingAgents' free-text debate:** there's no claim → evidence-id
  structure, so nothing to validate.
- **ExpeL's ADD/EDIT/UPVOTE insight operations:** Cortex already does
  reinforcement through `write_back(retrieved=...)` and promotion
  through Replay. A second voting system would compete with it.

## 3. Anatomy shared by every archetype

```python
@dataclass(frozen=True)
class ArchetypeSpec:
    name: str                     # "evidence_analyst", ...
    output_schema: dict           # JSON schema; enforced via Ollama `format` or tool_use
    validator: Validator          # fixed, deterministic — §4
    max_heal_attempts: int = 2    # self-healing loop budget — §5

@dataclass
class RoleConfig:                 # consumer-owned, RSI-editable — this is the c1/c7 surface
    role: str                     # "disclosure_analyst"
    system_prompt: str            # loaded from the consumer's prompt file
    temperature: float = 0.2
    extra: dict = field(default_factory=dict)  # tunables the consumer exposes to RSI

@dataclass
class Call:                       # one invocation
    inputs: dict                  # evidence block, context, constraints
    allowed_evidence: set[str]    # ids this role may cite
    evidence: EvidenceIndex       # id -> payload, for number checks
    memory: RetrievalContext | None  # Cortex retrieve() output, already scoped

@dataclass
class Result:
    output: dict                  # validated, possibly with claims dropped
    validation: ValidationReport  # problems found, per attempt
    attempts: int
    meta: dict                    # model, seconds, tokens
```

`run(spec, role, call, llm) -> Result` is the single entry point. It never
raises on model failure — it returns a schema-valid default with
`validation.failed=True` — because a consumer pipeline with 50 calls per
run should degrade per role, not crash (fintrack learned this the hard
way on its first run).

**Prompts are data, not code.** The consumer keeps each role's prompt in
its own file (e.g. `fintrack/backend/app/services/fund/prompts/*.md`) and
passes the text in. RSI edits those files on the consumer's lineage
branch; the runtime just reads them. This is what unblocks §6.

## 4. Validators are policy, not prompt

Same principle as `agents/research/policies.py` (Omnigent-style: enforced
outside the model, not instructions it can ignore), extended from run-level
caps to per-claim checks:

- **Citation check** — each claim lists evidence ids; unknown or
  disallowed ids → claim dropped, problem recorded.
- **Number check** — numbers in the claim text must match a number in a
  cited item's payload within tolerance, accepting ×100 / ×1e3 / ×1e6 /
  ×1e9 unit scaling; small integers (≤10) exempt. Failing claims are kept
  but flagged `numbers_unverified` (dropping them loses too much signal;
  flagging lets downstream roles and benchmarks count them).
- **Constraint check (A4)** — consumer supplies a pure function
  `constrain(decision, state) -> (decision', overrides[])`. The archetype
  applies it and records every override. The override list is both an
  audit log and a benchmark metric.

fintrack's `services/fund/validate.py` is the reference implementation and
moves here largely as-is. Validators live outside every `HarnessSpec`
component tree by construction, exactly like `cortex/` (see the comment in
`rsi/harnesses.py`): an edit benchmarked by a validator must not be able to
change the validator.

## 5. Self-healing loop (per call)

```
attempt 1: model → output → validator
  problems == 0          → return
  problems > 0, budget   → attempt n+1 with the problem list appended:
                           "Claim 2 cites ev12, but ev12 says 3 officers, not 4."
  budget exhausted       → return best attempt (fewest problems), failed=False
                           unless schema itself never validated
after return:
  any problem class seen on ≥2 attempts → candidate Caution text for Cortex
  ("disclosure_analyst: inflates officer counts when several Form 4s share a date")
```

Today fintrack retries blindly (same prompt twice). Feeding the validator's
specific complaints back is the "healer" from `coding_agent_plan.md`
applied to structured output instead of code — `HealerAttempt` in
`cortex/schema.py` already has the right shape (hypothesis / fix / outcome).

## 6. Task-level RSI, unblocked

Spec 5 §6 Q1 left the task-level probe as a deliberate stub for two
reasons: no contrast judge without ground truth, and no safe way to run a
candidate prompt without mutating a live vendored file. Archetypes remove
both:

1. **The judge exists.** A2–A6 have deterministic validators. "Candidate
   prompt produced fewer validator problems than baseline on the *same*
   inputs" is a ground-truth contrast, not a model's opinion.
2. **No file mutation.** Prompts are data (§3). A candidate prompt from
   `propose_from_trajectory()` is run in memory for one call; files are
   never touched mid-episode, so a crash can't corrupt the harness.

```
trigger:  same problem class on every heal attempt for a role,
          across ≥ N calls in one consumer run (N configurable, default 3)
propose:  propose_from_trajectory(role prompt, validator problem list)
probe:    re-run those N calls with baseline and candidate prompt, same inputs
compose:  candidate_problems < baseline_problems on ≥ 2/3 of calls
          → use candidate for the rest of this run, log to rsi_data/<harness>/task_level.jsonl
promote:  same edit adopted in ≥ 3 runs → surfaced to the harnessing-level
          proposer as a pre-vetted proposal (mirrors Replay's promote_threshold=3)
```

Task-level adoption never writes to disk and never reaches `main`;
promotion still goes through the harnessing-level benchmark gate.

## 7. Harnessing-level RSI for a consumer repo

The consumer builds its own `HarnessSpec` pointing at its own prompt files,
in its own repo, with its own lineage branch — the path the README
already describes for third parties. One engine gap blocks it today:

- **`probe.run_sanity_suite()` hardcodes `python -m tests.test_cortex`.**
  A consumer worktree has no such module. Add
  `HarnessSpec.sanity_fn: Callable[[Path, str], tuple[bool, str]] | None`,
  defaulting to the current behaviour for this repo's own harnesses.

Benchmarks run as subprocesses inside the worktree, as
`WRITING_YOUR_OWN_BENCHMARK.md` requires, against **frozen fixtures**: a
fixture is a saved `Call` (inputs + evidence + allowed ids, no memory) plus
its expected mechanical facts. fintrack already persists everything needed
(`fund_evidence`, `fund_analyst_reports`, `fund_debates`) — an export
command turns real past calls into fixtures, and a human adds the expected
facts. The scalar score keeps the existing shape:

```
scalar_score = 100 * fixture_pass_rate
             +  20 * citation_validity      # 0..1
             +  20 * number_verified_rate   # 0..1
             -   5 * avg_heal_attempts
             - (A4 only) 30 * override_rate
```

## 8. What RSI must never optimize: realized returns

Stated as a hard rule because it is the obvious-looking wrong benchmark
for any finance consumer:

- **Live returns are too slow and too noisy.** A 5–30 day paper trade's
  excess return vs SPY is dominated by noise; a gate on it accepts edits
  at random (the failure mode `WRITING_YOUR_OWN_BENCHMARK.md` warns about).
- **Historical returns are contaminated.** The model's training data
  contains what happened next; prompt edits would be selected for
  recalling history, not for reasoning (fintrack spec §7, Look-Ahead-Bench).

Archetype benchmarks therefore measure **faithfulness and constraint
discipline**, which are checkable today. Whether better-behaved agents
also pick better is answered separately, by the consumer's own live
evaluation against baselines (fintrack: agent vs quant_only vs cloud_top5),
over months, never inside an RSI gate.

## 9. Cortex changes (small, but in the fixed substrate)

1. **Point-in-time retrieval.** Add `created_at` to every collection's
   metadata on write and an optional `as_of` to `retrieve()` /
   `store._query()` (`where={"created_at": {"$lte": as_of_ts}}`, combined
   with `scope`). Without it, a historical replay retrieves lessons from
   episodes that closed after the replay date — look-ahead through memory.
2. **Deferred outcomes.** A trade's outcome is known days or weeks after
   the decision. Rule: write the outcome episode **at close**, not at
   decision time. The consumer holds the decision context in its own DB
   until then. Process episodes (validator problems, heal attempts) are
   written immediately — their outcome is known at once.
3. **Noise band on outcomes.** Let the consumer map a raw result to
   `Outcome` with a dead zone (e.g. |excess vs SPY| < 1 σ of that horizon
   → `PARTIAL`), so Hebbian `reinforce()` doesn't strengthen grooves on
   coin flips. Instinct promotion keeps Replay's ≥3 independent episodes,
   and the consumer can raise it per domain.
4. **Separate store per consumer.** `CortexStore(persist_dir=...)` already
   allows it; fintrack uses its own directory, `domain="fintrack_fund"`.

All four keep `cortex/` model-agnostic and are covered by extending
`tests/test_cortex.py` (as_of filtering, deferred write, noise band).

## 10. Packaging

- **Core install is light:** `pip install self-healing-ai` → `cortex`,
  `rsi`, `agents.general` (chromadb, networkx, numpy, pydantic, httpx).
- **Extras:** `[research]` → vendored GPT Researcher + LangChain stack;
  `[claude]` → anthropic. Today every consumer pays for LangChain even if it
  only wants Cortex.
- **New adapter `rsi/llm_adapters.ollama_json_call()`** (and a runtime
  equivalent for archetypes): direct Ollama `/api/chat` with `format=<json
  schema>`, `temperature`, `num_ctx`. The existing `qwen_ollama_llm_call`
  goes through GPT Researcher's `GenericLLMProvider`, which forces the
  `[research]` extra just to call a local model.

## 11. File layout

```
agents/general/
  __init__.py            # ArchetypeSpec, RoleConfig, Call, Result, run()
  runtime.py             # run(): memory injection → model → validate → heal loop
  validators.py          # citation / number / schema / constraint checks (fixed)
  evidence.py            # EvidenceIndex: id → payload, render to prompt block
  evidence_analyst.py    # A2: schema + defaults
  debater.py             # A3
  decision_maker.py      # A4: constrain() hook, override logging
  extractor.py           # A5
  reflector.py           # A6: writes Episode fields; groundedness/abstraction checks
  fixtures.py            # Fixture format, load/save, subprocess benchmark runner
  task_rsi.py            # §6 task-level propose/probe/compose
  llm.py                 # JsonLLM: ollama_json, anthropic_json, ScriptedLLM (tests)
  templates/
    benchmark_template.py
    prompts/             # starter prompt per archetype, consumer copies and owns
rsi/
  harness_spec.py        # + sanity_fn (§7)
  probe.py               # run_sanity_suite uses sanity_fn when given
  llm_adapters.py        # + ollama_json_call (§10)
cortex/
  store.py, retrieval.py # + created_at / as_of (§9)
tests/
  test_cortex.py         # + as_of, deferred write, noise band
  test_general_agents.py     # validators, heal loop, task-level probe with a stub llm
docs/
  09_agent_archetypes_spec.md
```

## 12. First consumer: fintrack's fund

| fintrack today (`backend/app/services/fund/`) | Becomes |
|---|---|
| `analysts.py` disclosure / market / fundamentals | A2 roles; prompts → `fund/prompts/*.md` |
| `analysts.py` news (3 fixed SearXNG queries) | A1 on finalists only, `searx` retriever → evidence items with URL + date; A5 for 8-K content |
| `debate.py` bull / bear | A3 roles |
| `pm.py` | A4; fintrack's in-code PM rules become its `constrain()` |
| `validate.py`, `evidence.py` | Move into `agents/general/validators.py`, `agents/general/evidence.py` |
| *(new)* trade-close lessons | A6, written to fintrack's Cortex store at close |
| `risk.py`, `ledger.py`, `quant.py`, `policies.py`, `runner.py` | **Stay in fintrack, never RSI-editable** — sizing, accounting and the no-LLM baseline must not move when prompts improve |

fintrack's `HarnessSpec`: component tree = the prompt files only;
`sanity_fn` = fintrack's own pytest subset; `benchmark_fn` = fixtures
exported from real runs (§7).

## 13. Build order

1. `ollama_json_call` adapter + packaging split (§10) — unblocks a light install.
2. `agents/general/validators.py` + `evidence.py`, ported from fintrack, with tests.
3. `runtime.run()` with the heal loop (§5); A2 first, then A3, A4.
4. fintrack switches its analysts / debate / PM onto A2–A4 with no
   behaviour change except validator-guided retries. Compare one live run
   before and after: heal attempts, problems per call, run time.
5. Cortex `as_of` + deferred outcome + noise band (§9); A6 reflector;
   fintrack writes lessons at trade close.
6. `HarnessSpec.sanity_fn` (§7), fixture export in fintrack, first
   harnessing-level cycle on one role (disclosure analyst).
7. Task-level RSI (§6).
8. A5 extractor for 8-Ks; A1 news analyst on SearXNG.

Steps 1–4 deliver value with no RSI at all (better retries, shared code).
RSI arrives only once fixtures exist, which is the right order — Spec 5's
own lesson was that the engine is easy and the benchmark is the work.

## 14. Non-goals

- Not rewriting the research or coding pipelines onto archetypes.
- Not a general agent framework (no planner, no tool router). Archetypes
  are single-call roles with a validator; orchestration stays with the consumer.
- Not optimizing any agent on financial returns (§8).
- Not making validators or Cortex RSI-editable, ever.

## 15. Open questions

1. Number-check tolerance per archetype: fintrack's analyst check uses
   relative tolerance plus unit scaling. Is that right for A3/A4, where
   claims often restate rounded numbers ("about 40%")?
2. Should the self-healing loop's feedback text be part of the RSI-editable
   surface (it is effectively a prompt), or fixed alongside the validator?
   Leaning editable, since it can't weaken the validator itself.
3. Fixture labeling cost: how many fixtures per role before a gate is
   meaningful? Starting point from `WRITING_YOUR_OWN_BENCHMARK.md`: 5–10
   per role, weighted toward cases the current prompt gets wrong.
4. A4 override rate as a metric could reward a timid model that always
   picks the default action. Pair it with clear-case accuracy (which
   penalizes missing an obvious exit), or add a minimum-action-rate floor?
5. Should task-level adoptions (§6) be written to Cortex as process
   episodes, so Replay can promote recurring ones to Instinct alongside
   the RSI promotion path, or kept only in `rsi_data/`?

## 16. Build status

| §13 step | State |
|---|---|
| 1. `ollama_json_call` + packaging split | **done.** `rsi.ollama_json_call` (async, for the proposer), `agents.general.ollama_json` / `anthropic_json` (sync, schema-enforced). Core install = chromadb, httpx, networkx, numpy, pydantic; `[research]`, `[claude]`, `[all]` extras. A clean core-only venv pulls in no LangChain, Anthropic or tiktoken packages, and both test suites pass from it. |
| 2. validators + evidence | **done.** Ported from fintrack, plus `find_span` (A5 grounding), a minimal JSON-schema checker, and `NumberPolicy` for per-archetype tolerance (§15 Q1). |
| 3. runtime + heal loop, A2–A4 | **done**, and A5/A6 too. `run()`; `debate()` + `digest()` for A3; `constrain()` hook for A4; `Result.to_episode()` / `caution_candidates()` for Cortex. |
| 6 (part). `HarnessSpec.sanity_fn` | **done.** `rsi.run_command_sanity([...])` builds one from a command; the default is unchanged for this repo's harnesses. |
| fixtures + scalar (§7) | **done.** `Fixture`, `score_fixtures()`, plus `templates/benchmark_template.py` and starter prompts. Template run end-to-end against qwen3:14b in a mock consumer repo. |
| 4. fintrack switch-over | not started (fintrack repo) |
| 5. Cortex `as_of`, deferred outcomes, noise band | not started |
| 6 (rest). fixture export in fintrack, first harnessing cycle | not started |
| 7. task-level RSI (§6) | not started; `Result.history` holds what its probe needs |
| 8. A1 on SearXNG for news | not started |

Tests: `python -m tests.test_general_agents`, 16 tests, no model needed (`ScriptedLLM`).

**Live findings against qwen3:14b** (fixed, with regression tests):
- **`S&P 500` flagged as a misquoted 500.** fintrack's validator has the
  same false positive. With nothing actually wrong, the model resubmitted
  the identical claim until the budget ran out (3 calls, 58s). Two fixes:
  - `NumberPolicy.ignore` takes regexes for name-numerals and strips them
    from the claim side. The default is empty, because which names occur
    is domain knowledge. The consumer supplies its own: for fintrack,
    index names, Form N, N-K, Item N.NN, Qn.
  - The heal loop now stops when an attempt returns the same problems as
    the one before (`meta["stalled"]`). This is the generic safety net for
    any false positive a consumer hasn't listed.

  With fintrack's patterns supplied, the same call finishes clean in one
  attempt (15s). **fintrack's `validate.py` should take the same fix
  when it moves here (step 4).**
- **Domain-neutral code rule:** `agents/general/` and its tests contain no
  domain vocabulary. Examples and fixtures use a neutral domain
  (evaluating a service release), and the repos each archetype borrows
  from are credited here in §2.1, not in code.
- **A5 accepted a whole sentence as a number field's value**, because the
  number did appear in its quote. Number values must now parse as a
  number (`not_a_number`). Re-run live, the model fixed it to `"250"` on
  its single heal attempt.
