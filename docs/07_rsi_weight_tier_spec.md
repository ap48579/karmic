# Spec 7 — RSI Weight Tier: Fine-Tuning the Model When Harness Edits Plateau

Status: **design only, nothing built.** Targets the research harness
(`agents/research/`), not fintrack. Builds on Spec 5's `rsi/` engine.
Supersedes nothing — Spec 6 (exporting Cortex traces as SFT/DPO data)
stays valid as a data source, but this spec's main data source is fresh
rollouts against public QA datasets (§2), since the current Cortex data
is dry-run placeholders.

## 0. The idea in one paragraph

RSI today edits text (c1–c8, Spec 5 §1) and accepts an edit only if
`benchmark_fn` scores strictly higher. When text edits stop helping, the
remaining inefficiency usually lives in the model itself: it needs long
prompts, guardrail text and Grader↔Reviser loops to get the right answer.
This spec adds **c9 = model weights (a LoRA adapter)** as one more
editable component. A plateau in text edits triggers a weight proposal
(a training run), which goes through the **same** Probe → Compose gate
as any other edit. One loop, one acceptance rule.

```
RSI generation loop
  ├─ plateau? no  → propose text edit (c1–c8) → benchmark → accept/reject
  └─ plateau? yes → propose weight edit (c9)  → train LoRA → benchmark → accept/reject
                     └─ after accept: resume text edits, incl. pruning (§6)
```

Lineage entries become `(harness commit, adapter id)` pairs.

## 1. Goal: efficiency, not just accuracy

The target is "do the task more efficiently": same or better correctness
with fewer revisions, tokens and search calls. The scalar score must say
so explicitly. Current (`WRITING_YOUR_OWN_BENCHMARK.md`):

```
scalar = success_rate*100 − avg_revisions*5
```

Proposed:

```
scalar = success_rate*100 − avg_revisions*5 − tokens_per_correct/1000 − avg_search_calls*1
```

Weights are a starting point; tune so a human reading the per-task
breakdown would agree with the ranking. Note `CostActual` currently
undercounts GPT Researcher's internal tokens (Spec 3 §2) — the token term
is only as good as that measurement; fix or accept the undercount knowingly.

## 2. Prerequisite: a benchmark that actually needs research

`rsi/research_benchmark.py`'s 5 tasks (Chernobyl, Versailles, WWW,
Python, boiling point) are answerable from model memory. Fine-tuning
against them would teach the model to skip research. Replace with public
multi-hop QA, which already fits the `question + required_keywords` shape:

| Dataset | Role |
|---|---|
| HotpotQA / 2WikiMultiHopQA | large train split for rollouts; 2-hop |
| MuSiQue | harder 2–4 hop, shortcut-resistant; train + dev |
| FRAMES (~800 multi-source questions) | held-out test — closest to "deep research" |

- **Splits**: `train` (rollouts for training), `dev` (~50 questions — the
  RSI gate), `test` (frozen; only for final reporting, never the gate).
- **Matching**: keep substring checking but normalize (lowercase, strip
  punctuation, alias lists).
- **No-tools filter**: ask the bare model each question with no
  retrieval; drop questions it already gets right.
- **Cost**: each task is a full pipeline run (minutes on local models).
  Dev must stay small enough that one gate evaluation finishes in a
  reasonable time; consider running dev tasks in parallel.

## 3. Plateau detector

Reads `rsi_data/generations.jsonl`. Trigger the weight phase when either:

- K consecutive text proposals rejected (start with K=5), or
- best scalar improved by < ε over the last N generations.

Log the trigger reason in the generation record — "when does prompt
editing run out of steam" is itself a result worth having.

## 4. Bottleneck attribution before training

The pipeline is several roles (planner → researcher/query generation →
writer → grader → reviser → reflection). Attribute dev failures to a role
from traces before choosing what to train:

| Symptom | Bottleneck |
|---|---|
| Right source retrieved, keyword missing from report | writer |
| Right source never retrieved | query generation |
| Correct draft, revised anyway | planner/writer verbosity, or grader strictness |

- **One LoRA adapter per bottleneck role**, same base model. Ollama and
  vLLM can both serve multiple adapters on one base.
- **The grader is never trained.** It stays frozen, like
  `agents/research/policies.py`. Training policy roles on the grader's own
  verdicts lets them learn to please the grader instead of being correct.
  Ground truth is the only reward source.

## 5. Training methods (cheapest first; stop when the gate stops accepting)

### A. Distill the harness into the weights

At the plateau, the harness gets right answers through extra work. Teach
the model to skip it:

- **Revision distillation**: for correct episodes, train on
  (question + evidence → *final revised* report), dropping the drafts.
  Targets `avg_revisions` directly.
- **Context distillation**: train the model *without* long c1/c2 prompt
  sections to reproduce its outputs *with* them. Targets prompt tokens on
  every call.

Data: rejection sampling — run the current best harness 4–8× per train
question, keep only correct runs. SFT + LoRA/QLoRA; feasible on a Mac via
`mlx-lm`.

### B. DPO on efficiency pairs

Per question, pair two correct runs: chosen = fewer tokens/searches/
revisions, rejected = more. Correctness is equal, so the only learned
preference is efficiency. Add correct-vs-wrong pairs for accuracy.

### C. GRPO for query generation (only if A and B plateau)

Search-R1-style: 8 rollouts per question with live search calls; reward
= exact match minus a small per-search cost; loss masked to model-
generated tokens only (not retrieved text). Needs a rented GPU with
verl or OpenRLHF — the only stage that does.

## 6. Hand back to RSI: pruning proposals

After a weight edit is accepted, many earlier text edits were only
compensating for weaknesses the adapter now fixes. Add a **pruning**
proposal type: remove a prompt/guardrail section, gate it on the same
benchmark. Saves tokens; the gate catches anything that was still needed.

Then continue alternating: text edits → plateau → weights → pruning → …

## 7. Where it lands in the code

| Piece | Location |
|---|---|
| c9 component, `propose_weight_edit()`, pruning proposals | `rsi/component_tree.py`, `rsi/proposer.py` |
| Plateau detector | `rsi/composer.py` (reads `generations.jsonl`) |
| Dataset loaders + splits + no-tools filter | new `rsi/datasets.py` |
| Rollout collector (train split, N samples/question) | new `rsi/rollouts.py` — subprocess to `cli.py --research`, same pattern as the benchmark |
| Trainers (SFT / DPO / GRPO) → adapter path | new `rsi/train/` |
| Adapter-aware benchmark | `rsi/research_benchmark.py` passes an adapter/Ollama model tag via env |
| Lineage records adapter id + trigger reason | `generations.jsonl` schema |

Nothing in `cortex/` changes (same principle as Specs 2 and 6).

## 8. What to measure

- Scalar vs. generation, with the text-plateau point and each weight
  edit marked — does the plateau actually move?
- Tokens per correct answer and revisions per task, before/after A, B, C.
- Per-role attribution: which adapter produced the gain.
- Harness text removed by pruning after each weight update.
- Final numbers on the frozen `test` split (FRAMES) only.

## 9. Build order

1. New benchmark (§2) + efficiency scalar (§1).
2. Plateau detector (§3) + run text-only RSI to an actual plateau.
3. Bottleneck attribution (§4).
4. Method A end-to-end on a Mac — proves the whole loop.
5. B, then C only if needed.
6. Pruning proposals (§6).

## 10. Later: other domains (e.g. fintrack)

RSI is generic (`HarnessSpec` + `benchmark_fn`), so another domain is
another harness with its own benchmark. Fintrack fits only once its
benchmark uses answers checkable immediately (reported figures, filing
dates — see `WRITING_YOUR_OWN_BENCHMARK.md`), not delayed outcomes like
"was the bullish call right."

## 11. Open questions

1. Exact plateau thresholds (K, ε, N) — set after observing a real
   text-only RSI run.
2. Whether `qwen3:14b` (Spec 3's live-tested model) or the 7B in
   `plan_rl.md` is the training target — 14B QLoRA SFT on a Mac is
   possible but slow; 7B iterates faster.
3. How to serve per-role adapters through `GenericLLMProvider` — one
   Ollama model tag per adapter is simplest; confirm it doesn't break
   GPT Researcher's own model config.
4. Whether accepting a weight edit should require a larger margin than a
   text edit, since it's costlier to roll back and easier to overfit dev.
