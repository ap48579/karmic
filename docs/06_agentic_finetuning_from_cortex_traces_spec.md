# Spec 6 — Agentic Fine-Tuning from Cortex Traces (SFT/DPO export + RL curriculum seed)

Status: design only, nothing built. Blocked on real trace data existing —
see §5. Distinct experimental track, same framing as `plan_rl.md`: this
does **not** change Cortex's own "no fine-tuning/RL, inference-time
scaffolding only" design (Spec 2 §1). Cortex produces data this track
consumes; nothing in `cortex/` changes.

## 1. What this spec answers

Can `Episode` records already written by Cortex's write-back path
(`cortex/schema.py`, persisted to `episodes.jsonl`) be turned into training
data for `plan_rl.md`'s agentic fine-tuning track, without redesigning
Cortex itself? Short answer: partially, directly, today — for SFT
warm-start (`plan_rl.md` Phase 1) and for DPO-style preference pairs. Full
on-policy RL (`plan_rl.md` Phases 2–3) needs a live, steppable environment
wrapping the harness, not just these logs; this spec does not build that
environment, only the two lower-cost pieces that can be built from what
Cortex already records.

## 2. Field → role mapping

`Episode` (`cortex/schema.py:37`) already carries everything these exports
need; no schema change required.

| `Episode` field | Role here |
|---|---|
| `action_arc` (list of `action`/`tool_used`/`output`) | the trajectory itself — SFT imitation target |
| `healer_attempts` (list of `hypothesis`/`fix`/`outcome`) | ordered attempts on one problem — DPO chosen/rejected source |
| `outcome` (success/partial/failure) | SFT filter; also the terminal label for building preference pairs |
| `predicted_complexity` + `healer_attempt_count` → `salience_boost()` (`schema.py:68`) | curriculum ordering signal — see §4.3 |
| `domain` / `scope` | keeps exports separable by domain, same as Cortex's own retrieval scoping (Spec 2 §5) |
| `cost_actual` | optional reward-shaping input (penalize token-heavy trajectories), not used in v1 |

## 3. Where this code lives

Not inside `cortex/`. Spec 2 is explicit that Cortex stays model-agnostic,
inference-time scaffolding (`cortex/writeback.py`'s own stated intent,
quoted in Spec 2 §1) — an export pipeline that shapes data for weight
updates is a different concern and a different consumer, not a Cortex
responsibility. New top-level directory:

```
finetune_export/
├── sft_export.py      # §4.1 — episodes.jsonl -> SFT jsonl
├── dpo_export.py      # §4.2 — episodes.jsonl -> preference pairs
├── curriculum.py       # §4.3 — salience-ranked episode ordering
└── README.md
```

Each script reads `episodes.jsonl` read-only. Nothing here calls into
`cortex/store.py`, `graph.py`, or any Cortex write path — one-directional,
Cortex → export tool, never the reverse.

## 4. The three deliverables

### 4.1 SFT export (buildable now, data permitting)

Filter `outcome == "success"`. For each surviving episode, flatten
`action_arc` into a trajectory the policy model should imitate — one
training example per episode, not per step, since the whole arc is the
unit of "did this sequence of actions solve the problem."

Target format: match whatever `verl`'s SFT data loader expects
(`plan_rl.md` §5 names `verl` as the inherited training infra). **Open
question, not resolved by this spec**: confirm the exact field names
`verl`'s SFT path requires — treat the shape below as a draft to validate
against `verl`'s docs before relying on it, not a settled contract.

```json
{"episode_id": "...", "domain": "...", "messages": [
  {"role": "user", "content": "<problem_summary>"},
  {"role": "assistant", "content": "<action 1 + tool_used 1 + output 1>"},
  {"role": "assistant", "content": "<action 2 + ...>"},
  ...
]}
```

### 4.2 DPO preference pairs (buildable now, with a caveat)

`healer_attempts` records multiple tries at the same problem in order —
e.g. the sample episode in `cortex_data/episodes.jsonl` has three
revisions, outcomes `revise, revise, revise`, final `outcome: partial`.
Later attempts that moved the episode closer to (or to) success are
`chosen`; earlier attempts superseded by a later one are `rejected`.

**Caveat, current schema gap**: `HealerAttempt` (`schema.py:25`) stores
only `hypothesis`/`fix`/`outcome` — it does not record which point in
`action_arc` each attempt branched from. Building a clean (prompt, chosen,
rejected) triple needs that shared context. Two options, neither requiring
a Cortex change: (a) reconstruct it heuristically from arc position for
existing data, accepting some noise; (b) for episodes recorded going
forward, have the harness thread an `arc_index` into each `HealerAttempt`
at write time (a harness-side change, not a Cortex schema change — Cortex
just persists whatever `Episode` it's given). Recommend (b) before
investing in (a).

### 4.3 Curriculum seed for the eventual RL phase

Not a training dataset — an ordering. `plan_rl.md` §3's ScalingInter-RL
needs an easy→hard task schedule; `salience_boost()` already flags
episodes where predicted difficulty and actual difficulty (via
`healer_attempt_count`) diverged. Rank stored episodes by
`(predicted_complexity, healer_attempt_count)` and hand that ordering to
whichever live environment eventually wraps the harness (`plan_rl.md`
Phase 0) — real salience data instead of a guessed horizon schedule.

## 5. The actual blocker: there is no real data yet

Checked both trace files in the repo
(`self_healing_ai/cortex_data/episodes.jsonl` and
`self_healing_ai/notebook_trace_data/cortex_data/episodes.jsonl`): one row
each, both dry runs — `initial_plan` and other fields contain literal
`[DRY RUN — no ANTHROPIC_API_KEY set...]` placeholder text, not real model
output. None of §4 is meaningful against this data. Prerequisite to any of
this: run the harness for real (API key set, or against the local
`deepseek-r1:7b` per `plan_rl.md`'s model choice) enough times to
accumulate genuine `action_arc` / `healer_attempts` data across a range of
`predicted_complexity` values.

## 6. Non-goals

- Not building the live/steppable RL environment — that's `plan_rl.md`
  Phase 0, a harness-wrapping problem, out of scope here.
- Not changing anything inside `cortex/` — confirmed unnecessary per §3.
- Not picking the RL algorithm or reward function — `plan_rl.md` §2–3
  already covers that ground.

## 7. Open questions

1. Exact `verl` SFT ingestion schema (§4.1) — needs confirming against
   `verl`'s own docs before the export format is final.
2. Whether to backfill `arc_index` into `HealerAttempt` for the harness
   going forward (§4.2b) — a harness change, needs sign-off there, not
   here.
3. How many real episodes, across how wide a `predicted_complexity`
   spread, before SFT warm-start data is actually useful — no target
   number yet, revisit once real (non-dry-run) volume exists.
