# Spec 8 — Coding Agent + Agentic RL Fine-Tuning (Spec 7 retargeted to code)

Status: **design, with a partial prototype.** The data/sandbox/agent/
benchmark layer exists in `../self_healing_coder/` (§9) and has produced
one real baseline number (§3). Nothing has been trained. Parked until
fintrack has been tested with base RSI + Cortex.

Builds on Spec 7's weight tier (c9 = LoRA adapter, plateau → train →
same gate). This spec changes the *domain*, not the mechanism: coding
replaces multi-hop QA because correctness is fully mechanical (run the
tests), datasets are public and large, and the reward needs no keyword
matching or alias lists.

## 0. The idea in one paragraph

Build a new consumer project, `self_healing_coder/`, that depends on
`self-healing-ai` as a package (Cortex + `rsi/`) instead of living
inside it, which also tests the package-reuse story. One small local
model plays Coder *and* Debugger in a multi-turn loop: write code → run
visible tests → read the failure → repair. Hidden tests grade the final
submission. RSI tunes the loop's text (c1 prompts, c7 config) as usual;
when that plateaus, GRPO trains a LoRA adapter on full multi-turn
episodes, and the adapter goes through the same dev gate.

## 1. Why coding instead of research QA

| | Research QA (Spec 7) | Coding (this spec) |
|---|---|---|
| Ground truth | substring/alias match on report | hidden unit tests pass |
| Reward noise | matching heuristics, web variance | none (deterministic sandbox) |
| Episode cost | minutes (live search) | seconds (local exec) |
| Rollouts per RL step on a Mac | impractical | 64 episodes ≈ tens of seconds |
| "Needs the harness" filter | no-tools filter on QA | built in: visible-test feedback is the tool |

## 2. Datasets and splits

Same roles as Spec 7 §2:

| Split | Source | Size | Role |
|---|---|---|---|
| `train` | MBPP train | 373 | RL rollouts; never used for acceptance |
| `dev` | MBPP validation | 90 | **the RSI gate**, for text and weight edits alike |
| `test_mbpp` | MBPP test | 497 | frozen, final reporting only |
| `test_humaneval` | HumanEval | 164 | frozen, different prompt style (generalization check) |

- **Visible vs. hidden tests.** The agent's own tester sees only the
  first MBPP assert, which is enough to pin down the name and signature.
  The grade runs every assert plus `challenge_test_list`. If the agent saw
  every test, RL would learn to special-case the visible inputs; this way
  a hard-coded answer passes the agent's tester and then fails the grade.
- **HumanEval visible tests** are taken from docstring `>>>` examples. A
  fallback check that the function is defined covers prompts with no
  usable examples. Examples the reference solution fails are dropped, so
  feedback never calls a correct solution wrong.
- **Broken tasks excluded**: `mbpp/927, 180, 367, 493`. Their reference
  solutions fail their own tests (checked). After this, reference
  solutions pass 100% of hidden and visible tests on every split.
- **Contamination caveat**: Qwen2.5-Coder has very likely seen MBPP and
  HumanEval. Absolute numbers are inflated. The experiment is about
  *deltas* (base vs. adapter, same harness). For a cleaner final number,
  add LiveCodeBench problems after the model's cutoff as a third test
  split (open question 2).
- Later, larger train pools: APPS / TACO / CodeContests (stdin/stdout
  style; needs a second runner mode).

## 3. Baseline already observed — the RL target

Qwen2.5-Coder-1.5B-Instruct (bf16, MLX), greedy, `MAX_TURNS=3`, first 30
dev tasks:

- **13/30 solved, all 13 on the first try. 0 repaired.**
- Every failed task used all three turns. In the trace read, the model got
  `expected {5: 2, 6: 2, ...}, got {5: 1, 6: 1, ...}` and resubmitted
  byte-identical code twice.
- 21 s wall-clock for 30 tasks × up to 3 turns, batched.

So the self-healing loop is currently pure scaffolding: the model can't
use test feedback. That gives this track a crisp, measurable target:
**repair rate** (of tasks whose first attempt fails the visible test, the
fraction solved later). This is plan_rl.md §1's "recover from its own
mistakes", in a setting small enough to measure on a laptop. Next step
before any training: run the full 90-task dev and both test splits for
the base model.

## 4. The agent (episode = one conversation)

```
system: write one ```python block; you'll see test results; fix on failure
user:   problem + visible test
asst:   code            → sandbox(visible) → pass? stop : feedback
user:   "0/1 passed. Failed: <assert>  expected X, got Y" (or traceback)
asst:   repaired code   → ...  (≤ MAX_TURNS)
grade:  last code block → sandbox(hidden) → solved?
```

- One policy for every role (the Coder/Debugger split in
  `self_healing_ai/agents/pipeline_agents.py` collapses), so each episode
  has one set of weights and one reward to assign credit over.
- Episodes run batched: every live conversation advances one turn per
  `batch_generate` call, and solved ones drop out.
- **Sandbox**: fresh `python -I` subprocess, a temp dir, a CPU rlimit, a
  per-test `SIGALRM`, stdin closed, stdout captured. Per-test results,
  plus the actual left-hand value for `assert f(x) == y`. This is not a
  security boundary; it's fine for MBPP-class tasks only.
- **Cortex**: optional `memory_context` is injected into the first user
  turn, and episodes can be written back with `domain="coding_agent"`.
  This also unblocks Spec 6 §5, since these are real non-dry-run Episodes
  with real `healer_attempts`. Memory stays **off** for the benchmark and
  RL so the gate is stationary. Nothing in `cortex/` changes.

## 5. RSI surface (text tier, unchanged engine)

| Component | File | Contents |
|---|---|---|
| c1 system_prompt | `coder/prompts.py` | system prompt, task and feedback templates |
| c7 scripts | `coder/config.py` | `MAX_TURNS`, `MAX_NEW_TOKENS`, `FEEDBACK_MAX_FAILURES`, `FEEDBACK_MAX_CHARS` |
| c9 weights | LoRA adapter dir | §6 |

Fixed substrate, never editable: `datasets.py` (it defines what counts
as correct), `sandbox.py`, `benchmark.py`, `rl/` (reward lives there).

**Gate scalar** (Spec 7 §1 minus the search term):

```
scalar = pass_rate*100 − avg_revisions*5 − tokens_per_correct/1000
```

**Greedy decoding for the gate.** Acceptance is "strictly better than
the lineage tip", so the score has to be deterministic. With sampling, a
re-run of the unchanged tip could beat itself.

`benchmark_fn` runs the worktree's own `cli.py eval --split dev` as a
subprocess, the same pattern as `rsi/research_benchmark.py`. The
accepted adapter reaches it through the `RSI_ADAPTER` env var.

## 6. Weight tier: agentic GRPO on MLX

Spec 7 §5C, retargeted from search queries to code repair. It runs
locally rather than on a rented GPU, because the model is 1.5B and
episodes are cheap.

**One step**
1. Sample B=8 train tasks and run G=8 full episodes each at T=1.0, using
   the *same* `run_episodes` the benchmark scores.
2. `reward = 1[all hidden pass] − 0.1·(turns − 1)`. Ground truth only:
   no model judges anything, consistent with Spec 7 §4's frozen-grader
   rule.
3. Advantage `(r − mean_g)/(std_g + ε)`. Groups with equal rewards carry
   no signal and are dropped.
4. Policy-gradient loss **on the policy's own sampled tokens only**:
   - Each turn is a separate `(context_ids, completion_ids)` pair, stored
     at sampling time. This avoids re-tokenizing decoded text, which can
     silently differ from what was sampled.
   - Test-runner feedback is context and never a target.
   - All turns share the episode's advantage.
   - Losses are normalized over the step's total completion tokens
     (DAPO-style), so long repair turns aren't down-weighted.

**Deliberate simplifications** (each is an open question if it fails):
- One update per fresh rollout batch, so the importance ratio is 1 and
  PPO clipping would be a no-op. It's omitted.
- No KL to a reference model (β=0, as in DAPO / Dr. GRPO). LoRA rank
  limits drift, and the dev gate decides whether an adapter is kept.

**Config to start**

| Setting | Value |
|---|---|
| Base model | Qwen2.5-Coder-1.5B-Instruct, bf16 |
| LoRA | rank 16, scale 20, all 28 layers, q/k/v/o + MLP |
| Optimizer | Adam, lr 1e-5 |
| Micro-batching | ~4k padded tokens per chunk, with gradient accumulation |
| Schedule | ~60 steps, with a greedy dev eval + checkpoint every 10 |

Adapters are saved in mlx-lm's native format, so `mlx_lm.load(...,
adapter_path=)`, `mlx_lm.generate` and `mlx_lm.fuse` all read them.

**Horizon schedule** (plan_rl.md §3, ScalingInter-RL):
`--turn-schedule "2:20,3"` gives 2 turns for the first 20 steps, then 3.
Start fixed at 3, since episodes are already short. Try the schedule only
if training is unstable.

**Credit-assignment caveat.** An episode that fails turn 1 and repairs on
turn 2 gives its *wrong* turn-1 tokens the same positive advantage as the
fix. Turn-level credit is the obvious refinement: rewarding repair turns
on visible→hidden improvement, or "repair-start" rollouts that begin from
a recorded failing turn 1. Hold it back unless repair rate doesn't move.

**Cheaper methods first?** Spec 7's order is A (SFT/distill) → B (DPO) →
C (GRPO). Here A has a weak premise: the base model formats correctly
and solves first-try tasks already. What it lacks is a *behaviour* (using
feedback) that it never demonstrates, so there's little to
rejection-sample. Go straight to C. Keep A as a warm-start fallback if
GRPO groups rarely contain a successful repair (no signal for the target
behaviour).

## 7. Where it lands

New project, `self_healing_coder/` (own git repo, own venv, depends on
`self-healing-ai` editable):

```
coder/   datasets.py sandbox.py config.py prompts.py policy.py agent.py benchmark.py
rl/      grpo.py  (+ sft.py fallback)
harness.py   HarnessSpec(name="coder", c1/c7, benchmark_fn, cortex_domain="coding_agent")
cli.py       eval | solve | train-grpo | self-improve
```

Generic pieces from Spec 7 go in `self_healing_ai/rsi/`: plateau
detector, c9 component and weight cycle, `(commit, adapter id)` lineage
records. Domain-specific training stays in the consumer.

**Required change in `rsi/`, independent of this track:**
`probe.run_sanity_suite()` hard-codes `python -m tests.test_cortex` in
the candidate worktree. For any consumer repo without that module (this
one, *and fintrack*), every cycle would be rejected as "test suite
failed". `HarnessSpec` needs a `sanity_cmd` field defaulting to the
Cortex suite.

## 8. What to measure

- Dev and test, base vs. each adapter: pass rate, **repair rate**,
  first-try rate, avg revisions, tokens per correct.
- Training curves: mean reward, rollout solve rate, rollout repair rate,
  fraction of groups with signal.
- Generalization: does a repair-rate gain on MBPP carry over to
  HumanEval's different prompt style?
- After an accepted adapter, Spec 7 §6 pruning: does `MAX_TURNS` or the
  feedback text shrink without losing dev score?
- Qualitative: read failed repairs. Is it still resubmitting identical
  code, or making real but wrong changes?

## 9. Prototype status (`../self_healing_coder/`)

| Piece | State |
|---|---|
| venv (py3.12, mlx 0.32.4, mlx-lm 0.32.2, datasets, self-healing-ai -e) | done |
| `datasets.py`: splits, visible/hidden, broken-task exclusion | done, validated against references |
| `sandbox.py` | done, tested: timeouts, syntax errors, `sys.exit`, per-test detail |
| `config.py`, `prompts.py`, `policy.py`, `agent.py` | done; produced the §3 baseline |
| `benchmark.py` | written, not yet run end-to-end |
| `rl/grpo.py` | designed (§6), not written |
| `harness.py`, `cli.py`, tests, git init | not started |

## 10. Build order (when resumed)

1. `cli.py eval` and full base-model numbers on dev + both test splits.
2. `sanity_cmd` fix in `rsi/` (also unblocks fintrack). Then `harness.py`
   and one text-only RSI cycle, with qwen3:14b via Ollama as proposer.
3. `rl/grpo.py`, a 5-step smoke run (check loss masking and that
   adapters reload), then the ~60-step run.
4. Gate the best checkpoint via RSI's weight cycle and report frozen test
   numbers.
5. Turn-level credit or SFT warm-start, only if repair rate doesn't move.

## 11. Open questions

1. 1.5B vs. 0.5B: the smaller model has more headroom and is faster, but
   may be too weak to ever produce a successful repair at T=1.0.
2. A contamination-free test split (LiveCodeBench post-cutoff), worth it
   for the final report?
3. Turn penalty 0.1: it pushes toward first-try solves, which could fight
   learning to repair. Compare against 0.05 / 0.
4. Should weight edits need a bigger dev margin than text edits (Spec 7
   §11 Q4)? Dev is 90 tasks, so one task ≈ 1.1 points of pass rate.
