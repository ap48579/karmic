# Writing Your Own Benchmark

The RSI engine (`karmic/rsi/`) accepts a proposed edit if and only if it scores
strictly better on your harness's `benchmark_fn` than the current lineage
tip. That's the *entire* acceptance criterion — there is no human review
step, no self-reported "this looks better," no second opinion. **A
benchmark that doesn't test anything your domain actually cares about
means RSI will accept and reject edits essentially at random with respect
to what matters to you.**

This is not a minor setup step. It is the one piece of this whole system
that cannot be supplied generically, because only you know what "better"
means for your use case.

## What makes a benchmark actually useful here

Look at `karmic/rsi/research_benchmark.py` and `karmic/rsi/coding_benchmark.py` in this
repo for the pattern, but the two properties that matter are:

1. **Ground truth, not self-report.** Every task needs a way to check
   correctness that doesn't involve asking a model "did you do a good
   job?" — the coding benchmark runs the generated code and checks exact
   stdout; the research benchmark checks that specific required keywords
   (a date, a name, a number) appear in the final report. Pick whatever
   mechanical check fits your domain: does the structured output contain
   the right field value, does the generated SQL return the expected row
   count, does the extracted number fall within a known-correct range.
2. **Actually exercises your real pipeline.** The benchmark should call
   through the *same* entrypoint real usage calls, run as a subprocess
   against whatever source tree is being evaluated (baseline or
   candidate) — not an in-process call that might accidentally import
   code that's already loaded instead of the worktree's edited copy. See
   `karmic/rsi/research_benchmark.py`'s docstring for why this has to be a real
   subprocess.

## A concrete worked example

Say you're using the research harness for financial research (see the
`fintrack_research_agent_plan.md`-style integration this project was
originally scoped around). A generic trivia benchmark ("what year was the
Treaty of Versailles signed") tells you nothing about whether an edit
improved financial research quality. Instead:

```python
ResearchTask(
    "q3_2024_revenue",
    "What was [Company]'s reported total revenue for Q3 2024, in USD?",
    required_keywords=["$12.3 billion"],  # the actual reported figure
)
```

Five to ten of these, covering the kinds of questions your real users
actually ask, each with a verifiable correct answer, is enough to start.
You don't need hundreds — the benchmark only needs to be large enough that
a genuinely bad edit can't pass by coincidence, and small enough to run
in a reasonable amount of time (each task is a full pipeline run; local
models can take minutes per task).

## The scalar score

Both built-in benchmarks combine success rate and effort into one number:

```python
scalar_score = success_rate * 100 - avg_revisions * 5
```

This rewards correctness heavily and penalizes wasted revision/debugging
effort lightly — an edit that gets the same success rate with fewer
retries still counts as an improvement. Keep this shape, or design your
own tradeoff, but make sure `scalar_score` genuinely reflects what you'd
want a human reviewer to conclude if they read the same task-by-task
breakdown. The acceptance decision is "candidate's scalar_score is
strictly greater than baseline's," and nothing else.

## Where this plugs in

A `BenchmarkFn` is any `Callable[[Path, str], ScoreLike]` — repo root and
a Python interpreter path in, something with a `scalar_score: float` and a
`to_prompt_text() -> str` out. Pass it as `benchmark_fn` when constructing
your own `karmic.rsi.HarnessSpec` (see `karmic/agents/research/scaffold.py`'s exported
`README.md` for the full setup sequence once you've exported your own copy
of the harness).
