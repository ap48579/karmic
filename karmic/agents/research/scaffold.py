"""export_strategy() -- gives a consumer their own copy of the research
harness, in their own repo, so RSI can operate on it in isolation (see
docs/05_rsi_harness2_design_spec.md and the "publishing as a general
package" design discussion).

Correction made while implementing this, worth recording: the original
design assumed only the ~10 RSI-editable "strategy surface" files needed
exporting, with Python's import system "shadowing" them over the library's
bundled copy via sys.path ordering. That doesn't work -- Python doesn't
merge two directories for one package; whichever directory wins on
sys.path must be a complete, self-sufficient `gpt_researcher` package, or
internal imports inside it (agent.py -> actions -> llm_provider -> ...)
break the moment something reaches for a file that wasn't exported. The
fix: export the *entire* vendored tree, not just the editable slice. RSI's
editable surface (what it's actually allowed to propose edits to) stays
exactly the same small list -- this only changes what gets copied to disk,
not what gets edited.
"""

from __future__ import annotations

import shutil
from pathlib import Path

_THIS_DIR = Path(__file__).parent
_GPT_RESEARCHER_SRC = _THIS_DIR / "gpt_researcher"

# The RSI-editable surface, repo-relative to wherever export_strategy() puts
# the copy -- same list rsi.RESEARCH_HARNESS.allowed_files uses today for
# this repo's own copy. A consumer building their own HarnessSpec should
# point at these same relative paths under their own target_dir.
STRATEGY_SURFACE_FILES = [
    "gpt_researcher/prompts.py",
    "gpt_researcher/actions/agent_creator.py",
    "gpt_researcher/actions/retriever.py",
    "gpt_researcher/actions/web_scraping.py",
    "gpt_researcher/skills/researcher.py",
    "gpt_researcher/skills/writer.py",
    "gpt_researcher/skills/curator.py",
    "gpt_researcher/actions/query_processing.py",
    "gpt_researcher/actions/report_generation.py",
    "research_pipeline.py",
]

_RESEARCH_PIPELINE_TEMPLATE = (_THIS_DIR.parent.parent / "research_pipeline.py")
_README_TEMPLATE = """# Your own copy of the research harness

This directory is a full, self-sufficient copy of the vendored GPT
Researcher tree, exported so you can run the self-improvement engine
(`karmic.rsi`) against your own git repo instead of sharing one physical copy
with anyone else.

## What you should actually edit

Only these files are the "strategy surface" -- the only things RSI will
ever propose edits to, and the only things worth hand-editing yourself:

```
{file_list}
```

Everything else under `gpt_researcher/` (retrievers, scrapers, the LLM
provider layer, document loaders, ...) is infrastructure. Leave it alone;
it's identical to the pip-installed library's own copy and gets updated by
upgrading the package, not by editing it here.

## Before running RSI against this copy

1. Commit this directory to your own repo's git history. `rsi.run_cycle()`
   needs `repo_root` to be a real git repo with these files already
   committed, or every proposed edit will be rejected with
   "file not found in worktree."
2. **Write your own benchmark** (`benchmark_template.py` has 2-3 example
   tasks marked clearly -- replace them with your own domain's checkable
   ground truth). This is the single most important step: RSI's
   benchmark-gate is only as meaningful as the questions you put in it. A
   generic trivia benchmark tells you nothing about whether an edit helped
   your actual use case.
3. Build a `HarnessSpec` pointing at your own files (see
   `karmic.rsi.harness_spec.HarnessSpec` and `karmic.rsi.harnesses.RESEARCH_HARNESS` in
   the installed package for the shape to follow) and an `llm_call` (reuse
   `karmic.rsi.qwen_ollama_llm_call()` / `karmic.rsi.claude_llm_call()`, or write your own
   -- any `async def (system: str, user: str) -> str` works).
4. Put this directory earlier on `sys.path` than the installed package's
   own copy before importing `gpt_researcher`, so your edited files are
   the ones actually used.
"""

_BENCHMARK_TEMPLATE = '''"""Your own ground-truth benchmark -- REPLACE these example tasks with
questions whose correct answer you can check deterministically (a date, a
name, a number) in YOUR domain. RSI's benchmark-gate is only as meaningful
as what you put here: a generic trivia benchmark tells you nothing about
whether a proposed edit actually helped your actual use case.

Shape mirrors karmic.rsi.research_benchmark / karmic.rsi.coding_benchmark in the
installed package -- a BenchmarkFn is any
`(repo_root: Path, python_bin: str) -> ScoreLike` where ScoreLike has a
`scalar_score: float` and a `to_prompt_text() -> str`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# EXAMPLE tasks -- replace with your own domain's checkable facts before
# using this for anything real.
EXAMPLE_TASKS = [
    # ("question you'd ask the research pipeline", ["required", "keywords", "in", "the", "answer"]),
    ("What year did your company IPO, and on which exchange?", ["REPLACE", "ME"]),
]


@dataclass
class TaskRun:
    name: str
    correct: bool
    notes: str = ""


@dataclass
class BenchmarkScore:
    task_runs: list[TaskRun] = field(default_factory=list)
    success_rate: float = 0.0
    scalar_score: float = 0.0

    def to_prompt_text(self) -> str:
        lines = [f"scalar_score={self.scalar_score:.1f} success_rate={self.success_rate:.0%}"]
        lines += [f"  - {r.name}: correct={r.correct} {r.notes}" for r in self.task_runs]
        return "\\n".join(lines)


def run_benchmark(repo_root: Path, python_bin: str = "python3") -> BenchmarkScore:
    """TODO: wire this up to actually run your pipeline against each task
    and check its output for the required keywords -- see
    karmic.rsi.research_benchmark._run_one_task in the installed package for the
    subprocess + keyword-match pattern to copy."""
    raise NotImplementedError("Replace EXAMPLE_TASKS above and implement this against your own pipeline entrypoint.")
'''


def export_strategy(target_dir: Path) -> list[Path]:
    """Copies the entire vendored gpt_researcher/ tree plus a starter
    research_pipeline.py, benchmark_template.py, and README.md into
    target_dir. Returns the list of files written.

    target_dir should be inside the consumer's own git repo -- nothing
    here touches git itself; committing is the caller's job (the README
    says so explicitly).
    """
    target_dir = Path(target_dir)
    written: list[Path] = []

    gpt_researcher_dst = target_dir / "gpt_researcher"
    if gpt_researcher_dst.exists():
        shutil.rmtree(gpt_researcher_dst)
    shutil.copytree(_GPT_RESEARCHER_SRC, gpt_researcher_dst,
                     ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    written.extend(p for p in gpt_researcher_dst.rglob("*") if p.is_file())

    research_pipeline_dst = target_dir / "research_pipeline.py"
    research_pipeline_dst.write_text(_RESEARCH_PIPELINE_TEMPLATE.read_text())
    written.append(research_pipeline_dst)

    benchmark_dst = target_dir / "benchmark_template.py"
    benchmark_dst.write_text(_BENCHMARK_TEMPLATE)
    written.append(benchmark_dst)

    readme_dst = target_dir / "README.md"
    file_list = "\n".join(STRATEGY_SURFACE_FILES)
    readme_dst.write_text(_README_TEMPLATE.format(file_list=file_list))
    written.append(readme_dst)

    return written
