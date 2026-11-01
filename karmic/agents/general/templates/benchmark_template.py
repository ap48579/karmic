"""Benchmark template for an archetype role -- copy into your repo and fill in the TODOs.

Two halves, matching WRITING_YOUR_OWN_BENCHMARK.md's subprocess rule:

  python benchmark.py --json-out score.json      # runs INSIDE the candidate worktree,
                                                 # reading that worktree's prompt files
  run_benchmark(repo_root, python_bin)           # your HarnessSpec.benchmark_fn; calls the above

Fixtures are frozen karmic.agents.general.Fixture records (jsonl). Build them from real past calls with
Fixture.from_call(...) and add the mechanical facts a correct answer must have -- 5-10 per role
to start, weighted toward cases the current prompt gets wrong. Only facts checkable from the
fixture itself -- never a slow, noisy downstream outcome (docs/09_agent_archetypes_spec.md S8).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures" / "my_role.jsonl"  # TODO
PROMPTS = {"my_role": HERE / "prompts" / "my_role.md"}  # TODO: role -> prompt file
MODEL = "qwen3:14b"  # TODO: keep fixed across baseline and candidate


def main() -> None:
    from karmic.agents.general import RoleConfig, evidence_analyst, load_fixtures, ollama_json, score_fixtures

    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()
    roles = {name: RoleConfig.from_file(name, path, temperature=0.0) for name, path in PROMPTS.items()}
    spec = evidence_analyst()  # TODO: the archetype these fixtures exercise
    score = score_fixtures(spec, roles, load_fixtures(FIXTURES), ollama_json(MODEL))
    Path(args.json_out).write_text(json.dumps(asdict(score), default=str))
    print(score.to_prompt_text())


class _Score:
    def __init__(self, data: dict):
        self.data, self.scalar_score = data, data["scalar_score"]

    def to_prompt_text(self) -> str:
        return json.dumps({k: v for k, v in self.data.items() if k != "runs"}) + "\n" + "\n".join(
            f"  - {r['fixture_id']}: {'pass' if r['passed'] else 'FAIL ' + '; '.join(r['failures'])}"
            for r in self.data["runs"])


def run_benchmark(repo_root: Path, python_bin: str = "python3"):
    """HarnessSpec.benchmark_fn. TODO: adjust the path to this file inside your repo."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "score.json"
        proc = subprocess.run([python_bin, "benchmark.py", "--json-out", str(out)], cwd=str(repo_root),
                              capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0 or not out.exists():
            return _Score({"scalar_score": -1e9, "runs": [], "error": proc.stderr[-500:]})
        return _Score(json.loads(out.read_text()))


if __name__ == "__main__":
    main()
