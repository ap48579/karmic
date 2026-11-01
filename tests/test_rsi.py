"""RSI engine wiring tests -- no API key required. Run with:
    python -m tests.test_rsi
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from karmic.rsi import HARNESSES


def test_harness_allowlists_point_at_real_files():
    # apply_edits() resolves these against a worktree of this repo; a path that doesn't exist
    # (e.g. after moving a package) silently turns every proposal into "file not found".
    missing = [f for h in HARNESSES.values() for f in h.allowed_files if not (REPO_ROOT / f).is_file()]
    assert not missing, missing
    print(f"PASS: test_harness_allowlists_point_at_real_files ({sum(len(h.allowed_files) for h in HARNESSES.values())} files)")


if __name__ == "__main__":
    test_harness_allowlists_point_at_real_files()
    print("\nAll RSI tests passed.")
