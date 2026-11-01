"""The research harness: vendored GPT Researcher (agents/research/gpt_researcher/,
see VENDOR.md here) plus our own Qwen configuration shim. Importing this
package makes `import gpt_researcher` work without callers needing to know
where the vendored tree lives.
"""

from __future__ import annotations

import sys
from pathlib import Path

_RESEARCH_DIR = Path(__file__).parent
if str(_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_DIR))

from .qwen_config import configure_qwen

__all__ = ["configure_qwen"]
