"""Episode session lifecycle for the research harness: a workspace
directory per episode, an append-only checkpoint log (crash-visibility, not
yet resumability -- see docs/01_harness_spec.md S9), and a steer-file check
so a long-running episode can pick up an operator note between stages.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EpisodeSession:
    episode_id: str
    workspace: Path

    @classmethod
    def start(cls, workspace_dir: str) -> "EpisodeSession":
        episode_id = str(uuid.uuid4())
        workspace = Path(workspace_dir) / episode_id
        workspace.mkdir(parents=True, exist_ok=True)
        return cls(episode_id=episode_id, workspace=workspace)

    def checkpoint(self, stage: str, data: dict) -> None:
        record = {"stage": stage, "timestamp": time.time(), **data}
        with open(self.workspace / "checkpoints.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def check_steer(self) -> str | None:
        """Reads workspace/steer.txt if present, deletes it, and returns its
        contents to fold into the next stage's prompt as an operator note."""
        steer_path = self.workspace / "steer.txt"
        if not steer_path.exists():
            return None
        note = steer_path.read_text().strip()
        steer_path.unlink()
        return note or None
