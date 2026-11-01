"""Trace aggregation — turns the existing Cortex episode log (already written
by every pipeline run) into a report the Self-Improvement Planner reads.

No new logging format: Episode.action_arc / healer_attempts / outcome *are*
the trace. This just summarizes them across runs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..cortex import CortexStore, Outcome


@dataclass
class SelfImprovementReport:
    total_episodes: int
    success_rate: float
    avg_healer_attempts: float
    surprise_count: int  # predicted-vs-actual difficulty mismatches (Episode.salience_boost() > 0)
    top_failure_summaries: list[str] = field(default_factory=list)
    top_abstracted_patterns: list[str] = field(default_factory=list)
    recent_samples: list[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        if self.total_episodes == 0:
            return "(no production episodes yet — nothing to learn from)"
        lines = [
            f"Episodes observed: {self.total_episodes}",
            f"Success rate: {self.success_rate:.0%}",
            f"Avg healer attempts per episode: {self.avg_healer_attempts:.2f}",
            f"Predicted-vs-actual difficulty surprises: {self.surprise_count}",
        ]
        if self.top_failure_summaries:
            lines.append("\nRecurring failure summaries:")
            lines += [f"- {s}" for s in self.top_failure_summaries]
        if self.top_abstracted_patterns:
            lines.append("\nRecurring successful patterns:")
            lines += [f"- {s}" for s in self.top_abstracted_patterns]
        if self.recent_samples:
            lines.append("\nRecent episode samples:")
            lines += [f"- {s}" for s in self.recent_samples]
        return "\n".join(lines)


def build_report(
    cortex_dir: str = "./cortex_data", domain: str | None = None, recent_n: int = 8, top_n: int = 5,
) -> SelfImprovementReport:
    store = CortexStore(persist_dir=cortex_dir)
    episodes = store.all_episodes()
    if domain is not None:
        episodes = [e for e in episodes if e.domain == domain]

    if not episodes:
        return SelfImprovementReport(
            total_episodes=0, success_rate=0.0, avg_healer_attempts=0.0, surprise_count=0,
        )

    successes = sum(1 for e in episodes if e.outcome == Outcome.SUCCESS)
    avg_attempts = sum(e.healer_attempt_count for e in episodes) / len(episodes)
    surprises = sum(1 for e in episodes if e.salience_boost() > 0)

    failure_counts = Counter(e.failure_summary for e in episodes if e.failure_summary)
    pattern_counts = Counter(e.abstracted_pattern for e in episodes if e.abstracted_pattern)

    recent = sorted(episodes, key=lambda e: e.created_at, reverse=True)[:recent_n]
    recent_samples = [
        f"[{e.outcome.value}, {e.healer_attempt_count} healer attempts] {e.problem_summary[:120]}"
        for e in recent
    ]

    return SelfImprovementReport(
        total_episodes=len(episodes),
        success_rate=successes / len(episodes),
        avg_healer_attempts=avg_attempts,
        surprise_count=surprises,
        top_failure_summaries=[s for s, _ in failure_counts.most_common(top_n)],
        top_abstracted_patterns=[s for s, _ in pattern_counts.most_common(top_n)],
        recent_samples=recent_samples,
    )
