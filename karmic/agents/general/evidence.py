"""Evidence index: every item a role can cite gets a stable id, a one-line
label the model reads, and the structured payload validators check
numbers against.

Builders that turn domain rows into items stay in the consumer, which
knows what its rows mean. `entity` is whatever the item is about (a
service, a customer, a document), for consumers that scope citations
per entity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date


@dataclass
class Item:
    evidence_id: str
    kind: str
    label: str
    payload: dict
    entity: str | None = None
    url: str | None = None
    published_at: date | None = None


@dataclass
class EvidenceIndex:
    items: dict[str, Item] = field(default_factory=dict)
    prefix: str = "ev"
    _counter: int = 0

    def add(self, kind: str, label: str, payload: dict | None = None, entity: str | None = None,
            url: str | None = None, published_at: date | None = None) -> str:
        self._counter += 1
        eid = f"{self.prefix}{self._counter}"
        self.items[eid] = Item(eid, kind, label, payload or {}, entity, url, published_at)
        return eid

    def render(self, ids: list[str] | set[str] | None = None) -> str:
        """The block shown to the model: one line per item, id first, in insertion order."""
        wanted = self.items.keys() if ids is None else [i for i in self.items if i in set(ids)]
        return "\n".join(f"[{eid}] {self.items[eid].label}" for eid in wanted)

    def text(self, evidence_id: str) -> str:
        """Everything the number check may match against: the label the model read plus the
        raw payload (so a figure the label rounded is still found)."""
        item = self.items[evidence_id]
        return item.label + " " + json.dumps(item.payload, default=str)

    def to_dict(self) -> dict:
        return {"prefix": self.prefix, "items": [
            {"evidence_id": i.evidence_id, "kind": i.kind, "label": i.label, "payload": i.payload,
             "entity": i.entity, "url": i.url,
             "published_at": i.published_at.isoformat() if i.published_at else None}
            for i in self.items.values()
        ]}

    @classmethod
    def from_dict(cls, data: dict) -> "EvidenceIndex":
        idx = cls(prefix=data.get("prefix", "ev"))
        for raw in data["items"]:
            published = date.fromisoformat(raw["published_at"]) if raw.get("published_at") else None
            idx.items[raw["evidence_id"]] = Item(raw["evidence_id"], raw["kind"], raw["label"], raw["payload"],
                                                 raw.get("entity"), raw.get("url"), published)
        suffixes = [e[len(idx.prefix):] for e in idx.items if e.startswith(idx.prefix)]
        idx._counter = max((int(s) for s in suffixes if s.isdigit()), default=0)  # add() must not reuse an id
        return idx
