"""Declarative task completion outcomes (replace inline callbacks)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from .memory import KVBlock

OutcomeKind = Literal[
    "hbm_resident",
    "tier_resident",
    "pull_complete",
    "hbm_evict_before",
    "spill_complete",
]


@dataclass(frozen=True)
class TaskOutcome:
    kind: OutcomeKind
    req_id: str
    block_hash: str
    tier_key: str | None = None
    src_key: str | None = None
    remove_hbm_hash: str | None = None


class OutcomeHandler(Protocol):
    def __call__(self, outcome: TaskOutcome, block: KVBlock, now: float) -> None: ...
