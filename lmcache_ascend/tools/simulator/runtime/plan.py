"""Plan layer: ``BatchPlan`` consumed by ``BatchRunner``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from simulator.core.kv_content import ContentKey

if TYPE_CHECKING:
    from simulator.core.memory import KVBlock
    from simulator.core.request import Request

BlockAction = (
    Literal["compute"]
    | Literal["wait"]
    | tuple[Literal["pull"], str]
)
BlockActions = dict[str, BlockAction]


@dataclass(frozen=True)
class StoreOp:
    """Async write of one chunk slot to a downstream tier."""

    tier_key: str
    content: ContentKey
    storage_key: str
    hbm_block_hash: str


@dataclass
class EvictPlan:
    """One local-tier victim and optional spill store ops."""

    block: KVBlock
    store_ops: list[StoreOp] = field(default_factory=list)


@dataclass
class EntryPlan:
    """Per-request KV intent: all executor-visible memory actions."""

    evicts: list[EvictPlan] = field(default_factory=list)
    blocks: BlockActions = field(default_factory=dict)
    store_ops: dict[str, list[StoreOp]] = field(default_factory=dict)


@dataclass
class WorkEntry:
    """One request's slice of a batch."""

    req: Request
    block_hashes: list[str]
    plan: EntryPlan
    num_scheduled_tokens: int = 0
    remote_kv: bool = False


@dataclass
class ScheduleResult:
    """Output of ``Scheduler.schedule()`` before batch id assignment."""

    entries: list[WorkEntry] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)
    total_num_scheduled_tokens: int = 0


@dataclass
class BatchPlan:
    """Single batch artifact: schedule output + execute input."""

    batch_id: int
    engine_id: str
    scheduled_at: float
    entries: list[WorkEntry] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)
    total_num_scheduled_tokens: int = 0


def dedupe_batch_evicts(plan: BatchPlan) -> None:
    """Drop duplicate HBM victims scheduled across entries in one batch."""
    seen: set[int] = set()
    for entry in plan.entries:
        unique: list[EvictPlan] = []
        for evict_plan in entry.plan.evicts:
            token = id(evict_plan.block)
            if token in seen:
                continue
            seen.add(token)
            unique.append(evict_plan)
        entry.plan.evicts = unique
