"""Plan layer: ``BatchPlan`` consumed by ``BatchExecutor``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .content_key import ContentKey

if TYPE_CHECKING:
    from .memory import KVBlock
    from .request import Request

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


@dataclass(frozen=True)
class RetentionProfile:
    """Frozen retention knobs planned into outcomes (no live policy in Execute)."""

    kind: Literal["unbounded", "single_copy", "consume_on_pull", "global_cap"] = "unbounded"
    max_total: int = 0
    tier_keys: tuple[str, ...] = ()
    per_tier_cap: int | None = None


@dataclass
class EntryPlan:
    """Per-request KV intent: all executor-visible memory actions."""

    evicts: list[KVBlock] = field(default_factory=list)
    blocks: BlockActions = field(default_factory=dict)
    store_ops: dict[str, list[StoreOp]] = field(default_factory=dict)
    spill_store_ops: dict[int, list[StoreOp]] = field(default_factory=dict)


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
    retention: RetentionProfile = field(default_factory=RetentionProfile)
