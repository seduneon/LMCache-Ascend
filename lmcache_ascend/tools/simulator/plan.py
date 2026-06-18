"""Plan layer: authoritative ``BatchPlan`` consumed mechanically by Execute."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Mapping

from .content_key import ContentKey
from .task_outcomes import TaskOutcome

if TYPE_CHECKING:
    from .memory import KVBlock
    from .request import Request
    from .resource import BandwidthResource, ComputeResource

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
    spill_reqs: dict[int, Request | None] = field(default_factory=dict)
    hbm_mirror_tiers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    resident_outcomes: dict[str, TaskOutcome] = field(default_factory=dict)
    pull_outcomes: dict[str, TaskOutcome] = field(default_factory=dict)


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


# Backward-compatible alias (Phase 3 merged artifact)
BatchWork = BatchPlan


@dataclass(frozen=True)
class QueueSnapshot:
    """Read-only resource queue depths captured at plan time."""

    compute_depth: int
    link_depth: Mapping[str, int]


@dataclass(frozen=True)
class SimContext:
    """Read-only simulation slice passed to planners."""

    now: float
    local_memory: str
    block_size: int
    queues: QueueSnapshot

    @classmethod
    def capture(
        cls,
        *,
        now: float,
        local_memory: str,
        block_size: int,
        compute_res: ComputeResource | None = None,
        transfer_links: Mapping[str, BandwidthResource] | None = None,
    ) -> SimContext:
        compute_depth = compute_res.queued_load() if compute_res is not None else 0
        link_depth = {
            key: link.queued_load()
            for key, link in (transfer_links or {}).items()
        }
        return cls(
            now=now,
            local_memory=local_memory,
            block_size=block_size,
            queues=QueueSnapshot(compute_depth=compute_depth, link_depth=link_depth),
        )


@dataclass
class ExecuteResult:
    """Output of ``BatchExecutor.execute()``."""

    tasks: list
    peak_duplicate_count: int
