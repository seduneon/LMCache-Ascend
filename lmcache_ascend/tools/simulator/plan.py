"""Pipeline types: schedule intent (``ScheduleResult``) → execute (``BatchWork``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Mapping

from content_key import ContentKey

if TYPE_CHECKING:
    from memory import KVBlock
    from request import Request
    from resource import BandwidthResource, ComputeResource

BlockAction = (
    Literal["compute"]
    | Literal["wait"]
    | tuple[Literal["pull"], str]
)
BlockActions = dict[str, BlockAction]


@dataclass(frozen=True)
class BlockRef:
    """Logical block identity for a single HBM slot in a request prefix."""

    hbm_hash: str
    content: ContentKey

    @classmethod
    def for_hbm(cls, req: Request | None, hbm_hash: str) -> BlockRef:
        return cls(hbm_hash=hbm_hash, content=ContentKey.for_hbm_block(req, hbm_hash))


@dataclass
class EntryPlan:
    """Per-request KV intent: HBM evictions + per-block actions."""

    evicts: list[KVBlock] = field(default_factory=list)
    blocks: BlockActions = field(default_factory=dict)


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
class BatchWork:
    """Fully identified batch flowing through execute → commit."""

    batch_id: int
    engine_id: str
    scheduled_at: float
    entries: list[WorkEntry] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)
    total_num_scheduled_tokens: int = 0

    @classmethod
    def from_schedule(
        cls,
        scheduled: ScheduleResult,
        *,
        batch_id: int,
        engine_id: str,
        scheduled_at: float,
    ) -> BatchWork:
        return cls(
            batch_id=batch_id,
            engine_id=engine_id,
            scheduled_at=scheduled_at,
            entries=list(scheduled.entries),
            preempted=list(scheduled.preempted),
            total_num_scheduled_tokens=scheduled.total_num_scheduled_tokens,
        )


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
