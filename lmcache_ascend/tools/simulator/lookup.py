"""Lookup policies: resolve per-block compute, pull, or local hit."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from .chunk_hash import tier_covers_hbm_block, tier_inflight_hbm_block, transfer_work_units
from .cost_model import block_recompute_work
from .eviction import EvictionPolicy, LRUEviction
from .memory import Memory
from .plan import BlockAction, BlockActions, EntryPlan, SimContext
from .request import Request

if TYPE_CHECKING:
    from .resource import LinearShareResource

_LOCAL = Literal["local"]
BlockResolution = BlockAction | _LOCAL | None


def local_satisfied(local: Memory, block_hash: str) -> bool:
    return (
        local.best_resident(block_hash) is not None
        or local.inflight_incoming(block_hash) is not None
    )


def slots_needed(actions: BlockActions) -> int:
    return sum(
        1
        for action in actions.values()
        if action == "compute"
        or (isinstance(action, tuple) and action[0] == "pull")
    )


def remote_wait_source(
    memories: dict[str, Memory],
    pull_sources: list[str],
    block_hash: str,
    *,
    req: Request | None = None,
) -> str | None:
    """Pull source with in-flight chunk (resident copy not ready yet)."""
    for src_key in pull_sources:
        src = memories[src_key]
        if tier_covers_hbm_block(src, req, block_hash):
            continue
        if tier_inflight_hbm_block(src, req, block_hash):
            return src_key
    return None


def first_resident_pull_source(
    memories: dict[str, Memory],
    pull_sources: list[str],
    block_hash: str,
    *,
    req: Request | None = None,
) -> str | None:
    for src_key in pull_sources:
        src = memories[src_key]
        if tier_covers_hbm_block(src, req, block_hash):
            return src_key
    return None


class LookupPolicy(ABC):
    """Resolve per-block actions: local hit, pull from a tier, or recompute."""

    def __init__(
        self,
        local_memory: str,
        eviction_policy: EvictionPolicy | None = None,
    ):
        self.local_memory = local_memory
        self.eviction_policy = eviction_policy or LRUEviction()

    @property
    @abstractmethod
    def pull_sources(self) -> list[str]:
        pass

    @abstractmethod
    def resolve_block(
        self,
        memories: dict[str, Memory],
        block_hash: str,
        *,
        allow_compute: bool,
        req: Request | None = None,
        block_size: int = 1,
    ) -> BlockResolution:
        pass

    def bind_resources(
        self,
        *,
        compute_res: LinearShareResource | None,
        transfer_links: dict[str, LinearShareResource] | None,
        work_per_transfer: float,
        work_per_block: float,
        work_per_prefill_token: float | None = None,
        work_per_decode_req: float | None = None,
        block_size: int = 1,
    ) -> None:
        """Optional hook for Engine to wire runtime resources (cost-based policies)."""

    def begin_allocate_batch(self) -> None:
        """Reset per-batch reservation state before ``schedule()`` allocations."""

    def lookup(
        self,
        memories: dict[str, Memory],
        block_hashes: list[str],
        *,
        allow_compute: bool = True,
        req: Request | None = None,
        block_size: int = 1,
        ctx: SimContext | None = None,
    ) -> EntryPlan | None:
        """Try to allocate ``block_hashes`` without preemption (SSOT for actions + evicts)."""
        actions = self.resolve_actions(
            memories,
            block_hashes,
            allow_compute=allow_compute,
            req=req,
            block_size=block_size,
            ctx=ctx,
        )
        if actions is None:
            return None
        evicts = self.eviction_policy.plan(
            memories[self.local_memory], slots_needed(actions), set(block_hashes)
        )
        if evicts is None:
            return None
        return EntryPlan(evicts=list(evicts), blocks=dict(actions))

    def resolve_actions(
        self,
        memories: dict[str, Memory],
        block_hashes: list[str],
        *,
        allow_compute: bool = True,
        req: Request | None = None,
        block_size: int = 1,
        ctx: SimContext | None = None,
    ) -> BlockActions | None:
        del ctx
        actions: BlockActions = {}
        for block_hash in block_hashes:
            resolution = self.resolve_block(
                memories,
                block_hash,
                allow_compute=allow_compute,
                req=req,
                block_size=block_size,
            )
            if resolution is None:
                return None
            if resolution == "local":
                continue
            actions[block_hash] = resolution
        return actions


class ComputeOnlyLookupPolicy(LookupPolicy):
    """Local hit or recompute. No remote tiers."""

    @property
    def pull_sources(self) -> list[str]:
        return []

    def resolve_block(
        self,
        memories: dict[str, Memory],
        block_hash: str,
        *,
        allow_compute: bool,
        req: Request | None = None,
        block_size: int = 1,
    ) -> BlockResolution:
        local = memories[self.local_memory]
        if local_satisfied(local, block_hash):
            return "local"
        if allow_compute:
            return "compute"
        return None


class OrderedPullLookupPolicy(LookupPolicy):
    """First ``pull_sources`` entry with a resident copy, else recompute."""

    def __init__(
        self,
        local_memory: str,
        pull_sources: list[str],
        eviction_policy: EvictionPolicy | None = None,
    ):
        super().__init__(local_memory, eviction_policy)
        self._pull_sources = list(pull_sources)

    @property
    def pull_sources(self) -> list[str]:
        return self._pull_sources

    def resolve_block(
        self,
        memories: dict[str, Memory],
        block_hash: str,
        *,
        allow_compute: bool,
        req: Request | None = None,
        block_size: int = 1,
    ) -> BlockResolution:
        local = memories[self.local_memory]
        if local_satisfied(local, block_hash):
            return "local"

        if remote_wait_source(memories, self._pull_sources, block_hash, req=req) is not None:
            return "wait"

        src_key = first_resident_pull_source(
            memories, self._pull_sources, block_hash, req=req
        )
        if src_key is not None:
            return ("pull", src_key)

        if allow_compute:
            return "compute"
        return None


class CostBasedPullLookupPolicy(LookupPolicy):
    """Pick min-cost pull source vs recompute using per-link bandwidth queues."""

    def __init__(
        self,
        local_memory: str,
        pull_sources: list[str],
        eviction_policy: EvictionPolicy | None = None,
    ):
        super().__init__(local_memory, eviction_policy)
        self._pull_sources = list(pull_sources)
        self._compute_res: LinearShareResource | None = None
        self._transfer_links: dict[str, LinearShareResource] = {}
        self._work_per_transfer = 1.0
        self._work_per_block = 1.0
        self._work_per_prefill_token = 1.0
        self._work_per_decode_req = 1.0
        self._block_size = 1
        self._pending_pulls: dict[str, int] = {}
        self._forward_reserved_in_alloc = False
        self._alloc_req: Request | None = None
        self._alloc_block_size = 1
        self._alloc_memories: dict[str, Memory] | None = None

    @property
    def pull_sources(self) -> list[str]:
        return self._pull_sources

    def bind_resources(
        self,
        *,
        compute_res: LinearShareResource | None,
        transfer_links: dict[str, LinearShareResource] | None,
        work_per_transfer: float,
        work_per_block: float,
        work_per_prefill_token: float | None = None,
        work_per_decode_req: float | None = None,
        block_size: int = 1,
    ) -> None:
        self._compute_res = compute_res
        self._transfer_links = transfer_links or {}
        self._work_per_transfer = work_per_transfer
        self._work_per_block = work_per_block
        self._work_per_prefill_token = (
            work_per_prefill_token
            if work_per_prefill_token is not None
            else work_per_block
        )
        self._work_per_decode_req = (
            work_per_decode_req if work_per_decode_req is not None else work_per_block
        )
        self._block_size = block_size

    def begin_allocate_batch(self) -> None:
        self._pending_pulls = {}
        self._forward_reserved_in_alloc = False

    def resolve_actions(
        self,
        memories: dict[str, Memory],
        block_hashes: list[str],
        *,
        allow_compute: bool = True,
        req: Request | None = None,
        block_size: int = 1,
        ctx: SimContext | None = None,
    ) -> BlockActions | None:
        self._alloc_req = req
        self._alloc_block_size = block_size
        self._alloc_memories = memories
        return super().resolve_actions(
            memories,
            block_hashes,
            allow_compute=allow_compute,
            req=req,
            block_size=block_size,
            ctx=ctx,
        )

    def _pull_time(self, link: LinearShareResource, src_key: str, block_hash: str) -> float:
        pending = self._pending_pulls.get(src_key, 0)
        work = self._work_per_transfer
        if self._alloc_memories is not None:
            src = self._alloc_memories[src_key]
            work *= transfer_work_units(src, self._alloc_req, block_hash)
        return link.time_for(
            work,
            link.queued_load() + pending + 1,
        )

    def _compute_time(self) -> float:
        if self._forward_reserved_in_alloc:
            return 0.0
        compute_res = self._compute_res
        assert compute_res is not None
        work = block_recompute_work(
            self._alloc_req,
            block_size=self._alloc_block_size,
            work_per_prefill_token=self._work_per_prefill_token,
            work_per_decode_req=self._work_per_decode_req,
            work_per_block=self._work_per_block,
        )
        pending_forward = 1
        return compute_res.time_for(
            work,
            compute_res.queued_load() + pending_forward,
        )

    def _note_resolution(self, resolution: BlockResolution, block_hash: str) -> None:
        if isinstance(resolution, tuple) and resolution[0] == "pull":
            src_key = resolution[1]
            self._pending_pulls[src_key] = self._pending_pulls.get(src_key, 0) + 1
        elif resolution == "compute":
            self._forward_reserved_in_alloc = True

    def resolve_block(
        self,
        memories: dict[str, Memory],
        block_hash: str,
        *,
        allow_compute: bool,
        req: Request | None = None,
        block_size: int = 1,
    ) -> BlockResolution:
        local = memories[self.local_memory]
        if local_satisfied(local, block_hash):
            return "local"

        if remote_wait_source(memories, self._pull_sources, block_hash, req=req) is not None:
            return "wait"

        if not self._transfer_links or self._compute_res is None:
            src_key = first_resident_pull_source(
                memories, self._pull_sources, block_hash, req=req
            )
            if src_key is not None:
                resolution: BlockResolution = ("pull", src_key)
                self._note_resolution(resolution, block_hash)
                return resolution
            if allow_compute:
                resolution = "compute"
                self._note_resolution(resolution, block_hash)
                return resolution
            return None

        pull_candidates: list[tuple[float, int, str]] = []

        for order, src_key in enumerate(self._pull_sources):
            src = memories[src_key]
            if not tier_covers_hbm_block(src, req, block_hash):
                if tier_inflight_hbm_block(src, req, block_hash):
                    return "wait"
                continue

            link = self._transfer_links.get(src_key)
            if link is None:
                continue
            pull_candidates.append(
                (
                    self._pull_time(link, src_key, block_hash),
                    order,
                    src_key,
                )
            )

        best_pull: tuple[float, int, str] | None = (
            min(pull_candidates, key=lambda item: (item[0], item[1]))
            if pull_candidates
            else None
        )

        if best_pull is None:
            if remote_wait_source(memories, self._pull_sources, block_hash, req=req):
                return "wait"
            if allow_compute:
                resolution = "compute"
                self._note_resolution(resolution, block_hash)
                return resolution
            return None

        if not allow_compute:
            resolution = ("pull", best_pull[2])
            self._note_resolution(resolution, block_hash)
            return resolution

        t_compute = self._compute_time()
        t_pull = best_pull[0]

        if t_pull < t_compute:
            resolution = ("pull", best_pull[2])
        elif t_compute < t_pull:
            resolution = "compute"
        else:
            resolution = ("pull", best_pull[2])
        self._note_resolution(resolution, block_hash)
        return resolution
