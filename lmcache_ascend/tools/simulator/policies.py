from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from content_key import ContentKey
from chunk_hash import (
    chunk_key_for_hbm_block,
    tier_covers_hbm_block,
    tier_inflight_hbm_block,
    transfer_work_units,
)
from cost_model import block_recompute_work
from eviction import EvictionPolicy, LRUEviction
from memory import BlockState, KVBlock, Memory, collect_content_copies
from plan import BlockAction, BlockActions
from request import Request
from tier_allocator import TierAllocator

if TYPE_CHECKING:
    from resource import BandwidthResource, ComputeResource

_LOCAL = Literal["local"]
BlockResolution = BlockAction | _LOCAL | None


@dataclass(frozen=True)
class StoreOp:
    """Async write of one chunk slot to a downstream tier."""

    tier_key: str
    content: ContentKey
    storage_key: str
    hbm_block_hash: str


@dataclass
class LookupResult:
    evicts: list[KVBlock] = field(default_factory=list)
    blocks: BlockActions = field(default_factory=dict)


# --- Placement ---


class PlacementPolicy(ABC):
    """Where to retain KV copies when a block becomes resident on local HBM."""

    @abstractmethod
    def place_copy(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block: KVBlock,
        req: Request,
        now: float,
    ) -> None:
        """Synchronously mirror ``block`` to additional tier(s) if policy allows."""

    def spill_on_evict(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block: KVBlock,
        now: float,
        req: Request | None = None,
    ) -> None:
        """Called before an HBM victim is removed. Default: drop (no spill)."""

    def bind_retention(self, retention: RetentionPolicy) -> None:
        """Optional hook for tier mirrors to enforce copy caps (``HBMAndDRAM``)."""

    def plan_async_stores(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block_hash: str,
        req: Request,
    ) -> list[StoreOp]:
        """Return paid async mirror ops after compute/pull (empty for sync-only placement)."""
        return []

    def plan_spill_stores(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block_hash: str,
        req: Request,
    ) -> list[StoreOp]:
        """Return paid async spill ops before HBM victim is removed."""
        return []


class HBMOnly(PlacementPolicy):
    """Keep computed KV on local HBM only (default)."""

    def place_copy(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block: KVBlock,
        req: Request,
        now: float,
    ) -> None:
        pass


class HBMAndDRAM(PlacementPolicy):
    """Mirror HBM residents to DRAM; spill on HBM evict; LRU-evict DRAM when full."""

    def __init__(
        self,
        dram_memory: str,
        *,
        dram_eviction_policy: EvictionPolicy | None = None,
    ):
        self.dram_memory = dram_memory
        self._dram_eviction = dram_eviction_policy or LRUEviction()
        self._allocator = TierAllocator(self._dram_eviction)
        self._retention: RetentionPolicy = UnboundedRetention()

    def bind_retention(self, retention: RetentionPolicy) -> None:
        self._retention = retention

    def _ensure_dram_resident(
        self,
        memories: dict[str, Memory],
        *,
        req: Request,
        block_hash: str,
        now: float,
    ) -> bool:
        dram = memories[self.dram_memory]
        chunk_key = chunk_key_for_hbm_block(req, block_hash, dram.chunk_blocks)
        if self._allocator.tier_covers(dram, chunk_key):
            return dram.best_resident(chunk_key) is not None

        copy = self._allocator.ensure_slot(
            dram,
            chunk_key,
            state=BlockState.RESIDENT,
            exclude={chunk_key},
            eviction=self._dram_eviction,
        )
        if copy is None:
            return False
        dram.touch(copy, now)
        self._retention.on_block_resident(
            memories, tier_key=self.dram_memory, block=copy, now=now
        )
        return True

    def place_copy(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block: KVBlock,
        req: Request,
        now: float,
    ) -> None:
        self._ensure_dram_resident(
            memories, req=req, block_hash=block.hash, now=now
        )

    def spill_on_evict(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block: KVBlock,
        now: float,
        req: Request | None = None,
    ) -> None:
        self._ensure_dram_resident(
            memories, req=req, block_hash=block.hash, now=now
        )


class TieredPlacement(PlacementPolicy):
    """Mirror/spill HBM to one or more downstream tiers; optional paid writes per tier."""

    def __init__(
        self,
        tier_keys: list[str],
        *,
        tier_eviction: dict[str, EvictionPolicy] | None = None,
        paid_write_tiers: frozenset[str] | None = None,
    ):
        self.tier_keys = list(tier_keys)
        self._tier_eviction = tier_eviction or {}
        self._paid_write_tiers = paid_write_tiers or frozenset()
        self._allocator = TierAllocator()
        self._retention: RetentionPolicy = UnboundedRetention()

    def bind_retention(self, retention: RetentionPolicy) -> None:
        self._retention = retention

    def _eviction_for(self, tier_key: str) -> EvictionPolicy:
        return self._allocator.eviction_for(tier_key, self._tier_eviction)

    def _ensure_tier_resident_sync(
        self,
        memories: dict[str, Memory],
        *,
        tier_key: str,
        req: Request,
        block_hash: str,
        now: float,
    ) -> bool:
        tier = memories[tier_key]
        chunk_key = chunk_key_for_hbm_block(req, block_hash, tier.chunk_blocks)
        if self._allocator.tier_covers(tier, chunk_key):
            return True

        copy = self._allocator.ensure_slot(
            tier,
            chunk_key,
            state=BlockState.RESIDENT,
            exclude={chunk_key},
            eviction=self._eviction_for(tier_key),
        )
        if copy is None:
            return False
        tier.touch(copy, now)
        self._retention.on_block_resident(
            memories, tier_key=tier_key, block=copy, now=now
        )
        return True

    def _reserve_tier_loading(
        self,
        memories: dict[str, Memory],
        *,
        tier_key: str,
        req: Request,
        block_hash: str,
    ) -> KVBlock | None:
        tier = memories[tier_key]
        chunk_key = chunk_key_for_hbm_block(req, block_hash, tier.chunk_blocks)
        return self._allocator.ensure_slot(
            tier,
            chunk_key,
            state=BlockState.RESERVED,
            exclude={chunk_key},
            eviction=self._eviction_for(tier_key),
        )

    def place_copy(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block: KVBlock,
        req: Request,
        now: float,
    ) -> None:
        for tier_key in self.tier_keys:
            if tier_key in self._paid_write_tiers:
                continue
            self._ensure_tier_resident_sync(
                memories, tier_key=tier_key, req=req, block_hash=block.hash, now=now
            )

    def spill_on_evict(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block: KVBlock,
        now: float,
        req: Request | None = None,
    ) -> None:
        if req is None:
            return
        for tier_key in self.tier_keys:
            if tier_key in self._paid_write_tiers:
                continue
            self._ensure_tier_resident_sync(
                memories, tier_key=tier_key, req=req, block_hash=block.hash, now=now
            )

    def plan_async_stores(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block_hash: str,
        req: Request,
    ) -> list[StoreOp]:
        return self._paid_stores(
            memories, req=req, block_hash=block_hash, paid_only=True
        )

    def plan_spill_stores(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block_hash: str,
        req: Request,
    ) -> list[StoreOp]:
        return self._paid_stores(
            memories, req=req, block_hash=block_hash, paid_only=True
        )

    def _paid_stores(
        self,
        memories: dict[str, Memory],
        *,
        req: Request,
        block_hash: str,
        paid_only: bool,
    ) -> list[StoreOp]:
        ops: list[StoreOp] = []
        for tier_key in self.tier_keys:
            if tier_key not in self._paid_write_tiers:
                continue
            tier = memories[tier_key]
            chunk_key = chunk_key_for_hbm_block(req, block_hash, tier.chunk_blocks)
            if tier.best_resident(chunk_key) is not None:
                continue
            if tier.inflight_incoming(chunk_key) is not None:
                continue
            if self._reserve_tier_loading(
                memories, tier_key=tier_key, req=req, block_hash=block_hash
            ) is None:
                continue
            ops.append(
                StoreOp(
                    tier_key=tier_key,
                    content=ContentKey.for_hbm_block(req, block_hash),
                    storage_key=chunk_key,
                    hbm_block_hash=block_hash,
                )
            )
        return ops


# --- Retention ---


class PullDisposition(StrEnum):
    RETAIN = "retain"
    CONSUME = "consume"


class RetentionPolicy(ABC):
    """Caps per-tier duplicate residents and post-pull source lifecycle."""

    def max_copies(self, memory: Memory, block_hash: str) -> int | None:
        """Max resident copies of ``block_hash`` in ``memory``; ``None`` = unbounded."""
        return None

    def pull_disposition(self) -> PullDisposition:
        return PullDisposition.RETAIN

    def on_block_resident(
        self,
        memories: dict[str, Memory],
        *,
        tier_key: str,
        block: KVBlock,
        now: float,
        req: Request | None = None,
    ) -> None:
        memory = memories[tier_key]
        cap = self.max_copies(memory, block.hash)
        if cap is None:
            return
        self._trim_to_cap(memory, block.hash, cap)

    def after_pull(
        self,
        memories: dict[str, Memory],
        *,
        src_key: str,
        dst_key: str,
        block_hash: str,
        now: float,
        req: Request | None = None,
    ) -> None:
        if self.pull_disposition() != PullDisposition.CONSUME:
            return
        src = memories[src_key]
        storage_key = chunk_key_for_hbm_block(req, block_hash, src.chunk_blocks)
        src_block = src.best_resident(storage_key)
        if src_block is None or not src.can_evict_block(src_block):
            return
        src.remove_block(src_block)

    @staticmethod
    def _trim_to_cap(memory: Memory, block_hash: str, cap: int) -> None:
        copies = memory.resident_copies(block_hash)
        while len(copies) > cap:
            evictable = [b for b in copies if memory.can_evict_block(b)]
            if not evictable:
                break
            victim = min(evictable, key=lambda block: block.last_touch)
            memory.remove_block(victim)
            copies = memory.resident_copies(block_hash)


class UnboundedRetention(RetentionPolicy):
    """Default: unlimited copies per hash; pull leaves source resident."""


class SingleCopyPerTier(RetentionPolicy):
    """At most one unheld resident copy per content hash per tier."""

    def max_copies(self, memory: Memory, block_hash: str) -> int:
        return 1


class ConsumeOnPull(RetentionPolicy):
    """Remove the pull source copy when it has no remaining holders."""

    def pull_disposition(self) -> PullDisposition:
        return PullDisposition.CONSUME


class GlobalCopyCap(RetentionPolicy):
    """Cap total resident copies of a content key across selected tiers."""

    def __init__(
        self,
        max_total: int,
        tier_keys: list[str],
        *,
        per_tier_cap: int | None = 1,
    ):
        self.max_total = max_total
        self.tier_keys = list(tier_keys)
        self._per_tier_cap = per_tier_cap

    def max_copies(self, memory: Memory, block_hash: str) -> int | None:
        if self._per_tier_cap is None:
            return None
        return self._per_tier_cap

    def on_block_resident(
        self,
        memories: dict[str, Memory],
        *,
        tier_key: str,
        block: KVBlock,
        now: float,
        req: Request | None = None,
    ) -> None:
        super().on_block_resident(
            memories, tier_key=tier_key, block=block, now=now, req=req
        )
        content = ContentKey.for_storage_key(block.hash)
        while len(collect_content_copies(memories, self.tier_keys, content, req=req)) > self.max_total:
            copies = collect_content_copies(memories, self.tier_keys, content, req=req)
            evictable = [
                (tier, blk)
                for tier, blk in copies
                if memories[tier].can_evict_block(blk)
                and memories[tier].inflight_incoming(blk.hash) is None
            ]
            if not evictable:
                break
            tier_order = {t: i for i, t in enumerate(reversed(self.tier_keys))}
            victim_tier, victim = min(
                evictable,
                key=lambda item: (tier_order.get(item[0], 0), item[1].last_touch),
            )
            memories[victim_tier].remove_block(victim)


# --- Lookup helpers ---


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


# --- Lookup ---


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
        compute_res: ComputeResource | None,
        transfer_links: dict[str, BandwidthResource] | None,
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
    ) -> LookupResult | None:
        """Try to allocate ``block_hashes`` without preemption (SSOT for actions + evicts)."""
        actions = self.resolve_actions(
            memories,
            block_hashes,
            allow_compute=allow_compute,
            req=req,
            block_size=block_size,
        )
        if actions is None:
            return None
        evicts = self.eviction_policy.plan(
            memories[self.local_memory], slots_needed(actions), set(block_hashes)
        )
        if evicts is None:
            return None
        return LookupResult(evicts=evicts, blocks=actions)

    def resolve_actions(
        self,
        memories: dict[str, Memory],
        block_hashes: list[str],
        *,
        allow_compute: bool = True,
        req: Request | None = None,
        block_size: int = 1,
    ) -> BlockActions | None:
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
        self._compute_res: ComputeResource | None = None
        self._transfer_links: dict[str, BandwidthResource] = {}
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
        compute_res: ComputeResource | None,
        transfer_links: dict[str, BandwidthResource] | None,
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
        )

    def _pull_time(self, link: BandwidthResource, src_key: str, block_hash: str) -> float:
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
