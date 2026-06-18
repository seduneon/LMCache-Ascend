"""Placement policies: where KV copies live when blocks become HBM-resident."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .content_key import ContentKey
from .chunk_hash import chunk_key_for_hbm_block
from .eviction import EvictionPolicy, LRUEviction
from .memory import BlockState, KVBlock, Memory
from .plan import StoreOp
from .request import Request
from .retention import RetentionPolicy, UnboundedRetention
from .tier_allocator import TierAllocator


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

    def mirror_tier_keys(self) -> tuple[str, ...]:
        """Downstream tiers that receive sync mirrors when HBM blocks become resident."""
        return ()


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

    def mirror_tier_keys(self) -> tuple[str, ...]:
        return (self.dram_memory,)


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
            assert not block.hash.startswith("blk:"), (
                "decode-generated blocks (blk:req:N) do not spill to downstream tiers"
            )
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
                    content=ContentKey.for_hbm_block(block_hash),
                    storage_key=chunk_key,
                    hbm_block_hash=block_hash,
                )
            )
        return ops

    def mirror_tier_keys(self) -> tuple[str, ...]:
        return tuple(
            tier_key
            for tier_key in self.tier_keys
            if tier_key not in self._paid_write_tiers
        )
