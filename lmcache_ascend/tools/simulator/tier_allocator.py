"""Shared tier slot acquisition and eviction (placement says *where*, this says *how*)."""

from __future__ import annotations

from .memory import BlockState, KVBlock, Memory
from .eviction import EvictionPolicy, LRUEviction


class TierAllocator:
    """Acquire tier slots by evicting via ``EvictionPolicy`` when a tier is full."""

    def __init__(self, default_eviction: EvictionPolicy | None = None):
        self._default_eviction = default_eviction or LRUEviction()

    def eviction_for(
        self,
        tier_key: str,
        tier_eviction: dict[str, EvictionPolicy] | None = None,
    ) -> EvictionPolicy:
        if tier_eviction and tier_key in tier_eviction:
            return tier_eviction[tier_key]
        return self._default_eviction

    @staticmethod
    def tier_covers(memory: Memory, storage_key: str) -> bool:
        return (
            memory.best_resident(storage_key) is not None
            or memory.inflight_incoming(storage_key) is not None
        )

    def ensure_slot(
        self,
        memory: Memory,
        storage_key: str,
        *,
        state: BlockState,
        exclude: set[str],
        eviction: EvictionPolicy | None = None,
    ) -> KVBlock | None:
        """Return an existing or newly allocated block; ``None`` if the tier cannot fit."""
        if self.tier_covers(memory, storage_key):
            if state == BlockState.RESIDENT:
                return memory.best_resident(storage_key)
            return None

        ev = eviction or self._default_eviction
        while memory.free_size() <= 0:
            victims = ev.pick_victims(memory, 1, exclude)
            if not victims:
                return None
            memory.remove_block(victims[0])

        block = KVBlock(storage_key, state)
        memory.append(block)
        return block
