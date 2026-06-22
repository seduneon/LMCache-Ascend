"""Shared tier slot acquisition and eviction (placement says *where*, this says *how*)."""

from __future__ import annotations

from .memory import BlockState, KVBlock, Memory
from .tier import Tier


class TierAllocator:
    """Acquire tier slots by evicting via the tier's ``EvictionPolicy`` when full."""

    @staticmethod
    def tier_covers(memory: Memory, storage_key: str) -> bool:
        return (
            memory.best_resident(storage_key) is not None
            or memory.inflight_incoming(storage_key) is not None
        )

    def ensure_slot(
        self,
        tier: Tier,
        storage_key: str,
        *,
        state: BlockState,
        exclude: set[str],
    ) -> KVBlock | None:
        """Return an existing or newly allocated block; ``None`` if the tier cannot fit."""
        memory = tier.memory
        if self.tier_covers(memory, storage_key):
            if state == BlockState.RESIDENT:
                return memory.best_resident(storage_key)
            return None

        while memory.free_size() <= 0:
            victims = tier.eviction.pick_victims(memory, 1, exclude)
            if not victims:
                return None
            memory.remove_block(victims[0])
            memory.tier_evictions += 1

        block = KVBlock(storage_key, state)
        memory.append(block)
        return block
