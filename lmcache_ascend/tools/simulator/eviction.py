"""Eviction policies: which resident blocks to remove when a tier is full."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .memory import KVBlock, Memory


class EvictionPolicy(ABC):
    @abstractmethod
    def pick_victims(self, hbm: Memory, count: int, exclude: set[str]) -> list[KVBlock]:
        pass

    def plan(
        self, local: Memory, slots_needed: int, exclude: set[str]
    ) -> list[KVBlock] | None:
        deficit = slots_needed - local.free_size()
        if deficit <= 0:
            return []
        evicts = self.pick_victims(local, deficit, exclude)
        if len(evicts) < deficit:
            return None
        return evicts


class LRUEviction(EvictionPolicy):
    """Evict resident, unheld blocks with the oldest ``last_touch`` first."""

    def pick_victims(self, hbm: Memory, count: int, exclude: set[str]) -> list[KVBlock]:
        candidates: list[KVBlock] = []
        for block_hash, copies in hbm.blocks.items():
            if block_hash in exclude:
                continue
            for block in copies:
                if hbm.can_evict_block(block):
                    candidates.append(block)
        candidates.sort(key=lambda block: block.last_touch)
        return candidates[:count]
