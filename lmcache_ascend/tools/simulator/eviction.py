"""Eviction policies: which resident blocks to remove when a tier is full."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Literal

from .memory import KVBlock, Memory

HbmEvictionKind = Literal["lru", "fifo", "random"]


def _evictable_candidates(hbm: Memory, exclude: set[str]) -> list[KVBlock]:
    candidates: list[KVBlock] = []
    for block_hash, copies in hbm.blocks.items():
        if block_hash in exclude:
            continue
        for block in copies:
            if hbm.can_evict_block(block):
                candidates.append(block)
    return candidates


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
        candidates = _evictable_candidates(hbm, exclude)
        candidates.sort(key=lambda block: block.last_touch)
        return candidates[:count]


class FIFOEviction(EvictionPolicy):
    """Evict resident, unheld blocks with the oldest ``insert_seq`` first."""

    def pick_victims(self, hbm: Memory, count: int, exclude: set[str]) -> list[KVBlock]:
        candidates = _evictable_candidates(hbm, exclude)
        candidates.sort(key=lambda block: block.insert_seq)
        return candidates[:count]


class RandomEviction(EvictionPolicy):
    """Evict a uniform random subset of evictable blocks (seeded for reproducibility)."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def pick_victims(self, hbm: Memory, count: int, exclude: set[str]) -> list[KVBlock]:
        candidates = _evictable_candidates(hbm, exclude)
        if len(candidates) <= count:
            return candidates
        return self._rng.sample(candidates, count)


def make_hbm_eviction(kind: HbmEvictionKind, *, seed: int = 0) -> EvictionPolicy:
    if kind == "fifo":
        return FIFOEviction()
    if kind == "random":
        return RandomEviction(seed=seed)
    return LRUEviction()
