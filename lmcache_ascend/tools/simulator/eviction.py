"""Eviction policies: which resident blocks to remove when a tier is full."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal

from .memory import KVBlock, Memory

EvictionKind = Literal["lru", "fifo", "random", "lfu"]


def _evictable_candidates(memory: Memory, exclude: set[str]) -> list[KVBlock]:
    candidates: list[KVBlock] = []
    for block_hash, copies in memory.blocks.items():
        if block_hash in exclude:
            continue
        for block in copies:
            if memory.can_evict_block(block):
                candidates.append(block)
    return candidates


class EvictionPolicy(ABC):
    def score(self, block: KVBlock, *, now: float = 0.0) -> float:
        """Lower score = evict first."""
        del now
        return block.last_touch

    @abstractmethod
    def pick_victims(self, memory: Memory, count: int, exclude: set[str]) -> list[KVBlock]:
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

    def score(self, block: KVBlock, *, now: float = 0.0) -> float:
        del now
        return block.last_touch

    def pick_victims(self, memory: Memory, count: int, exclude: set[str]) -> list[KVBlock]:
        candidates = _evictable_candidates(memory, exclude)
        candidates.sort(key=lambda block: self.score(block))
        return candidates[:count]


class LFUEviction(EvictionPolicy):
    """Evict blocks with lowest access count."""

    def score(self, block: KVBlock, *, now: float = 0.0) -> float:
        del now
        return float(block.access_count)

    def pick_victims(self, memory: Memory, count: int, exclude: set[str]) -> list[KVBlock]:
        candidates = _evictable_candidates(memory, exclude)
        candidates.sort(key=lambda block: self.score(block))
        return candidates[:count]


class FIFOEviction(EvictionPolicy):
    """Evict resident, unheld blocks with the oldest ``insert_seq`` first."""

    def score(self, block: KVBlock, *, now: float = 0.0) -> float:
        del now
        return float(block.insert_seq)

    def pick_victims(self, memory: Memory, count: int, exclude: set[str]) -> list[KVBlock]:
        candidates = _evictable_candidates(memory, exclude)
        candidates.sort(key=lambda block: self.score(block))
        return candidates[:count]


class RandomEviction(EvictionPolicy):
    """Evict a uniform random subset of evictable blocks (seeded for reproducibility)."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def pick_victims(self, memory: Memory, count: int, exclude: set[str]) -> list[KVBlock]:
        candidates = _evictable_candidates(memory, exclude)
        if len(candidates) <= count:
            return candidates
        return self._rng.sample(candidates, count)


class CostPrefixEviction(EvictionPolicy):
    """Keep frequently touched blocks; deprioritize suffix-like newer inserts."""

    def score(self, block: KVBlock, *, now: float = 0.0) -> float:
        del now
        return block.access_count * 1_000_000.0 - float(block.insert_seq)

    def pick_victims(self, memory: Memory, count: int, exclude: set[str]) -> list[KVBlock]:
        candidates = _evictable_candidates(memory, exclude)
        candidates.sort(key=lambda block: self.score(block))
        return candidates[:count]


EvictionFactory = Callable[[int], EvictionPolicy]


def lru_eviction(_seed: int = 0) -> EvictionPolicy:
    return LRUEviction()


def fifo_eviction(_seed: int = 0) -> EvictionPolicy:
    return FIFOEviction()


def lfu_eviction(_seed: int = 0) -> EvictionPolicy:
    return LFUEviction()


def random_eviction(seed: int = 0) -> EvictionPolicy:
    return RandomEviction(seed=seed)


def cost_prefix_eviction(_seed: int = 0) -> EvictionPolicy:
    return CostPrefixEviction()


def eviction_factory_for_kind(kind: EvictionKind, *, seed: int = 0) -> EvictionFactory:
    if kind == "fifo":
        return fifo_eviction
    if kind == "lfu":
        return lfu_eviction
    if kind == "random":
        if seed:
            return lambda _rng_seed=0: RandomEviction(seed=seed)
        return random_eviction
    return lru_eviction


def make_eviction(kind: EvictionKind, *, seed: int = 0) -> EvictionPolicy:
    return eviction_factory_for_kind(kind, seed=seed)(seed)
