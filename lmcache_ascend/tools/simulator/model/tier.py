"""Resolved memory tiers: storage + eviction policy per tier key."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from simulator.policy.eviction import EvictionPolicy, LRUEviction
from simulator.core.memory import BlockState, KVBlock, Memory

TierRole = Literal["local", "downstream"]


TierEvictObserver = Callable[[float, KVBlock], None]


@dataclass
class Tier:
    """One tier: dumb ``Memory`` plus the eviction policy used when it is full."""

    key: str
    memory: Memory
    eviction: EvictionPolicy
    role: TierRole = "downstream"
    on_tier_evict: TierEvictObserver | None = None


@dataclass(frozen=True)
class TierGraph:
    """All tiers in a simulation run."""

    tiers: dict[str, Tier]

    @property
    def memories(self) -> dict[str, Memory]:
        return {key: tier.memory for key, tier in self.tiers.items()}

    def get(self, key: str) -> Tier:
        return self.tiers[key]

    def eviction_for(self, key: str) -> EvictionPolicy:
        tier = self.tiers.get(key)
        if tier is None:
            raise KeyError(f"no tier {key!r} in graph")
        return tier.eviction

    def tier_roles(self) -> dict[str, TierRole]:
        return {key: tier.role for key, tier in self.tiers.items()}

    def local_tier_keys(self) -> frozenset[str]:
        return frozenset(key for key, tier in self.tiers.items() if tier.role == "local")


def graph_from_memories(
    memories: dict[str, Memory],
    *,
    eviction: EvictionPolicy | None = None,
) -> TierGraph:
    """Build a ``TierGraph`` from a plain memories dict (tests and migration)."""
    policy = eviction or LRUEviction()
    return TierGraph(
        tiers={key: Tier(key, memory, policy) for key, memory in memories.items()}
    )


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
        now: float = 0.0,
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
            victim = victims[0]
            memory.remove_block(victim)
            memory.tier_evictions += 1
            if tier.on_tier_evict is not None:
                tier.on_tier_evict(now, victim)

        block = KVBlock(storage_key, state)
        memory.append(block)
        return block
