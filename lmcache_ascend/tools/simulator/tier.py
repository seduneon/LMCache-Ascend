"""Resolved memory tiers: storage + eviction policy per tier key."""

from __future__ import annotations

from dataclasses import dataclass

from .eviction import EvictionPolicy, LRUEviction
from .memory import Memory


@dataclass
class Tier:
    """One tier: dumb ``Memory`` plus the eviction policy used when it is full."""

    key: str
    memory: Memory
    eviction: EvictionPolicy


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
            return LRUEviction()
        return tier.eviction


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
