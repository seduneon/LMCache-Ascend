"""Memory tier topology for PD experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from .capacity import (
    TierSpec,
    build_tier,
    default_tiers,
    kv_bytes_per_token as resolve_kv_bpt,
    resolve_tier_slots,
)
from .eviction import EVICTION, EvictionPolicy, LRUEviction
from .policy_registry import PolicyContext
from .memory import Memory
from .tier import Tier, TierGraph, TierRole

TOPOLOGY_TIERS: dict[str, tuple[str, ...]] = {
    "hbm_only": ("npu-0:hbm", "npu-1:hbm"),
    "hbm_dram": ("npu-0:hbm", "npu-1:hbm", "npu-0:dram"),
    "hbm_dram_ssd": ("npu-0:hbm", "npu-1:hbm", "npu-0:dram", "npu-0:ssd"),
}


@dataclass(frozen=True)
class Topology:
    """Named memory tiers and PD local/pull wiring."""

    graph: TierGraph
    prefill_local_tier: str = "npu-0:hbm"
    decode_local_tier: str = "npu-1:hbm"

    @property
    def memories(self) -> dict[str, Memory]:
        return self.graph.memories

    @property
    def prefill_engine_id(self) -> str:
        return self.prefill_local_tier.split(":")[0]

    @property
    def decode_engine_id(self) -> str:
        return self.decode_local_tier.split(":")[0]

    def decode_pull_sources(self, sources: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(s for s in sources if s in self.graph.tiers)


@dataclass(frozen=True)
class SimResources:
    """Resolved tier slot counts."""

    slots: dict[str, int] = field(default_factory=dict)
    chunk_blocks: dict[str, int] = field(default_factory=dict)
    tier_specs: tuple[TierSpec, ...] = ()

    @classmethod
    def from_tiers(
        cls,
        tiers: tuple[TierSpec, ...],
        *,
        tokens_per_block: int,
        kv_bytes_per_token: float,
    ) -> SimResources:
        slots: dict[str, int] = {}
        chunks: dict[str, int] = {}
        for tier in tiers:
            slots[tier.tier_key] = resolve_tier_slots(
                tier,
                tokens_per_block=tokens_per_block,
                kv_bytes_per_token=kv_bytes_per_token,
            )
            chunks[tier.tier_key] = tier.chunk_blocks
        return cls(slots=slots, chunk_blocks=chunks, tier_specs=tiers)

    @classmethod
    def default(
        cls,
        *,
        tokens_per_block: int = 512,
        kv_bytes_per_token: float | None = None,
    ) -> SimResources:
        kv_bpt = (
            kv_bytes_per_token
            if kv_bytes_per_token is not None
            else resolve_kv_bpt("llama3-8b")
        )
        return cls.from_tiers(
            default_tiers(),
            tokens_per_block=tokens_per_block,
            kv_bytes_per_token=kv_bpt,
        )

    def size(self, tier_key: str) -> int:
        return self.slots[tier_key]

    @property
    def hbm_size(self) -> int:
        return self.slots["npu-0:hbm"]


def build_eviction_map(
    tier_keys: tuple[str, ...],
    *,
    local_tiers: frozenset[str] | None = None,
    local: str = "lru",
    downstream: str = "lru",
    rng_seed: int = 0,
    overrides: dict[str, EvictionPolicy] | None = None,
) -> dict[str, EvictionPolicy]:
    """Assign eviction policy per tier key (local vs downstream names)."""
    local_set = local_tiers or frozenset(
        key for key in tier_keys if key.endswith(":hbm")
    )
    ctx = PolicyContext(seed=rng_seed)
    local_policy = EVICTION.create(local, ctx)
    downstream_policy = EVICTION.create(downstream, ctx)
    result: dict[str, EvictionPolicy] = {}
    for key in tier_keys:
        if overrides and key in overrides:
            result[key] = overrides[key]
        elif key in local_set:
            result[key] = local_policy
        else:
            result[key] = downstream_policy
    return result


def build_tier_graph(
    kind: str,
    *,
    resources: SimResources | None = None,
    eviction_map: dict[str, EvictionPolicy] | None = None,
    tokens_per_block: int = 512,
    kv_bytes_per_token: float | None = None,
) -> TierGraph:
    tier_keys = TOPOLOGY_TIERS.get(kind)
    if tier_keys is None:
        raise ValueError(f"unknown topology {kind!r}")

    cfg = resources or SimResources.default(tokens_per_block=tokens_per_block)
    kv_bpt = (
        kv_bytes_per_token
        if kv_bytes_per_token is not None
        else resolve_kv_bpt("llama3-8b")
    )
    evictions = eviction_map or {key: LRUEviction() for key in tier_keys}

    tiers: dict[str, Tier] = {}
    specs = cfg.tier_specs or default_tiers()
    spec_by_key = {spec.tier_key: spec for spec in specs}
    for tier_key in tier_keys:
        spec = spec_by_key.get(tier_key)
        if spec is None:
            raise ValueError(f"no TierSpec for topology tier {tier_key!r}")
        resolved_slots = cfg.slots.get(tier_key)
        role: TierRole = "local" if tier_key.endswith(":hbm") else "downstream"
        tiers[tier_key] = build_tier(
            spec,
            tokens_per_block=tokens_per_block,
            kv_bytes_per_token=kv_bpt,
            eviction=evictions.get(tier_key, LRUEviction()),
            slots=resolved_slots,
            role=role,
        )
    return TierGraph(tiers=tiers)


def build_topology(
    kind: str,
    *,
    resources: SimResources | None = None,
    eviction_map: dict[str, EvictionPolicy] | None = None,
    tokens_per_block: int = 512,
    kv_bytes_per_token: float | None = None,
) -> Topology:
    graph = build_tier_graph(
        kind,
        resources=resources,
        eviction_map=eviction_map,
        tokens_per_block=tokens_per_block,
        kv_bytes_per_token=kv_bytes_per_token,
    )
    return Topology(graph=graph)
