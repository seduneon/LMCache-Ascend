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
from .eviction import EvictionFactory, EvictionPolicy, LRUEviction, lru_eviction
from .memory import Memory
from .tier import Tier, TierGraph

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
    local: EvictionFactory | None = None,
    downstream: EvictionFactory | None = None,
    rng_seed: int = 0,
    overrides: dict[str, EvictionPolicy] | None = None,
) -> dict[str, EvictionPolicy]:
    """Assign eviction policy per tier key (local vs downstream factories)."""
    local_factory = local or lru_eviction
    downstream_factory = downstream or lru_eviction
    local_policy = local_factory(rng_seed)
    downstream_policy = downstream_factory(rng_seed)
    result: dict[str, EvictionPolicy] = {}
    for key in tier_keys:
        if overrides and key in overrides:
            result[key] = overrides[key]
        elif key.endswith(":hbm"):
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
        tiers[tier_key] = build_tier(
            spec,
            tokens_per_block=tokens_per_block,
            kv_bytes_per_token=kv_bpt,
            eviction=evictions.get(tier_key, LRUEviction()),
            slots=resolved_slots,
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
