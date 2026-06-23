"""Memory tier topology for PD / xPyD experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .capacity import (
    TierSpec,
    build_tier,
    default_tiers_for,
    kv_bytes_per_token as resolve_kv_bpt,
    resolve_tier_slots,
)
from simulator.policy.eviction import EVICTION, EvictionPolicy, LRUEviction
from simulator.policy.registry import PolicyContext
from simulator.core.memory import Memory
from .tier import Tier, TierGraph, TierRole

EngineRole = Literal["prefill", "decode"]

DEFAULT_PREFILL_IDS: tuple[str, ...] = ("npu-0",)
DEFAULT_DECODE_IDS: tuple[str, ...] = ("npu-1",)


def engine_ids_for_pd(
    *,
    num_prefill: int = 1,
    num_decode: int = 1,
    prefix: str = "npu",
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Assign contiguous engine ids: prefill npu-0.., decode npu-P.."""
    if num_prefill < 1 or num_decode < 1:
        raise ValueError("num_prefill and num_decode must be >= 1")
    prefill_ids = tuple(f"{prefix}-{i}" for i in range(num_prefill))
    decode_ids = tuple(
        f"{prefix}-{num_prefill + i}" for i in range(num_decode)
    )
    return prefill_ids, decode_ids


def local_tier_key(engine_id: str) -> str:
    return f"{engine_id}:hbm"


def engine_role_map(
    prefill_ids: tuple[str, ...],
    decode_ids: tuple[str, ...],
) -> dict[str, EngineRole]:
    return {
        **{eid: "prefill" for eid in prefill_ids},
        **{eid: "decode" for eid in decode_ids},
    }


def tier_keys_for(
    prefill_ids: tuple[str, ...],
    decode_ids: tuple[str, ...],
    kind: str,
) -> tuple[str, ...]:
    """Tier keys for ``kind`` over arbitrary prefill/decode engine sets."""
    if not prefill_ids or not decode_ids:
        raise ValueError("prefill_ids and decode_ids must be non-empty")
    hbm_keys = tuple(local_tier_key(eid) for eid in prefill_ids + decode_ids)
    if kind == "hbm_only":
        return hbm_keys
    primary = prefill_ids[0]
    if kind == "hbm_dram":
        return hbm_keys + (f"{primary}:dram",)
    if kind == "hbm_dram_ssd":
        return hbm_keys + (f"{primary}:dram", f"{primary}:ssd")
    raise ValueError(f"unknown topology {kind!r}")


TOPOLOGY_TIERS: dict[str, tuple[str, ...]] = {
    "hbm_only": tier_keys_for(DEFAULT_PREFILL_IDS, DEFAULT_DECODE_IDS, "hbm_only"),
    "hbm_dram": tier_keys_for(DEFAULT_PREFILL_IDS, DEFAULT_DECODE_IDS, "hbm_dram"),
    "hbm_dram_ssd": tier_keys_for(
        DEFAULT_PREFILL_IDS, DEFAULT_DECODE_IDS, "hbm_dram_ssd"
    ),
}


@dataclass(frozen=True)
class Topology:
    """Named memory tiers and PD local/pull wiring."""

    graph: TierGraph
    prefill_ids: tuple[str, ...] = DEFAULT_PREFILL_IDS
    decode_ids: tuple[str, ...] = DEFAULT_DECODE_IDS
    local_tier: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.local_tier:
            object.__setattr__(
                self,
                "local_tier",
                {eid: local_tier_key(eid) for eid in self.prefill_ids + self.decode_ids},
            )

    @property
    def memories(self) -> dict[str, Memory]:
        return self.graph.memories

    @property
    def engine_role(self) -> dict[str, EngineRole]:
        return engine_role_map(self.prefill_ids, self.decode_ids)

    @property
    def prefill_local_tier(self) -> str:
        return self.local_tier[self.prefill_ids[0]]

    @property
    def decode_local_tier(self) -> str:
        return self.local_tier[self.decode_ids[0]]

    @property
    def prefill_engine_id(self) -> str:
        return self.prefill_ids[0]

    @property
    def decode_engine_id(self) -> str:
        return self.decode_ids[0]

    def prefill_hbm_tiers(self) -> tuple[str, ...]:
        return tuple(self.local_tier[eid] for eid in self.prefill_ids)

    def decode_pull_sources(self, sources: tuple[str, ...]) -> tuple[str, ...]:
        """Resolve preset pull sources; expand prefill HBM refs to all prefill HBMs."""
        ordered: list[str] = []
        seen: set[str] = set()
        prefill_hbm = set(self.prefill_hbm_tiers())
        for source in sources:
            if source in prefill_hbm or (
                source.endswith(":hbm")
                and source.split(":")[0] in self.prefill_ids
            ):
                for tier in self.prefill_hbm_tiers():
                    if tier in self.graph.tiers and tier not in seen:
                        ordered.append(tier)
                        seen.add(tier)
            elif source in self.graph.tiers and source not in seen:
                ordered.append(source)
                seen.add(source)
        return tuple(ordered)


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
        prefill_ids: tuple[str, ...] = DEFAULT_PREFILL_IDS,
        decode_ids: tuple[str, ...] = DEFAULT_DECODE_IDS,
    ) -> SimResources:
        kv_bpt = (
            kv_bytes_per_token
            if kv_bytes_per_token is not None
            else resolve_kv_bpt("llama3-8b")
        )
        return cls.from_tiers(
            default_tiers_for(prefill_ids, decode_ids),
            tokens_per_block=tokens_per_block,
            kv_bytes_per_token=kv_bpt,
        )

    def size(self, tier_key: str) -> int:
        return self.slots[tier_key]

    def hbm_size(self, engine_id: str = "npu-0") -> int:
        return self.slots[local_tier_key(engine_id)]


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
    prefill_ids: tuple[str, ...] = DEFAULT_PREFILL_IDS,
    decode_ids: tuple[str, ...] = DEFAULT_DECODE_IDS,
    resources: SimResources | None = None,
    eviction_map: dict[str, EvictionPolicy] | None = None,
    tokens_per_block: int = 512,
    kv_bytes_per_token: float | None = None,
) -> TierGraph:
    tier_keys = tier_keys_for(prefill_ids, decode_ids, kind)

    cfg = resources or SimResources.default(
        tokens_per_block=tokens_per_block,
        kv_bytes_per_token=kv_bytes_per_token,
        prefill_ids=prefill_ids,
        decode_ids=decode_ids,
    )
    kv_bpt = (
        kv_bytes_per_token
        if kv_bytes_per_token is not None
        else resolve_kv_bpt("llama3-8b")
    )
    evictions = eviction_map or {key: LRUEviction() for key in tier_keys}

    tiers: dict[str, Tier] = {}
    specs = cfg.tier_specs or default_tiers_for(prefill_ids, decode_ids)
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
    prefill_ids: tuple[str, ...] = DEFAULT_PREFILL_IDS,
    decode_ids: tuple[str, ...] = DEFAULT_DECODE_IDS,
    resources: SimResources | None = None,
    eviction_map: dict[str, EvictionPolicy] | None = None,
    tokens_per_block: int = 512,
    kv_bytes_per_token: float | None = None,
) -> Topology:
    graph = build_tier_graph(
        kind,
        prefill_ids=prefill_ids,
        decode_ids=decode_ids,
        resources=resources,
        eviction_map=eviction_map,
        tokens_per_block=tokens_per_block,
        kv_bytes_per_token=kv_bytes_per_token,
    )
    return Topology(
        graph=graph,
        prefill_ids=prefill_ids,
        decode_ids=decode_ids,
    )
