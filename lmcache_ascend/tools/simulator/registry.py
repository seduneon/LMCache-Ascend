"""Policy registry: name -> factory for experiment sweeps."""

from __future__ import annotations

from typing import Callable

from .engine_config import PlacementSpec, build_placement_spec
from .eviction import EvictionKind, EvictionPolicy, make_eviction
from .read_path import ReadPathKind, ReadPathSpec
from .retention import (
    ConsumeOnPull,
    GlobalCopyCap,
    RetentionPolicy,
    SingleCopyPerTier,
    UnboundedRetention,
)
from .tier import TierGraph

ReadPathFactory = Callable[..., ReadPathSpec]
EvictionFactory = Callable[..., EvictionPolicy]
RetentionFactory = Callable[[TierGraph, PlacementSpec], RetentionPolicy]
PlacementFactory = Callable[..., PlacementSpec]

READ_PATH_REGISTRY: dict[str, ReadPathFactory] = {
    "ordered_pull": lambda **_: ReadPathSpec(kind="ordered_pull"),
    "compute_only": lambda **_: ReadPathSpec(kind="compute_only"),
    "min_cost": lambda **_: ReadPathSpec(kind="min_cost"),
    "min_cost_with_wait": lambda **_: ReadPathSpec(kind="min_cost_with_wait"),
    "threshold": lambda threshold_ratio=1.0, **_: ReadPathSpec(
        kind="threshold", threshold_ratio=float(threshold_ratio)
    ),
}

EVICTION_REGISTRY: dict[str, EvictionFactory] = {
    "lru": lambda seed=0, **_: make_eviction("lru", seed=seed),
    "fifo": lambda seed=0, **_: make_eviction("fifo", seed=seed),
    "lfu": lambda seed=0, **_: make_eviction("lfu", seed=seed),
    "random": lambda seed=0, **_: make_eviction("random", seed=seed),
}

RETENTION_REGISTRY: dict[str, RetentionFactory] = {
    "unbounded": lambda graph, _spec: UnboundedRetention(graph),
    "single_copy": lambda graph, _spec: SingleCopyPerTier(graph),
    "consume_on_pull": lambda graph, _spec: ConsumeOnPull(graph),
    "global_cap": lambda graph, spec: GlobalCopyCap(
        spec.global_cap_max,
        list(spec.global_cap_tiers),
        per_tier_cap=spec.global_cap_per_tier,
        graph=graph,
    ),
}


def make_read_path(name: str, **kwargs) -> ReadPathSpec:
    if name not in READ_PATH_REGISTRY:
        raise KeyError(f"unknown read_path: {name!r}")
    return READ_PATH_REGISTRY[name](**kwargs)


def make_retention(name: str, graph: TierGraph, spec: PlacementSpec) -> RetentionPolicy:
    if name not in RETENTION_REGISTRY:
        raise KeyError(f"unknown retention: {name!r}")
    return RETENTION_REGISTRY[name](graph, spec)


def make_placement(
  name: str = "default",
  *,
  mirror_tiers: tuple[str, ...] = (),
  async_write_tiers: frozenset[str] = frozenset(),
  **kwargs,
) -> PlacementSpec:
    del name
    return build_placement_spec(
        mirror_tiers=mirror_tiers,
        async_write_tiers=async_write_tiers,
        **kwargs,
    )


def list_read_paths() -> list[str]:
    return sorted(READ_PATH_REGISTRY)


def list_evictions() -> list[str]:
    return sorted(EVICTION_REGISTRY)
