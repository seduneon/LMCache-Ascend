"""Policy registry: name -> factory for experiment sweeps."""

from __future__ import annotations

from typing import Callable

from .engine_config import PlacementSpec, build_placement_spec
from .eviction import (
    EvictionFactory,
    fifo_eviction,
    lfu_eviction,
    lru_eviction,
    random_eviction,
)
from .read_path import ReadPathSpec
from .retention import (
    RetentionFactory,
    consume_on_pull_retention,
    global_copy_cap_retention,
    single_copy_retention,
    unbounded_retention,
)

ReadPathFactory = Callable[..., ReadPathSpec]
EvictionFactoryBuilder = Callable[..., EvictionFactory]
RetentionFactoryBuilder = Callable[..., RetentionFactory]
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

EVICTION_FACTORY_REGISTRY: dict[str, EvictionFactoryBuilder] = {
    "lru": lambda **_: lru_eviction,
    "fifo": lambda **_: fifo_eviction,
    "lfu": lambda **_: lfu_eviction,
    "random": lambda seed=0, **_: (
        (lambda _rng_seed=0: random_eviction(seed)) if seed else random_eviction
    ),
}

RETENTION_FACTORY_REGISTRY: dict[str, RetentionFactoryBuilder] = {
    "unbounded": lambda **_: unbounded_retention,
    "single_copy": lambda **_: single_copy_retention,
    "consume_on_pull": lambda **_: consume_on_pull_retention,
    "global_cap": lambda max_total=2, tier_keys=(), per_tier_cap=1, **_: global_copy_cap_retention(
        max_total=int(max_total),
        tier_keys=tuple(tier_keys),
        per_tier_cap=per_tier_cap,
    ),
}


def make_read_path(name: str, **kwargs) -> ReadPathSpec:
    if name not in READ_PATH_REGISTRY:
        raise KeyError(f"unknown read_path: {name!r}")
    return READ_PATH_REGISTRY[name](**kwargs)


def make_eviction_factory(name: str, **kwargs) -> EvictionFactory:
    if name not in EVICTION_FACTORY_REGISTRY:
        raise KeyError(f"unknown eviction: {name!r}")
    return EVICTION_FACTORY_REGISTRY[name](**kwargs)


def make_retention_factory(name: str, **kwargs) -> RetentionFactory:
    if name not in RETENTION_FACTORY_REGISTRY:
        raise KeyError(f"unknown retention: {name!r}")
    return RETENTION_FACTORY_REGISTRY[name](**kwargs)


def make_placement(
    name: str = "default",
    *,
    mirror_tiers: tuple[str, ...] = (),
    async_write_tiers: frozenset[str] = frozenset(),
    retention: RetentionFactory | None = None,
    **kwargs,
) -> PlacementSpec:
    del name, kwargs
    return build_placement_spec(
        mirror_tiers=mirror_tiers,
        async_write_tiers=async_write_tiers,
        retention=retention,
    )


def list_read_paths() -> list[str]:
    return sorted(READ_PATH_REGISTRY)


def list_evictions() -> list[str]:
    return sorted(EVICTION_FACTORY_REGISTRY)


def list_retentions() -> list[str]:
    return sorted(RETENTION_FACTORY_REGISTRY)
