"""Per-engine configuration: local tier, pull order, placement, lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .placement import PlacementEdge
from .read_path import ReadPathSpec


@dataclass(frozen=True)
class LifecycleSpec:
    hold_kv_on_complete: bool = False
    retain_prefix_cache: bool = False
    store_on_complete: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlacementSpec:
    edges: tuple[PlacementEdge, ...] = ()
    mirror_on_forward: bool = True
    retention_name: str = "unbounded"
    retention_params: dict = field(default_factory=dict)


def placement_edges(
    tier_keys: tuple[str, ...],
    *,
    async_write_tiers: frozenset[str] = frozenset(),
) -> tuple[PlacementEdge, ...]:
    """Build forward/complete/evict edges for each downstream tier."""
    edges: list[PlacementEdge] = []
    for dst in tier_keys:
        delivery: Literal["sync", "async"] = (
            "async" if dst in async_write_tiers else "sync"
        )
        for trigger in ("forward", "complete", "evict"):
            edges.append(PlacementEdge(dst, trigger, delivery))  # type: ignore[arg-type]
    return tuple(edges)


def build_placement_spec(
    *,
    mirror_tiers: tuple[str, ...] = (),
    async_write_tiers: frozenset[str] = frozenset(),
    mirror_on_forward: bool = True,
    retention_name: str = "unbounded",
    retention_params: dict | None = None,
) -> PlacementSpec:
    tier_keys = tuple(dict.fromkeys((*mirror_tiers, *async_write_tiers)))
    return PlacementSpec(
        edges=placement_edges(tier_keys, async_write_tiers=async_write_tiers),
        mirror_on_forward=mirror_on_forward,
        retention_name=retention_name,
        retention_params=dict(retention_params or {}),
    )


@dataclass(frozen=True)
class EngineConfig:
    engine_id: str
    local_tier: str
    pull_sources: tuple[str, ...] = ()
    pull_mode: Literal["compute_only", "ordered_pull"] = "compute_only"
    read_path: ReadPathSpec | None = None
    placement: PlacementSpec = field(default_factory=PlacementSpec)
    lifecycle: LifecycleSpec = field(default_factory=LifecycleSpec)

    def resolved_read_path(self) -> ReadPathSpec:
        if self.read_path is not None:
            return self.read_path
        from .read_path import read_path_from_pull_mode

        return read_path_from_pull_mode(self.pull_mode)
