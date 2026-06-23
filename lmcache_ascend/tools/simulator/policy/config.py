"""Per-engine configuration: local tier, pull order, placement, lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .placement import PlacementEdge
from .read_path import READ_PATH, read_path_from_pull_mode
from .registry import PolicyContext


@dataclass(frozen=True)
class LifecycleSpec:
    hold_kv_on_complete: bool = False
    retain_prefix_cache: bool = False
    store_on_complete: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlacementSpec:
    edges: tuple[PlacementEdge, ...] = ()
    mirror_on_forward: bool = True
    placement_name: str = "hbm_only"
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
        placement_name="tiered" if tier_keys else "hbm_only",
        retention_name=retention_name,
        retention_params=dict(retention_params or {}),
    )


@dataclass(frozen=True)
class EngineConfig:
    engine_id: str
    local_tier: str
    pull_sources: tuple[str, ...] = ()
    pull_mode: Literal["compute_only", "ordered_pull"] = "compute_only"
    read_path_name: str | None = None
    read_path_params: dict = field(default_factory=dict)
    placement: PlacementSpec = field(default_factory=PlacementSpec)
    lifecycle: LifecycleSpec = field(default_factory=LifecycleSpec)

    def resolved_read_path_name(self) -> str:
        if self.read_path_name is not None:
            return self.read_path_name
        return read_path_from_pull_mode(self.pull_mode)

    def build_read_path_strategy(self):
        return READ_PATH.create(
            self.resolved_read_path_name(),
            PolicyContext(params=self.read_path_params),
        )
