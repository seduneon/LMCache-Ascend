"""Execute-time policy: tier mirrors, spills, retention caps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .eviction import EvictionPolicy
from .memory import KVBlock, Memory
from .placement import HBMOnly, PlacementPolicy, TieredPlacement
from .plan import StoreOp
from .request import Request
from .retention import (
    ConsumeOnPull,
    GlobalCopyCap,
    RetentionPolicy,
    SingleCopyPerTier,
    UnboundedRetention,
)


@dataclass(frozen=True)
class EffectConfig:
    local_memory: str
    mirror_tiers: tuple[str, ...] = ()
    async_write_tiers: frozenset[str] = frozenset()
    tier_eviction: dict[str, EvictionPolicy] = field(default_factory=dict)
    retention: Literal[
        "unbounded", "single_copy", "consume_on_pull", "global_cap"
    ] = "unbounded"
    global_cap_max: int = 0
    global_cap_tiers: tuple[str, ...] = ()
    global_cap_per_tier: int | None = 1


def _build_retention(config: EffectConfig) -> RetentionPolicy:
    if config.retention == "single_copy":
        return SingleCopyPerTier()
    if config.retention == "consume_on_pull":
        return ConsumeOnPull()
    if config.retention == "global_cap":
        return GlobalCopyCap(
            config.global_cap_max,
            list(config.global_cap_tiers),
            per_tier_cap=config.global_cap_per_tier,
        )
    return UnboundedRetention()


def _build_placement(config: EffectConfig, retention: RetentionPolicy) -> PlacementPolicy:
    tier_keys = list(dict.fromkeys((*config.mirror_tiers, *config.async_write_tiers)))
    if not tier_keys:
        return HBMOnly()
    placement = TieredPlacement(
        tier_keys,
        tier_eviction=config.tier_eviction or None,
        paid_write_tiers=config.async_write_tiers,
    )
    placement.bind_retention(retention)
    return placement


class EffectPolicy:
    """Placement + retention at execute time; store planning at schedule time."""

    def __init__(self, config: EffectConfig):
        self.config = config
        self._retention = _build_retention(config)
        self._placement = _build_placement(config, self._retention)

    @classmethod
    def none(cls, local_memory: str) -> EffectPolicy:
        return cls(EffectConfig(local_memory=local_memory))

    def on_hbm_resident(
        self,
        memories: dict[str, Memory],
        block: KVBlock,
        req: Request,
        now: float,
    ) -> None:
        self._placement.place_copy(
            memories,
            local_memory=self.config.local_memory,
            block=block,
            req=req,
            now=now,
        )
        self._retention.on_block_resident(
            memories,
            tier_key=self.config.local_memory,
            block=block,
            now=now,
            req=req,
        )

    def on_tier_resident(
        self,
        memories: dict[str, Memory],
        tier_key: str,
        block: KVBlock,
        req: Request,
        now: float,
    ) -> None:
        self._retention.on_block_resident(
            memories,
            tier_key=tier_key,
            block=block,
            now=now,
            req=req,
        )

    def on_hbm_evict(
        self,
        memories: dict[str, Memory],
        victim: KVBlock,
        req: Request,
        now: float,
    ) -> None:
        self._placement.spill_on_evict(
            memories,
            local_memory=self.config.local_memory,
            block=victim,
            now=now,
            req=req,
        )

    def after_pull(
        self,
        memories: dict[str, Memory],
        src_key: str,
        block_hash: str,
        req: Request,
    ) -> None:
        self._retention.after_pull(
            memories,
            src_key=src_key,
            dst_key=self.config.local_memory,
            block_hash=block_hash,
            now=0.0,
            req=req,
        )

    def plan_async_stores(
        self,
        memories: dict[str, Memory],
        block_hash: str,
        req: Request,
    ) -> list[StoreOp]:
        return self._placement.plan_async_stores(
            memories,
            local_memory=self.config.local_memory,
            block_hash=block_hash,
            req=req,
        )

    def plan_spill_stores(
        self,
        memories: dict[str, Memory],
        block_hash: str,
        req: Request,
    ) -> list[StoreOp]:
        return self._placement.plan_spill_stores(
            memories,
            local_memory=self.config.local_memory,
            block_hash=block_hash,
            req=req,
        )
