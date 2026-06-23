"""Execute-time policy: tier mirrors, spills, retention caps."""

from __future__ import annotations

from dataclasses import dataclass

from .engine_config import PlacementSpec
from .memory import KVBlock, Memory
from .placement import PLACEMENT, PlacementPolicy
from .plan import StoreOp
from .policy_registry import PolicyContext
from .request import Request
from .retention import RETENTION, RetentionPolicy
from .tier import TierGraph


@dataclass(frozen=True)
class EffectConfig:
    local_memory: str
    graph: TierGraph
    placement: PlacementSpec


class EffectPolicy:
    """Placement + retention at execute time; store planning at schedule time."""

    def __init__(self, config: EffectConfig):
        self.config = config
        placement_spec = config.placement
        self._retention = RETENTION.create(
            placement_spec.retention_name,
            PolicyContext(
                graph=config.graph,
                params=placement_spec.retention_params,
            ),
        )
        self._placement = PLACEMENT.create(
            placement_spec.placement_name,
            PolicyContext(
                graph=config.graph,
                params={
                    "edges": placement_spec.edges,
                    "mirror_on_forward": placement_spec.mirror_on_forward,
                    "retention": self._retention,
                },
            ),
        )

    def on_local_resident(
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

    def on_local_evict(
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

    def store_prefix_on_complete(
        self,
        memories: dict[str, Memory],
        *,
        req: Request,
        now: float,
        tier_keys: tuple[str, ...],
    ) -> bool:
        return self._placement.store_prefix_on_complete(
            memories,
            local_memory=self.config.local_memory,
            req=req,
            now=now,
            tier_keys=tier_keys,
        )
