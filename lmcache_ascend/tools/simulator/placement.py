"""Placement policies: where KV copies live when blocks become local-tier resident."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from .kv_content import ContentKey, storage_key
from .memory import BlockState, KVBlock, Memory
from .plan import StoreOp
from .request import Request
from .policy_registry import PolicyContext, Registry
from .retention import RetentionPolicy
from .tier import TierAllocator, TierGraph

PLACEMENT = Registry["PlacementPolicy"]("placement")


@dataclass(frozen=True)
class PlacementEdge:
    """Directed copy from local tier to a downstream tier."""

    dst_tier: str
    trigger: Literal["forward", "complete", "evict"]
    delivery: Literal["sync", "async"]


def ensure_downstream_copy(
    graph: TierGraph,
    allocator: TierAllocator,
    *,
    dst: str,
    block_hash: str,
    req: Request,
    now: float,
    delivery: Literal["sync", "async"],
    exclude: set[str],
    retention: RetentionPolicy,
) -> tuple[bool, list[StoreOp]]:
    """Ensure ``block_hash`` has a downstream copy; async returns ``StoreOp`` list."""
    tier = graph.get(dst)
    memory = tier.memory
    chunk_key = storage_key(req, block_hash, memory.chunk_blocks)
    if allocator.tier_covers(memory, chunk_key):
        if delivery == "sync":
            resident = memory.best_resident(chunk_key)
            return resident is not None, []
        if memory.best_resident(chunk_key) is not None:
            return True, []
        if memory.inflight_incoming(chunk_key) is not None:
            return True, []

    if delivery == "sync":
        copy = allocator.ensure_slot(
            tier,
            chunk_key,
            state=BlockState.RESIDENT,
            exclude=exclude | {chunk_key},
            now=now,
        )
        if copy is None:
            return False, []
        memory.touch(copy, now)
        retention.on_block_resident(
            graph.memories, tier_key=dst, block=copy, now=now
        )
        return True, []

    if memory.best_resident(chunk_key) is not None:
        return True, []
    if memory.inflight_incoming(chunk_key) is not None:
        return True, []
    copy = allocator.ensure_slot(
        tier,
        chunk_key,
        state=BlockState.RESERVED,
        exclude=exclude | {chunk_key},
        now=now,
    )
    if copy is None:
        return False, []
    ops = [
        StoreOp(
            tier_key=dst,
            content=ContentKey.from_block(block_hash),
            storage_key=chunk_key,
            hbm_block_hash=block_hash,
        )
    ]
    return True, ops


class PlacementPolicy(ABC):
    """Where to retain KV copies when a block becomes resident on the local tier."""

    @abstractmethod
    def place_copy(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block: KVBlock,
        req: Request,
        now: float,
    ) -> None:
        """Synchronously mirror ``block`` to additional tier(s) if policy allows."""

    def spill_on_evict(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block: KVBlock,
        now: float,
        req: Request | None = None,
    ) -> None:
        """Called before a local-tier victim is removed. Default: drop (no spill)."""

    def plan_async_stores(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block_hash: str,
        req: Request,
    ) -> list[StoreOp]:
        return []

    def plan_spill_stores(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block_hash: str,
        req: Request,
    ) -> list[StoreOp]:
        return []

    def choose_targets(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block_hash: str,
        req: Request,
    ) -> tuple[str, ...]:
        """Optional dynamic downstream targets; default uses static edges only."""
        del memories, local_memory, block_hash, req
        return ()

    def store_prefix_on_complete(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        req: Request,
        now: float,
        tier_keys: tuple[str, ...],
    ) -> bool:
        del local_memory
        del memories
        del req
        del now
        del tier_keys
        return True


class HBMOnly(PlacementPolicy):
    """Keep computed KV on local tier only (default)."""

    def place_copy(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block: KVBlock,
        req: Request,
        now: float,
    ) -> None:
        pass


class TieredPlacement(PlacementPolicy):
    """Mirror/spill local tier to downstream tiers via ``PlacementEdge`` rules."""

    PRESSURE_THRESHOLD = 0.9

    def __init__(
        self,
        graph: TierGraph,
        edges: tuple[PlacementEdge, ...],
        *,
        mirror_on_forward: bool = True,
        retention: RetentionPolicy,
    ):
        self._graph = graph
        self._edges = edges
        self._allocator = TierAllocator()
        self._retention = retention
        self.mirror_on_forward = mirror_on_forward

    def _edges_for(self, trigger: str) -> list[PlacementEdge]:
        return [edge for edge in self._edges if edge.trigger == trigger]

    def _active_dst_tiers(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block_hash: str,
        req: Request,
        trigger: str,
    ) -> set[str]:
        dynamic = self.choose_targets(
            memories,
            local_memory=local_memory,
            block_hash=block_hash,
            req=req,
        )
        if dynamic:
            return set(dynamic)
        return {edge.dst_tier for edge in self._edges if edge.trigger == trigger}

    def choose_targets(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block_hash: str,
        req: Request,
    ) -> tuple[str, ...]:
        """Skip downstream tiers under pressure; empty tuple uses static edges only."""
        del local_memory, block_hash, req
        targets: list[str] = []
        for edge in self._edges:
            if edge.trigger != "forward":
                continue
            memory = memories.get(edge.dst_tier)
            if memory is None or memory.size <= 0:
                continue
            util = memory.used_size() / memory.size
            if util < self.PRESSURE_THRESHOLD:
                targets.append(edge.dst_tier)
        return tuple(targets)

    def _run_sync_edges(
        self,
        memories: dict[str, Memory],
        *,
        trigger: str,
        block_hash: str,
        req: Request,
        now: float,
        local_memory: str | None = None,
    ) -> bool:
        ok = True
        active = (
            self._active_dst_tiers(
                memories,
                local_memory=local_memory or "",
                block_hash=block_hash,
                req=req,
                trigger=trigger,
            )
            if local_memory is not None
            else {edge.dst_tier for edge in self._edges if edge.trigger == trigger}
        )
        for edge in self._edges_for(trigger):
            if edge.delivery != "sync":
                continue
            if edge.dst_tier not in active:
                continue
            success, _ = ensure_downstream_copy(
                self._graph,
                self._allocator,
                dst=edge.dst_tier,
                block_hash=block_hash,
                req=req,
                now=now,
                delivery="sync",
                exclude={block_hash},
                retention=self._retention,
            )
            ok = ok and success
        return ok

    def _plan_async_edges(
        self,
        memories: dict[str, Memory],
        *,
        trigger: str,
        block_hash: str,
        req: Request,
    ) -> list[StoreOp]:
        ops: list[StoreOp] = []
        for edge in self._edges_for(trigger):
            if edge.delivery != "async":
                continue
            _, edge_ops = ensure_downstream_copy(
                self._graph,
                self._allocator,
                dst=edge.dst_tier,
                block_hash=block_hash,
                req=req,
                now=0.0,
                delivery="async",
                exclude={block_hash},
                retention=self._retention,
            )
            ops.extend(edge_ops)
        return ops

    def place_copy(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block: KVBlock,
        req: Request,
        now: float,
    ) -> None:
        if not self.mirror_on_forward:
            return
        self._run_sync_edges(
            memories,
            trigger="forward",
            block_hash=block.hash,
            req=req,
            now=now,
            local_memory=local_memory,
        )

    def store_prefix_on_complete(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        req: Request,
        now: float,
        tier_keys: tuple[str, ...],
    ) -> bool:
        del local_memory
        ok = True
        for block_hash in req.block_hashes[: req.prefix_block_count]:
            if block_hash.startswith("blk:"):
                continue
            for edge in self._edges_for("complete"):
                if edge.dst_tier not in tier_keys:
                    continue
                if edge.delivery != "sync":
                    continue
                success, _ = ensure_downstream_copy(
                    self._graph,
                    self._allocator,
                    dst=edge.dst_tier,
                    block_hash=block_hash,
                    req=req,
                    now=now,
                    delivery="sync",
                    exclude={block_hash},
                    retention=self._retention,
                )
                ok = ok and success
        return ok

    def spill_on_evict(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block: KVBlock,
        now: float,
        req: Request | None = None,
    ) -> None:
        if req is None:
            assert not block.hash.startswith("blk:"), (
                "decode-generated blocks (blk:req:N) do not spill to downstream tiers"
            )
            return
        self._run_sync_edges(
            memories, trigger="evict", block_hash=block.hash, req=req, now=now
        )

    def plan_async_stores(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block_hash: str,
        req: Request,
    ) -> list[StoreOp]:
        return self._plan_async_edges(
            memories, trigger="forward", block_hash=block_hash, req=req
        )

    def plan_spill_stores(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        block_hash: str,
        req: Request,
    ) -> list[StoreOp]:
        return self._plan_async_edges(
            memories, trigger="evict", block_hash=block_hash, req=req
        )


@PLACEMENT.register("hbm_only")
def _hbm_only(_ctx: PolicyContext) -> PlacementPolicy:
    return HBMOnly()


@PLACEMENT.register("tiered")
def _tiered(ctx: PolicyContext) -> PlacementPolicy:
    params = ctx.params
    return TieredPlacement(
        ctx.graph,
        params["edges"],
        mirror_on_forward=params["mirror_on_forward"],
        retention=params["retention"],
    )
