"""Retention policies: copy caps and post-pull source lifecycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from .eviction import EvictionPolicy, LRUEviction
from .kv_content import ContentKey, storage_key
from .memory import KVBlock, Memory, collect_content_copies
from .policy_registry import PolicyContext, Registry
from .request import Request
from .tier import TierGraph

RETENTION = Registry["RetentionPolicy"]("retention")


class PullDisposition(StrEnum):
    RETAIN = "retain"
    CONSUME = "consume"


class RetentionPolicy(ABC):
    """Caps per-tier duplicate residents and post-pull source lifecycle."""

    def __init__(self, graph: TierGraph | None = None):
        self._graph = graph

    def _eviction_for(self, tier_key: str) -> EvictionPolicy:
        if self._graph is not None:
            return self._graph.eviction_for(tier_key)
        # Standalone unit tests construct retention without a TierGraph.
        return LRUEviction()

    def max_copies(self, memory: Memory, block_hash: str) -> int | None:
        """Max resident copies of ``block_hash`` in ``memory``; ``None`` = unbounded."""
        return None

    def pull_disposition(self) -> PullDisposition:
        return PullDisposition.RETAIN

    def on_block_resident(
        self,
        memories: dict[str, Memory],
        *,
        tier_key: str,
        block: KVBlock,
        now: float,
        req: Request | None = None,
    ) -> None:
        memory = memories[tier_key]
        cap = self.max_copies(memory, block.hash)
        if cap is None:
            return
        self._trim_to_cap(memory, block.hash, cap, tier_key=tier_key)

    def after_pull(
        self,
        memories: dict[str, Memory],
        *,
        src_key: str,
        dst_key: str,
        block_hash: str,
        now: float,
        req: Request | None = None,
    ) -> None:
        if self.pull_disposition() != PullDisposition.CONSUME:
            return
        src = memories[src_key]
        slot = storage_key(req, block_hash, src.chunk_blocks)
        src_block = src.best_resident(slot)
        if src_block is None or not src.can_evict_block(src_block):
            return
        src.remove_block(src_block)

    def _trim_to_cap(
        self,
        memory: Memory,
        block_hash: str,
        cap: int,
        *,
        tier_key: str,
    ) -> None:
        copies = memory.resident_copies(block_hash)
        eviction = self._eviction_for(tier_key)
        while len(copies) > cap:
            exclude = {h for h in memory.blocks if h != block_hash}
            victims = eviction.pick_victims(memory, 1, exclude)
            if not victims:
                break
            memory.remove_block(victims[0])
            copies = memory.resident_copies(block_hash)


class UnboundedRetention(RetentionPolicy):
    """Default: unlimited copies per hash; pull leaves source resident."""

    def __init__(self, graph: TierGraph | None = None):
        super().__init__(graph)


class SingleCopyPerTier(RetentionPolicy):
    """At most one unheld resident copy per content hash per tier."""

    def max_copies(self, memory: Memory, block_hash: str) -> int:
        return 1


class ConsumeOnPull(RetentionPolicy):
    """Remove the pull source copy when it has no remaining holders."""

    def __init__(self, graph: TierGraph | None = None):
        super().__init__(graph)

    def pull_disposition(self) -> PullDisposition:
        return PullDisposition.CONSUME


class GlobalCopyCap(RetentionPolicy):
    """Cap total resident copies of a content key across selected tiers."""

    def __init__(
        self,
        max_total: int,
        tier_keys: list[str],
        *,
        per_tier_cap: int | None = 1,
        graph: TierGraph | None = None,
    ):
        super().__init__(graph)
        self.max_total = max_total
        self.tier_keys = list(tier_keys)
        self._per_tier_cap = per_tier_cap

    def max_copies(self, memory: Memory, block_hash: str) -> int | None:
        if self._per_tier_cap is None:
            return None
        return self._per_tier_cap

    def on_block_resident(
        self,
        memories: dict[str, Memory],
        *,
        tier_key: str,
        block: KVBlock,
        now: float,
        req: Request | None = None,
    ) -> None:
        super().on_block_resident(
            memories, tier_key=tier_key, block=block, now=now, req=req
        )
        content = ContentKey.from_slot(block.hash)
        while len(collect_content_copies(memories, self.tier_keys, content, req=req)) > self.max_total:
            copies = collect_content_copies(memories, self.tier_keys, content, req=req)
            evictable = [
                (tier, blk)
                for tier, blk in copies
                if memories[tier].can_evict_block(blk)
                and memories[tier].inflight_incoming(blk.hash) is None
            ]
            if not evictable:
                break
            tier_order = {t: i for i, t in enumerate(reversed(self.tier_keys))}
            victim_tier, victim = min(
                evictable,
                key=lambda item: (
                    tier_order.get(item[0], 0),
                    item[1].insert_seq,
                ),
            )
            mem = memories[victim_tier]
            eviction = self._eviction_for(victim_tier)
            trimmed = eviction.pick_victims(mem, 1, {victim.hash})
            if trimmed:
                mem.remove_block(trimmed[0])
            else:
                mem.remove_block(victim)


@RETENTION.register("unbounded")
def _unbounded(ctx: PolicyContext) -> RetentionPolicy:
    return UnboundedRetention(ctx.graph)


@RETENTION.register("single_copy")
def _single_copy(ctx: PolicyContext) -> RetentionPolicy:
    return SingleCopyPerTier(ctx.graph)


@RETENTION.register("consume_on_pull")
def _consume_on_pull(ctx: PolicyContext) -> RetentionPolicy:
    return ConsumeOnPull(ctx.graph)


@RETENTION.register("global_cap")
def _global_cap(ctx: PolicyContext) -> RetentionPolicy:
    params = ctx.params
    if "max_total" not in params or "tier_keys" not in params:
        raise ValueError(
            "global_cap retention requires max_total and tier_keys in params"
        )
    return GlobalCopyCap(
        int(params["max_total"]),
        list(params["tier_keys"]),
        per_tier_cap=params.get("per_tier_cap", 1),
        graph=ctx.graph,
    )
