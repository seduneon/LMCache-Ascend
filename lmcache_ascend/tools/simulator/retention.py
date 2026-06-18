"""Retention policies: copy caps and post-pull source lifecycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from .chunk_hash import chunk_key_for_hbm_block
from .content_key import ContentKey
from .memory import KVBlock, Memory, collect_content_copies
from .request import Request


class PullDisposition(StrEnum):
    RETAIN = "retain"
    CONSUME = "consume"


class RetentionPolicy(ABC):
    """Caps per-tier duplicate residents and post-pull source lifecycle."""

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
        self._trim_to_cap(memory, block.hash, cap)

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
        storage_key = chunk_key_for_hbm_block(req, block_hash, src.chunk_blocks)
        src_block = src.best_resident(storage_key)
        if src_block is None or not src.can_evict_block(src_block):
            return
        src.remove_block(src_block)

    @staticmethod
    def _trim_to_cap(memory: Memory, block_hash: str, cap: int) -> None:
        copies = memory.resident_copies(block_hash)
        while len(copies) > cap:
            evictable = [b for b in copies if memory.can_evict_block(b)]
            if not evictable:
                break
            victim = min(evictable, key=lambda block: block.last_touch)
            memory.remove_block(victim)
            copies = memory.resident_copies(block_hash)


class UnboundedRetention(RetentionPolicy):
    """Default: unlimited copies per hash; pull leaves source resident."""


class SingleCopyPerTier(RetentionPolicy):
    """At most one unheld resident copy per content hash per tier."""

    def max_copies(self, memory: Memory, block_hash: str) -> int:
        return 1


class ConsumeOnPull(RetentionPolicy):
    """Remove the pull source copy when it has no remaining holders."""

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
    ):
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
        content = ContentKey.for_storage_key(block.hash)
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
                key=lambda item: (tier_order.get(item[0], 0), item[1].last_touch),
            )
            memories[victim_tier].remove_block(victim)
