from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from .content_key import ContentKey

if TYPE_CHECKING:
    from .request import Request
    from .tasks import Task


class BlockState(StrEnum):
    RESERVED = "reserved"
    EVICTING = "evicting"
    RESIDENT = "resident"
    LOADING = "loading"


class KVBlock:
    def __init__(
        self,
        hash: str,
        state: BlockState,
        task: Task | None = None,
        holders: set[str] | None = None,
        last_touch: float = 0.0,
    ):
        self.hash: str = hash
        self.state: BlockState = state
        self.task: Task | None = task
        self.holders: set[str] = holders if holders is not None else set()
        self.last_touch: float = last_touch

    def touch(self, t: float) -> None:
        self.last_touch = t


class Memory:
    def __init__(self, size: int, name: str, *, chunk_blocks: int = 1):
        self.name = name
        self.size = size
        self.chunk_blocks = max(1, chunk_blocks)
        self.blocks: dict[str, list[KVBlock]] = {}

    def get(self, block_hash: str) -> list[KVBlock]:
        return self.blocks.setdefault(block_hash, [])

    def used_size(self) -> int:
        return sum(len(copies) for copies in self.blocks.values())

    def free_size(self) -> int:
        return self.size - self.used_size()

    def append(self, block: KVBlock) -> None:
        self.get(block.hash).append(block)

    def remove_block(self, block: KVBlock) -> None:
        copies = self.blocks.get(block.hash)
        if copies is None:
            return
        copies.remove(block)
        if not copies:
            del self.blocks[block.hash]

    def best_resident(self, block_hash: str) -> KVBlock | None:
        return next(
            (b for b in self.get(block_hash) if b.state == BlockState.RESIDENT),
            None,
        )

    def resident_copies(self, block_hash: str) -> list[KVBlock]:
        return [b for b in self.get(block_hash) if b.state == BlockState.RESIDENT]

    def inflight_incoming(self, block_hash: str) -> KVBlock | None:
        """Block not yet resident: reserved slot, load in progress, or pending store."""
        return next(
            (
                b
                for b in self.get(block_hash)
                if b.state in (BlockState.RESERVED, BlockState.LOADING)
            ),
            None,
        )

    def find_reserved_for(self, block_hash: str, req_id: str) -> KVBlock | None:
        for block in reversed(self.get(block_hash)):
            if (
                block.state == BlockState.RESERVED
                and req_id in block.holders
                and block.task is None
            ):
                return block
        return None

    def list(self) -> list[KVBlock]:
        return [block for copies in self.blocks.values() for block in copies]

    def append_reserved(self, block_hash: str, req_id: str) -> KVBlock:
        block = KVBlock(block_hash, BlockState.RESERVED, holders={req_id})
        self.append(block)
        return block

    def free_request(self, req_id: str) -> None:
        """Drop all KV state for a request (vLLM kv_cache_manager.free)."""
        for copies in list(self.blocks.values()):
            for block in list(copies):
                if req_id not in block.holders:
                    continue
                block.holders.discard(req_id)
                if not block.holders:
                    self.remove_block(block)

    def can_evict_block(self, block: KVBlock) -> bool:
        return block.state == BlockState.RESIDENT and len(block.holders) == 0

    def touch(self, block: KVBlock, t: float) -> None:
        """Record last use time (simulation clock) for LRU eviction."""
        block.touch(t)

    def count_resident(self, block_hash: str) -> int:
        return len(self.resident_copies(block_hash))


def collect_content_copies(
    memories: dict[str, Memory],
    tier_keys: list[str],
    content: ContentKey,
    *,
    req: Request | None = None,
) -> list[tuple[str, KVBlock]]:
    """All resident copies of the same logical content across tiers."""
    from .content_key import tier_storage_key

    copies: list[tuple[str, KVBlock]] = []
    seen: set[tuple[str, int]] = set()
    anchor = content.anchor_hbm()
    for tier_key in tier_keys:
        mem = memories.get(tier_key)
        if mem is None:
            continue
        key = tier_storage_key(content, mem, req, anchor)
        for block in mem.resident_copies(key):
            token = (tier_key, id(block))
            if token in seen:
                continue
            seen.add(token)
            copies.append((tier_key, block))
    return copies
