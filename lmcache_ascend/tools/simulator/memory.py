from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .kv_content import ContentKey
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
        insert_seq: int = 0,
    ):
        self.hash: str = hash
        self.state: BlockState = state
        self.task: Task | None = task
        self.holders: set[str] = holders if holders is not None else set()
        self.last_touch: float = last_touch
        self.insert_seq: int = insert_seq
        self.access_count: int = 0

    def touch(self, t: float) -> None:
        self.last_touch = t
        self.access_count += 1


class Memory:
    def __init__(self, size: int, name: str, *, chunk_blocks: int = 1):
        self.name = name
        self.size = size
        self.chunk_blocks = max(1, chunk_blocks)
        self.blocks: dict[str, list[KVBlock]] = {}
        self._next_insert_seq = 0
        self.lifecycle_frees: int = 0
        self.tier_evictions: int = 0

    def get(self, block_hash: str) -> list[KVBlock]:
        return self.blocks.setdefault(block_hash, [])

    def used_size(self) -> int:
        return sum(len(copies) for copies in self.blocks.values())

    def free_size(self) -> int:
        return self.size - self.used_size()

    def append(self, block: KVBlock) -> None:
        block.insert_seq = self._next_insert_seq
        self._next_insert_seq += 1
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

    def free_request(
        self,
        req_id: str,
        *,
        retain_hashes: set[str] | None = None,
    ) -> int:
        """Drop a request's holders; remove unreferenced blocks unless retained.

        Returns the number of blocks removed from this tier (lifecycle frees).
        """
        removed = 0
        for copies in list(self.blocks.values()):
            for block in list(copies):
                if req_id not in block.holders:
                    continue
                block.holders.discard(req_id)
                if block.holders:
                    continue
                if retain_hashes is not None and block.hash in retain_hashes:
                    continue
                self.remove_block(block)
                removed += 1
        self.lifecycle_frees += removed
        return removed

    def can_evict_block(self, block: KVBlock) -> bool:
        return block.state == BlockState.RESIDENT and len(block.holders) == 0

    def touch(self, block: KVBlock, t: float) -> None:
        """Record last use time (simulation clock) for LRU eviction."""
        block.touch(t)

def collect_content_copies(
    memories: dict[str, Memory],
    tier_keys: list[str],
    content: ContentKey,
    *,
    req: Request | None = None,
) -> list[tuple[str, KVBlock]]:
    """All resident copies of the same logical content across tiers."""
    from .kv_content import storage_key

    copies: list[tuple[str, KVBlock]] = []
    seen: set[tuple[str, int]] = set()
    anchor = content.anchor()
    for tier_key in tier_keys:
        mem = memories.get(tier_key)
        if mem is None:
            continue
        key = storage_key(req, anchor, mem.chunk_blocks)
        for block in mem.resident_copies(key):
            token = (tier_key, id(block))
            if token in seen:
                continue
            seen.add(token)
            copies.append((tier_key, block))
    return copies
