from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tasks import Task


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
    ):
        self.hash: str = hash
        self.state: BlockState = state
        self.task: Task | None = task
        self.holders: set[str] = holders if holders is not None else set()


class Memory:
    def __init__(self, size: int, name: str):
        self.name = name
        self.size = size
        self.blocks: dict[str, KVBlock] = {}

    def state_of(self, block_hash: str) -> BlockState | None:
        b = self.blocks.get(block_hash)
        return b.state if b else None

    def used_size(self) -> int:
        return len(self.blocks)

    def free_size(self) -> int:
        return self.size - self.used_size()

    def update(self, block_hash: str, state: BlockState, task: Task | None = None):
        existing = self.blocks.get(block_hash)
        holders = existing.holders.copy() if existing else set()
        self.blocks[block_hash] = KVBlock(block_hash, state, task, holders)

    def remove(self, block_hash: str):
        self.blocks.pop(block_hash, None)

    def find(self, block_hash: str) -> KVBlock | None:
        return self.blocks.get(block_hash)

    def list(self) -> list[KVBlock]:
        return list(self.blocks.values())

    def reserve(self, block_hash: str, req_id: str) -> None:
        existing = self.blocks.get(block_hash)
        if existing is None:
            self.blocks[block_hash] = KVBlock(
                block_hash, BlockState.RESERVED, holders={req_id}
            )
            return
        existing.holders.add(req_id)

    def add_holder(self, block_hash: str, req_id: str) -> None:
        block = self.blocks.get(block_hash)
        if block is None:
            raise KeyError(f"block {block_hash!r} not in memory")
        block.holders.add(req_id)

    def release_request(self, req_id: str) -> None:
        for block_hash in list(self.blocks):
            block = self.blocks[block_hash]
            block.holders.discard(req_id)
            if block.state == BlockState.RESERVED and not block.holders:
                self.remove(block_hash)

    def can_evict(self, block_hash: str) -> bool:
        block = self.blocks.get(block_hash)
        if block is None:
            return False
        return block.state == BlockState.RESIDENT and len(block.holders) == 0
