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

    def inflight_incoming(self, block_hash: str) -> KVBlock | None:
        return next(
            (
                b
                for b in self.get(block_hash)
                if b.state in (BlockState.RESERVED, BlockState.LOADING) and b.task is not None
            ),
            None,
        )

    def evicting(self, block_hash: str) -> KVBlock | None:
        return next(
            (b for b in self.get(block_hash) if b.state == BlockState.EVICTING),
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

    def release_request(self, req_id: str) -> None:
        for copies in list(self.blocks.values()):
            for block in list(copies):
                block.holders.discard(req_id)
                if block.state == BlockState.RESERVED and not block.holders:
                    self.remove_block(block)

    def can_evict_block(self, block: KVBlock) -> bool:
        return block.state == BlockState.RESIDENT and len(block.holders) == 0
