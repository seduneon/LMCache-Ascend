"""Shared fixtures for simulator unit and critical tests."""

from __future__ import annotations

from memory import BlockState, Memory
from tasks import Task


class SimpleTask(Task):
    def on_start(self) -> None:
        pass

    def on_end(self) -> None:
        pass


def make_resident(memory: Memory, block_hash: str, req_id: str = "producer") -> None:
    memory.append_reserved(block_hash, req_id)
    block = memory.find_reserved_for(block_hash, req_id)
    assert block is not None
    block.state = BlockState.RESIDENT
