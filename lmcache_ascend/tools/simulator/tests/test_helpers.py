"""Shared fixtures for simulator unit and critical tests."""

from __future__ import annotations

from simulator.memory import BlockState, Memory
from simulator.plan import BatchPlan, WorkEntry
from simulator.tasks import Task


class SimpleTask(Task):
    def on_start(self) -> None:
        pass

    def on_end(self) -> None:
        pass


def make_plan(
    entries: list[WorkEntry],
    *,
    batch_id: int = 0,
    engine_id: str = "e0",
    scheduled_at: float = 0.0,
) -> BatchPlan:
    return BatchPlan(
        batch_id=batch_id,
        engine_id=engine_id,
        scheduled_at=scheduled_at,
        entries=entries,
    )


make_work = make_plan  # backward-compatible alias for tests


def make_resident(memory: Memory, block_hash: str, req_id: str = "producer") -> None:
    memory.append_reserved(block_hash, req_id)
    block = memory.find_reserved_for(block_hash, req_id)
    assert block is not None
    block.state = BlockState.RESIDENT
