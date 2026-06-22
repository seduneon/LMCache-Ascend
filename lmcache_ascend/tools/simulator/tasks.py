from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum

from .memory import BlockState, KVBlock, Memory
from .resource import Resource

BlockCallback = Callable[[KVBlock, float], None]


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"


_TERMINAL = frozenset({TaskStatus.COMPLETED})
_WORK_EPS = 1e-12
_TIME_EPS = 1e-9


class Task(ABC):
    def __init__(self, work_left: float, resource: Resource):
        self.prereqs: list[Task] = []
        self.work_left = work_left
        self.resource = resource
        self.status = TaskStatus.PENDING
        self.batch_id: int | None = None
        self._resource_reserved = False
        self.trace_meta: dict | None = None

    def reserve_resource(self) -> None:
        if self._resource_reserved:
            return
        self.resource.schedule()
        self._resource_reserved = True

    def is_ready(self) -> bool:
        if self.status != TaskStatus.PENDING:
            return False
        return not self.prereqs or all(
            p.status == TaskStatus.COMPLETED for p in self.prereqs
        )

    def start(self, time: float) -> None:
        assert self.status == TaskStatus.PENDING
        assert self.is_ready()
        assert self._resource_reserved, "task.start() without reserve_resource()"
        self.status = TaskStatus.RUNNING
        self.now = time + self.resource.latency
        self.resource.start()
        self.on_start()

    def _is_dust_work(self) -> bool:
        if self.work_left <= _WORK_EPS:
            return True
        rate = self.resource.speed()
        if rate <= 0:
            return False
        return self.work_left / rate <= _TIME_EPS

    def advance_to(self, time: float) -> None:
        assert self.status == TaskStatus.RUNNING
        if time <= self.now:
            return
        self.work_left -= (time - self.now) * self.resource.speed()
        self.now = time
        if self._is_dust_work():
            self.work_left = 0.0

    def is_done(self) -> bool:
        return self._is_dust_work()

    def finish(self) -> None:
        assert self.work_left <= 0
        self.work_left = 0
        self.complete()

    def complete(self) -> None:
        assert self.status == TaskStatus.RUNNING
        assert self.work_left == 0
        self.resource.finish()
        self.on_end()
        self.status = TaskStatus.COMPLETED

    @abstractmethod
    def on_start(self) -> None:
        pass

    @abstractmethod
    def on_end(self) -> None:
        pass

    def estimated_end(self):
        assert self.status == TaskStatus.RUNNING
        rate = self.resource.speed()
        if rate <= 0 or self._is_dust_work():
            return self.now
        end = self.now + self.work_left / rate
        if end <= self.now:
            return self.now + _TIME_EPS
        return end


class TaskPool:
    def __init__(self):
        self.tasks: list[Task] = []

    @staticmethod
    def filter(input: list[Task]) -> list[Task]:
        return [t for t in input if t.status not in _TERMINAL]

    def add(
        self,
        task: Task,
        prereqs: list[Task],
        *,
        batch_id: int | None = None,
    ) -> None:
        task.prereqs = self.filter(prereqs)
        if batch_id is not None:
            task.batch_id = batch_id
        task.reserve_resource()
        self.tasks.append(task)

    def compact(self) -> None:
        if any(t.status in _TERMINAL for t in self.tasks):
            self.tasks = [t for t in self.tasks if t.status not in _TERMINAL]

    def ready(self) -> list[Task]:
        return [t for t in self.tasks if t.is_ready()]

    def running(self) -> list[Task]:
        return [t for t in self.tasks if t.status == TaskStatus.RUNNING]

    def start_ready(self, t: float) -> None:
        for task in self.ready():
            task.start(t)

    def advance_running_to(self, t: float) -> None:
        for task in self.running():
            task.advance_to(t)

    def finish_done(self) -> None:
        for task in list(self.running()):
            if task.is_done():
                task.finish()


class MemoryTask(Task):
    def __init__(
        self,
        work_left: float,
        resource: Resource,
        memory: Memory,
        block: KVBlock,
    ):
        super().__init__(work_left, resource)
        self.memory = memory
        self.block = block


class ForwardTask(Task):
    """One batched model forward (vLLM: single GPU step for all compute in the batch)."""

    def __init__(
        self,
        work_left: float,
        resource: Resource,
        blocks: list[KVBlock],
        block_callbacks: dict[int, BlockCallback] | None = None,
    ):
        super().__init__(work_left, resource)
        self.blocks = blocks
        self._block_callbacks = block_callbacks or {}

    def on_start(self) -> None:
        for block in self.blocks:
            block.state = BlockState.LOADING

    def on_end(self) -> None:
        for block in self.blocks:
            block.state = BlockState.RESIDENT
            block.touch(self.now)
            callback = self._block_callbacks.get(id(block))
            if callback is not None:
                callback(block, self.now)


class BatchLoadTask(Task):
    """Pull multiple HBM blocks in one transfer (chunk latency amortized once)."""

    def __init__(
        self,
        work_left: float,
        resource: Resource,
        memory: Memory,
        blocks: list[KVBlock],
        block_callbacks: dict[int, BlockCallback],
    ):
        super().__init__(work_left, resource)
        self.memory = memory
        self.blocks = list(blocks)
        self._block_callbacks = dict(block_callbacks)

    def add_block(self, block: KVBlock, callback: BlockCallback) -> None:
        """Attach another destination block before the task starts (pull dedupe)."""
        assert self.status == TaskStatus.PENDING, "cannot extend pull after start"
        self.blocks.append(block)
        block.task = self
        self._block_callbacks[id(block)] = callback

    def on_start(self) -> None:
        for block in self.blocks:
            block.state = BlockState.LOADING
            block.task = self

    def on_end(self) -> None:
        for block in self.blocks:
            block.state = BlockState.RESIDENT
            block.task = None
            self.memory.touch(block, self.now)
            callback = self._block_callbacks.get(id(block))
            if callback is not None:
                callback(block, self.now)


class StoreTask(Task):
    """Write a chunk copy from local HBM to a downstream tier (paid bandwidth)."""

    def __init__(
        self,
        work_left: float,
        resource: Resource,
        tier: Memory,
        tier_block: KVBlock,
        *,
        on_complete: BlockCallback | None = None,
    ):
        super().__init__(work_left, resource)
        self.tier = tier
        self.tier_block = tier_block
        self._on_complete = on_complete

    def on_start(self) -> None:
        self.tier_block.state = BlockState.LOADING
        self.tier_block.task = self

    def on_end(self) -> None:
        self.tier_block.state = BlockState.RESIDENT
        self.tier_block.task = None
        self.tier.touch(self.tier_block, self.now)
        if self._on_complete is not None:
            self._on_complete(self.tier_block, self.now)


class EvictTask(MemoryTask):
    def __init__(
        self,
        work_left: float,
        resource: Resource,
        memory: Memory,
        block: KVBlock,
        *,
        on_evict_before: BlockCallback | None = None,
    ):
        super().__init__(work_left, resource, memory, block)
        self._on_evict_before = on_evict_before

    def on_start(self) -> None:
        if self._on_evict_before is not None:
            self._on_evict_before(self.block, self.now)
        self.block.state = BlockState.EVICTING
        self.block.task = self

    def on_end(self) -> None:
        self.memory.remove_block(self.block)
