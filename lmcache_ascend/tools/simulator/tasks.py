from abc import ABC, abstractmethod
from enum import StrEnum

from memory import BlockState, KVBlock, Memory
from resource import Resource


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


_TERMINAL = frozenset({TaskStatus.COMPLETED, TaskStatus.CANCELLED})


def _is_ready(task: "Task") -> bool:
    if task.status != TaskStatus.PENDING:
        return False
    if not task.prereqs:
        return True
    if any(p.status == TaskStatus.CANCELLED for p in task.prereqs):
        return False
    return all(p.status == TaskStatus.COMPLETED for p in task.prereqs)


class Task(ABC):
    def __init__(self, work_left: float, resource: Resource, req_id: str | None = None):
        self.req_id = req_id
        self.wake: list[Task] = []
        self.prereqs: list[Task] = []
        self.work_left = work_left
        self.resource = resource
        self.status = TaskStatus.PENDING

    def start(self, time: float) -> None:
        assert self.status == TaskStatus.PENDING
        assert _is_ready(self)
        self.status = TaskStatus.RUNNING
        self.now = time
        self.resource.add()
        self.on_start()

    def advance_to(self, time: float) -> None:
        assert self.status == TaskStatus.RUNNING
        self.work_left -= (time - self.now) * self.resource.speed
        self.now = time

    def is_done(self) -> bool:
        return self.work_left <= 0

    def finish(self) -> None:
        assert self.work_left <= 0
        self.work_left = 0
        self.complete()

    def complete(self) -> None:
        assert self.status == TaskStatus.RUNNING
        assert self.work_left == 0
        self.resource.remove()
        self.on_end()
        self.status = TaskStatus.COMPLETED
        self.wake.clear()

    @abstractmethod
    def on_start(self) -> None:
        pass

    @abstractmethod
    def on_end(self) -> None:
        pass

    def on_cancel(self) -> None:
        pass

    def estimated_end(self):
        assert self.status == TaskStatus.RUNNING
        return self.resource.time_takes(self.work_left) + self.now


class TaskPool:
    def __init__(self):
        self.tasks: list[Task] = []

    @staticmethod
    def filter(input: list[Task]) -> list[Task]:
        return [t for t in input if t.status not in _TERMINAL]

    def add(self, task: Task, prereqs: list[Task]) -> None:
        prereqs = self.filter(prereqs)
        task.prereqs = list(prereqs)
        for prereq in prereqs:
            prereq.wake.append(task)
        self.tasks.append(task)

    def ready(self) -> list[Task]:
        self.tasks = self.filter(self.tasks)
        return [t for t in self.tasks if _is_ready(t)]

    def running(self) -> list[Task]:
        self.tasks = self.filter(self.tasks)
        return [t for t in self.tasks if t.status == TaskStatus.RUNNING]

    def next(self) -> float | None:
        running = self.running()
        if not running:
            return None
        return min(t.estimated_end() for t in running)

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

    def cancel_tasks(self, tasks: list[Task]) -> None:
        """Cancel tasks and same-request dependents.

        Readiness is derived from prereq status (_is_ready), not a needs counter.
        Cross-request dependents of a cancelled prereq stay pending (poisoned) and
        are not started; they are not auto-cancelled.
        """
        to_cancel: set[Task] = set()
        frontier = list(tasks)
        while frontier:
            task = frontier.pop()
            if task in to_cancel or task.status in _TERMINAL:
                continue
            to_cancel.add(task)
            for dep in task.wake:
                if dep.req_id is not None and dep.req_id == task.req_id:
                    frontier.append(dep)

        for task in to_cancel:
            if task.status == TaskStatus.RUNNING:
                task.resource.remove()
            task.on_cancel()
            task.status = TaskStatus.CANCELLED
            task.wake.clear()

        self.tasks = [t for t in self.tasks if t not in to_cancel]


class MemoryTask(Task):
    def __init__(
        self,
        work_left: float,
        resource: Resource,
        memory: Memory,
        block: KVBlock,
        req_id: str | None = None,
    ):
        super().__init__(work_left, resource, req_id=req_id)
        self.memory = memory
        self.block = block
        self.block_hash = block.hash


class LoadTask(MemoryTask):
    def on_start(self) -> None:
        self.block.state = BlockState.LOADING
        self.block.task = self

    def on_end(self) -> None:
        self.block.state = BlockState.RESIDENT
        self.block.task = None

    def on_cancel(self) -> None:
        if self.block.task is self:
            self.block.task = None
        if self.block.state == BlockState.LOADING:
            self.block.state = BlockState.RESERVED


class EvictTask(MemoryTask):
    def on_start(self) -> None:
        self.block.state = BlockState.EVICTING
        self.block.task = self

    def on_end(self) -> None:
        self.memory.remove_block(self.block)

    def on_cancel(self) -> None:
        if self.block.task is self:
            self.block.task = None
        if self.block.state == BlockState.EVICTING:
            self.block.state = BlockState.RESIDENT
