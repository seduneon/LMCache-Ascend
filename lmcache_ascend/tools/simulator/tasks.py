from abc import ABC, abstractmethod
from enum import StrEnum

from memory import BlockState, KVBlock, Memory
from resource import Resource


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"


_TERMINAL = frozenset({TaskStatus.COMPLETED})
_WORK_EPS = 1e-12
_MAX_DRAIN_ITERS = 1_000_000


class Task(ABC):
    def __init__(self, work_left: float, resource: Resource):
        self.prereqs: list[Task] = []
        self.work_left = work_left
        self.resource = resource
        self.status = TaskStatus.PENDING

    def is_ready(self) -> bool:
        if self.status != TaskStatus.PENDING:
            return False
        return not self.prereqs or all(
            p.status == TaskStatus.COMPLETED for p in self.prereqs
        )

    def start(self, time: float) -> None:
        assert self.status == TaskStatus.PENDING
        assert self.is_ready()
        self.status = TaskStatus.RUNNING
        # Latency elapses before work begins (matches time_for used at schedule time).
        self.now = time + self.resource.latency
        self.resource.add()
        self.on_start()

    def advance_to(self, time: float) -> None:
        assert self.status == TaskStatus.RUNNING
        if time <= self.now:
            return
        self.work_left -= (time - self.now) * self.resource.speed()
        self.now = time
        if self.work_left <= _WORK_EPS:
            self.work_left = 0.0

    def is_done(self) -> bool:
        return self.work_left <= _WORK_EPS

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

    @abstractmethod
    def on_start(self) -> None:
        pass

    @abstractmethod
    def on_end(self) -> None:
        pass

    def estimated_end(self):
        assert self.status == TaskStatus.RUNNING
        rate = self.resource.speed()
        if rate <= 0 or self.work_left <= _WORK_EPS:
            return self.now
        return self.now + self.work_left / rate


class TaskPool:
    def __init__(self):
        self.tasks: list[Task] = []

    @staticmethod
    def filter(input: list[Task]) -> list[Task]:
        return [t for t in input if t.status not in _TERMINAL]

    def add(self, task: Task, prereqs: list[Task]) -> None:
        task.prereqs = self.filter(prereqs)
        self.tasks.append(task)

    def compact(self) -> None:
        """Drop completed tasks so idle checks stay cheap under load."""
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


def start_ready_tasks(tasks: list[Task], now: float) -> None:
    for task in tasks:
        if task.is_ready():
            task.start(now)


def _drain_status(tasks: list[Task]) -> str:
    pending = sum(1 for t in tasks if t.status == TaskStatus.PENDING)
    running = sum(1 for t in tasks if t.status == TaskStatus.RUNNING)
    done = sum(1 for t in tasks if t.status in _TERMINAL)
    return f"pending={pending} running={running} done={done}/{len(tasks)}"


def drain_tasks(
    tasks: list[Task],
    now: float,
    *,
    on_tick=None,
) -> tuple[float, int]:
    """Run a batch-local task DAG until done. Returns (finish_time, drain_iterations)."""
    if not tasks:
        return now, 0

    start_ready_tasks(tasks, now)
    iterations = 0
    t = now

    while not all(task.status in _TERMINAL for task in tasks):
        running = [task for task in tasks if task.status == TaskStatus.RUNNING]
        if not running:
            start_ready_tasks(tasks, t)
            running = [task for task in tasks if task.status == TaskStatus.RUNNING]
            if not running:
                pending = [task for task in tasks if task.status == TaskStatus.PENDING]
                raise RuntimeError(
                    "drain deadlock: no runnable tasks. "
                    f"{_drain_status(tasks)} pending_blocked={len(pending)}"
                )

        t_next = min(task.estimated_end() for task in running)
        if t_next <= t:
            stuck = [
                task
                for task in running
                if not task.is_done() and task.estimated_end() <= t + _WORK_EPS
            ]
            if stuck:
                raise RuntimeError(
                    "drain stuck: running tasks make no time progress at "
                    f"t={t:.6f} ({_drain_status(tasks)})"
                )
            for task in running:
                if task.is_done():
                    task.finish()
            start_ready_tasks(tasks, t)
            continue

        iterations += 1
        if iterations > _MAX_DRAIN_ITERS:
            raise RuntimeError(
                f"drain exceeded {_MAX_DRAIN_ITERS} iterations at t={t:.4f} "
                f"({_drain_status(tasks)})"
            )
        if on_tick is not None:
            on_tick(t, t_next, iterations, len(running), len(tasks))

        for task in running:
            task.advance_to(t_next)
        for task in running:
            if task.is_done():
                task.finish()
        t = t_next
        start_ready_tasks(tasks, t)

    return t, iterations


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
    ):
        super().__init__(work_left, resource)
        self.blocks = blocks

    def on_start(self) -> None:
        for block in self.blocks:
            block.state = BlockState.LOADING

    def on_end(self) -> None:
        for block in self.blocks:
            block.state = BlockState.RESIDENT


class LoadTask(MemoryTask):
    def on_start(self) -> None:
        self.block.state = BlockState.LOADING
        self.block.task = self

    def on_end(self) -> None:
        self.block.state = BlockState.RESIDENT
        self.block.task = None


class EvictTask(MemoryTask):
    def on_start(self) -> None:
        self.block.state = BlockState.EVICTING
        self.block.task = self

    def on_end(self) -> None:
        self.memory.remove_block(self.block)
