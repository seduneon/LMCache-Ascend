from abc import ABC, abstractmethod
from resource import Resource
from enum import StrEnum
from memory import BlockState, Memory

class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"

class Task(ABC):
    def __init__(self, work_left: float, resource: Resource):
        self.wake: list[Task] = []
        self.needs = 0
        self.work_left = work_left
        self.resource = resource
        self.status = TaskStatus.PENDING
    
    def start(self, time: float) -> None:
        assert self.status == TaskStatus.PENDING
        assert self.needs == 0
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
        for next in self.wake:
            next.needs -= 1
        self.wake = []
        self.status = TaskStatus.COMPLETED
    
    @abstractmethod
    def on_start(self) -> None:
        pass

    @abstractmethod
    def on_end(self) -> None:
        pass

    def estimated_end(self):
        assert self.status == TaskStatus.RUNNING
        return self.resource.time_takes(self.work_left) + self.now

class TaskPool:
    def __init__(self):
        self.tasks: list[Task] = []

    @staticmethod
    def filter(input):
        return [t for t in input
                if t.status != TaskStatus.COMPLETED]
    
    def add(self, task: Task, prereqs: list[Task]):
        prereqs = self.filter(prereqs)
        task.needs = len(prereqs)
        for p in prereqs:
            p.wake.append(task)
        self.tasks.append(task)
    
    def ready(self) -> list[Task]:
        self.tasks = self.filter(self.tasks)
        return [t for t in self.tasks
                if t.status == TaskStatus.PENDING and t.needs == 0]
    
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

class MemoryTask(Task):
    def __init__(
        self,
        work_left: float,
        resource: Resource,
        memory: Memory,
        block_hash: str,
    ):
        super().__init__(work_left, resource)
        self.memory = memory
        self.block_hash = block_hash

class LoadTask(MemoryTask):
    def on_start(self) -> None:
        self.memory.update(self.block_hash, BlockState.LOADING, self)

    def on_end(self) -> None:
        self.memory.update(self.block_hash, BlockState.RESIDENT, None)

class EvictTask(MemoryTask):
    def on_start(self) -> None:
        self.memory.update(self.block_hash, BlockState.EVICTING, self)

    def on_end(self) -> None:
        self.memory.remove(self.block_hash)

