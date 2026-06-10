import heapq
from collections import deque

from memory import BlockState, Memory
from policies import LookupPolicy, LookupResult
from request import Request, RequestStatus
from resource import ComputeResource
from tasks import EvictTask, LoadTask, Task, TaskPool, TaskStatus


class Engine:
    def __init__(
        self,
        requests: list[Request],
        pool: TaskPool,
        memories: dict[str, Memory],
        policy: LookupPolicy,
        compute_res: ComputeResource,
        work_per_block: float = 1.0,
        work_per_evict: float | None = None,
    ):
        self.pool = pool
        self.memories = memories
        self.policy = policy
        self.compute_res = compute_res
        self.work_per_block = work_per_block
        self.work_per_evict = work_per_evict if work_per_evict is not None else work_per_block

        self.pending: list[tuple[float, str, Request]] = []
        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []
        self.request_tasks: dict[str, list[Task]] = {}

        for r in requests:
            r.status = RequestStatus.PENDING
            heapq.heappush(self.pending, (r.arrival_time, r.req_id, r))

    def release_arrivals(self, now: float) -> None:
        while self.pending and self.pending[0][0] <= now:
            _, _, r = heapq.heappop(self.pending)
            r.status = RequestStatus.WAITING
            self.waiting.append(r)

    def admit(self) -> bool:
        """Admit the head waiting request if lookup succeeds. Returns True if admitted."""
        if not self.waiting:
            return False

        req = self.waiting[0]
        result = self.policy.lookup(self.memories, req.block_hashes)
        if result is None:
            return False

        self.waiting.popleft()
        req.status = RequestStatus.RUNNING
        self.active.append(req)
        self._reserve(req, result)
        self._build_tasks(req, result)
        self.check_completions()
        return True

    def _reserve(self, req: Request, result: LookupResult) -> None:
        hbm = self.memories["hbm"]

        for block_hash in req.block_hashes:
            if block_hash in result.blocks:
                hbm.reserve(block_hash, req.req_id)
            elif hbm.state_of(block_hash) == BlockState.RESIDENT:
                hbm.add_holder(block_hash, req.req_id)
            elif hbm.state_of(block_hash) in (BlockState.RESERVED, BlockState.LOADING):
                hbm.add_holder(block_hash, req.req_id)

    def _build_tasks(self, req: Request, result: LookupResult) -> None:
        hbm = self.memories["hbm"]
        tasks: list[Task] = []
        evict_tasks: list[Task] = []

        for victim in result.evicts:
            task = EvictTask(
                work_left=self.work_per_evict,
                resource=self.compute_res,
                memory=hbm,
                block_hash=victim,
            )
            self.pool.add(task, [])
            evict_tasks.append(task)
            tasks.append(task)

        prereqs_tail = list(evict_tasks)

        for block_hash in req.block_hashes:
            existing = hbm.find(block_hash)
            if existing and existing.task is not None:
                prereqs_tail = list(evict_tasks) + [existing.task]
                continue

            if block_hash not in result.blocks:
                continue

            action = result.blocks[block_hash]
            if action != "compute":
                raise NotImplementedError(f"v0 only supports compute, got {action!r}")

            task = LoadTask(
                work_left=self.work_per_block,
                resource=self.compute_res,
                memory=hbm,
                block_hash=block_hash,
            )
            self.pool.add(task, prereqs_tail)
            if existing is not None:
                existing.task = task
            tasks.append(task)
            prereqs_tail = [task]

        self.request_tasks[req.req_id] = tasks

    def check_completions(self) -> None:
        hbm = self.memories["hbm"]
        for req in list(self.active):
            tasks = self.request_tasks.get(req.req_id, [])
            if tasks and not all(t.status == TaskStatus.COMPLETED for t in tasks):
                continue
            req.status = RequestStatus.COMPLETE
            hbm.release_request(req.req_id)
            self.active.remove(req)
            self.request_tasks.pop(req.req_id, None)

    def next_arrival(self) -> float | None:
        return self.pending[0][0] if self.pending else None

    def is_idle(self) -> bool:
        if self.pending or self.waiting or self.active:
            return False
        return not self.pool.running() and not self.pool.ready()
