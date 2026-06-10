import heapq
from collections import deque

from memory import Memory
from policies import LookupPolicy, LookupResult
from request import Request, RequestStatus
from resource import ComputeResource
from tasks import LoadTask, Task, TaskPool, TaskStatus


class Engine:
    def __init__(
        self,
        requests: list[Request],
        pool: TaskPool,
        memories: dict[str, Memory],
        policy: LookupPolicy,
        compute_res: ComputeResource,
        work_per_block: float = 1.0,
    ):
        self.pool = pool
        self.memories = memories
        self.policy = policy
        self.compute_res = compute_res
        self.work_per_block = work_per_block

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
        self._build_tasks(req, result)
        self.check_completions()
        return True

    def _build_tasks(self, req: Request, result: LookupResult) -> None:
        hbm = self.memories["hbm"]
        tasks: list[Task] = []
        prev: Task | None = None

        for block_hash in req.block_hashes:
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
            prereqs = [prev] if prev else []
            self.pool.add(task, prereqs)
            tasks.append(task)
            prev = task

        self.request_tasks[req.req_id] = tasks

    def check_completions(self) -> None:
        for req in list(self.active):
            tasks = self.request_tasks.get(req.req_id, [])
            if tasks and not all(t.status == TaskStatus.COMPLETED for t in tasks):
                continue
            req.status = RequestStatus.COMPLETE
            self.active.remove(req)
            self.request_tasks.pop(req.req_id, None)

    def next_arrival(self) -> float | None:
        return self.pending[0][0] if self.pending else None

    def is_idle(self) -> bool:
        if self.pending or self.waiting or self.active:
            return False
        return not self.pool.running() and not self.pool.ready()
