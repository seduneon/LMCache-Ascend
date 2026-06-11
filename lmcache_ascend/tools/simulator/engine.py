import heapq
from collections import deque

from memory import Memory
from policies import LookupPolicy, LookupResult
from request import Request, RequestPD, RequestStatus
from resource import BandwidthResource, ComputeResource
from tasks import EvictTask, LoadTask, Task, TaskPool, TaskStatus


class Engine:
    def __init__(
        self,
        engine_id: str,
        requests: list[Request],
        pool: TaskPool,
        memories: dict[str, Memory],
        local_memory: str,
        policy: LookupPolicy,
        compute_res: ComputeResource,
        bandwidth_res: BandwidthResource | None = None,
        work_per_block: float = 1.0,
        work_per_transfer: float | None = None,
        work_per_evict: float | None = None,
    ):
        self.engine_id = engine_id
        self.pool = pool
        self.memories = memories
        self.local_memory = local_memory
        self.policy = policy
        self.compute_res = compute_res
        self.bandwidth_res = bandwidth_res
        self.work_per_block = work_per_block
        self.work_per_transfer = (
            work_per_transfer if work_per_transfer is not None else work_per_block
        )
        self.work_per_evict = work_per_evict if work_per_evict is not None else work_per_block

        self.pending: list[tuple[float, str, Request]] = []
        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []
        self.request_tasks: dict[str, list[Task]] = {}

        for r in requests:
            r.status = RequestStatus.PENDING
            heapq.heappush(self.pending, (r.arrival_time, r.req_id, r))

    def _local(self) -> Memory:
        return self.memories[self.local_memory]

    def schedule_request(self, req: Request) -> None:
        req.status = RequestStatus.PENDING
        heapq.heappush(self.pending, (req.arrival_time, req.req_id, req))

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
        local = self._local()

        for block_hash in req.block_hashes:
            if block_hash in result.blocks:
                local.append_reserved(block_hash, req.req_id)
                continue

            resident = local.best_resident(block_hash)
            if resident is not None:
                resident.holders.add(req.req_id)
                continue

            inflight = local.inflight_incoming(block_hash)
            if inflight is not None:
                inflight.holders.add(req.req_id)

    def _build_tasks(self, req: Request, result: LookupResult) -> None:
        local = self._local()
        tasks: list[Task] = []
        evict_tasks: list[Task] = []

        for victim in result.evicts:
            task = EvictTask(
                work_left=self.work_per_evict,
                resource=self.compute_res,
                memory=local,
                block=victim,
            )
            self.pool.add(task, [])
            evict_tasks.append(task)
            tasks.append(task)

        prereqs_tail = list(evict_tasks)

        for block_hash in req.block_hashes:
            inflight = local.inflight_incoming(block_hash)
            if inflight is not None and block_hash not in result.blocks:
                prereqs_tail = list(evict_tasks) + [inflight.task]
                continue

            if block_hash not in result.blocks:
                continue

            action = result.blocks[block_hash]
            dst_block = local.find_reserved_for(block_hash, req.req_id)
            if dst_block is None:
                raise RuntimeError(
                    f"no reserved block for {block_hash!r} request {req.req_id!r}"
                )

            if action == "compute":
                task = LoadTask(
                    work_left=self.work_per_block,
                    resource=self.compute_res,
                    memory=local,
                    block=dst_block,
                )
                self.pool.add(task, prereqs_tail)
                dst_block.task = task
                tasks.append(task)
                prereqs_tail = [task]
                continue

            if isinstance(action, tuple) and action[0] == "pull":
                if self.bandwidth_res is None:
                    raise RuntimeError("bandwidth_res required for pull")
                src_key = action[1]
                src_mem = self.memories[src_key]
                src_block = src_mem.best_resident(block_hash)
                if src_block is None:
                    raise RuntimeError(
                        f"no resident source block {block_hash!r} on {src_key!r}"
                    )
                prereqs = list(prereqs_tail)
                if src_block.task is not None and src_block.task not in prereqs:
                    prereqs.append(src_block.task)
                task = LoadTask(
                    work_left=self.work_per_transfer,
                    resource=self.bandwidth_res,
                    memory=local,
                    block=dst_block,
                )
                self.pool.add(task, prereqs)
                dst_block.task = task
                tasks.append(task)
                prereqs_tail = [task]
                continue

            raise NotImplementedError(f"unsupported action {action!r}")

        self.request_tasks[req.req_id] = tasks

    def check_completions(self) -> list[Request]:
        local = self._local()
        completed: list[Request] = []
        for req in list(self.active):
            tasks = self.request_tasks.get(req.req_id, [])
            if tasks and not all(t.status == TaskStatus.COMPLETED for t in tasks):
                continue
            req.status = RequestStatus.COMPLETE
            local.release_request(req.req_id)
            self.active.remove(req)
            self.request_tasks.pop(req.req_id, None)
            completed.append(req)
        return completed

    def next_arrival(self) -> float | None:
        return self.pending[0][0] if self.pending else None
