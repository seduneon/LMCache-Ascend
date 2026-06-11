from memory import Memory
from policies import LookupResult
from request import Request, RequestStatus
from resource import BandwidthResource, ComputeResource
from scheduler import Batch, BatchEntry, Scheduler
from tasks import EvictTask, LoadTask, Task, TaskPool, TaskStatus


class Engine:
    def __init__(
        self,
        engine_id: str,
        requests: list[Request],
        pool: TaskPool,
        memories: dict[str, Memory],
        local_memory: str,
        policy,
        compute_res: ComputeResource,
        bandwidth_res: BandwidthResource | None = None,
        work_per_block: float = 1.0,
        work_per_transfer: float | None = None,
        work_per_evict: float | None = None,
        max_blocks_per_step: int = 10_000,
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

        self.scheduler = Scheduler(
            policy=policy,
            memories=memories,
            local_memory=local_memory,
            max_blocks_per_step=max_blocks_per_step,
        )
        for req in requests:
            self.scheduler.add_request(req)

    @property
    def waiting(self):
        return self.scheduler.waiting

    @property
    def running(self):
        return self.scheduler.running

    @property
    def completed(self):
        return self.scheduler.completed

    def _local(self) -> Memory:
        return self.memories[self.local_memory]

    def schedule_request(self, req: Request) -> None:
        self.scheduler.add_request(req)

    def release_arrivals(self, now: float) -> None:
        self.scheduler.release_arrivals(now)

    def next_arrival(self) -> float | None:
        return self.scheduler.next_arrival()

    def schedule(self) -> Batch:
        return self.scheduler.schedule()

    def execute_batch(self, batch: Batch) -> list[Task]:
        """Reserve memory and enqueue all work for this batch."""
        all_tasks: list[Task] = []
        for entry in batch.entries:
            self._reserve(entry.req, entry.result, entry.block_hashes)
            tasks = self._build_tasks(entry.req, entry.result, entry.block_hashes)
            all_tasks.extend(tasks)
        return all_tasks

    def apply_batch(self, batch: Batch) -> list[Request]:
        """Advance state after the batch finishes (vLLM update_from_output)."""
        local = self._local()
        finished: list[Request] = []

        for entry in batch.entries:
            req = entry.req
            if req.num_computed_blocks < req.prefix_block_count:
                req.num_computed_blocks = req.prefix_block_count
            elif req.pending_block_hash is not None:
                req.block_hashes.append(req.pending_block_hash)
                req.pending_block_hash = None
                req.num_computed_blocks += 1

            if req.num_computed_blocks >= req.blocks_target():
                self.scheduler.finish_request(req)
                finished.append(req)

        return finished

    def _reserve(self, req: Request, result: LookupResult, block_hashes: list[str]) -> None:
        local = self._local()

        for block_hash in block_hashes:
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

    def _build_tasks(
        self,
        req: Request,
        result: LookupResult,
        block_hashes: list[str],
    ) -> list[Task]:
        local = self._local()
        tasks: list[Task] = []
        evict_tasks: list[Task] = []

        for victim in result.evicts:
            task = EvictTask(
                work_left=self.work_per_evict,
                resource=self.compute_res,
                memory=local,
                block=victim,
                req_id=req.req_id,
            )
            self.pool.add(task, [])
            evict_tasks.append(task)
            tasks.append(task)

        prereqs_tail: list[Task] = list(evict_tasks)

        for block_hash in block_hashes:
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
                    req_id=req.req_id,
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
                task = LoadTask(
                    work_left=self.work_per_transfer,
                    resource=self.bandwidth_res,
                    memory=local,
                    block=dst_block,
                    req_id=req.req_id,
                )
                self.pool.add(task, prereqs_tail)
                dst_block.task = task
                tasks.append(task)
                prereqs_tail = [task]
                continue

            raise NotImplementedError(f"unsupported action {action!r}")

        return tasks
