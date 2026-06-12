from memory import KVBlock, Memory
from policies import LookupResult
from request import Request, RequestPD, RequestStatus
from resource import BandwidthResource, ComputeResource
from scheduler import Batch, BatchEntry, Scheduler
from tasks import EvictTask, ForwardTask, LoadTask, Task, TaskPool


def _entry_has_compute(entry: BatchEntry) -> bool:
    return any(action == "compute" for action in entry.result.blocks.values())


def forward_cost(
    batch: Batch,
    *,
    work_per_prefill_token: float,
    work_per_decode_req: float,
) -> float:
    prefill_tokens = 0
    decode_reqs = 0
    for entry in batch.entries:
        if not _entry_has_compute(entry):
            continue
        if entry.req.is_prefill_chunk():
            prefill_tokens += entry.num_scheduled_tokens
        else:
            decode_reqs += 1
    return work_per_prefill_token * prefill_tokens + work_per_decode_req * decode_reqs


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
        block_size: int = 1,
        max_num_seqs: int = 10_000,
        max_num_batched_tokens: int = 10_000,
        enable_chunked_prefill: bool = False,
        remote_kv_wait: bool = False,
        hold_kv_on_complete: bool = False,
        work_per_prefill_token: float | None = None,
        work_per_decode_req: float | None = None,
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
        self.block_size = block_size
        self.work_per_prefill_token = (
            work_per_prefill_token if work_per_prefill_token is not None else work_per_block
        )
        self.work_per_decode_req = (
            work_per_decode_req if work_per_decode_req is not None else work_per_block
        )
        self.hold_kv_on_complete = hold_kv_on_complete
        self._remote_kv_wait = remote_kv_wait

        self.scheduler = Scheduler(
            policy=policy,
            memories=memories,
            local_memory=local_memory,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            block_size=block_size,
            enable_chunked_prefill=enable_chunked_prefill,
            remote_kv_wait=remote_kv_wait,
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

    @property
    def remote_kv_wait(self) -> bool:
        return self.scheduler.remote_kv_wait

    @remote_kv_wait.setter
    def remote_kv_wait(self, enabled: bool) -> None:
        self._remote_kv_wait = enabled
        self.scheduler.remote_kv_wait = enabled

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
        """Reserve memory, run evictions/pulls, then one batched forward for all compute."""
        local = self._local()
        all_tasks: list[Task] = []
        evict_tasks: list[Task] = []
        pull_tasks: list[Task] = []
        forward_blocks: list[KVBlock] = []

        for entry in batch.entries:
            self._reserve(entry.req, entry.result, entry.block_hashes)

            for victim in entry.result.evicts:
                task = EvictTask(
                    work_left=self.work_per_evict,
                    resource=self.compute_res,
                    memory=local,
                    block=victim,
                )
                self.pool.add(task, [])
                evict_tasks.append(task)
                all_tasks.append(task)

            prereqs_tail: list[Task] = list(evict_tasks)

            for block_hash in entry.block_hashes:
                if block_hash not in entry.result.blocks:
                    continue

                action = entry.result.blocks[block_hash]
                dst_block = local.find_reserved_for(block_hash, entry.req.req_id)
                if dst_block is None:
                    raise RuntimeError(
                        f"no reserved block for {block_hash!r} request {entry.req.req_id!r}"
                    )

                if action == "compute":
                    forward_blocks.append(dst_block)
                    continue

                if isinstance(action, tuple) and action[0] == "pull":
                    if self.bandwidth_res is None:
                        raise RuntimeError("bandwidth_res required for pull")
                    task = LoadTask(
                        work_left=self.work_per_transfer,
                        resource=self.bandwidth_res,
                        memory=local,
                        block=dst_block,
                    )
                    self.pool.add(task, prereqs_tail)
                    dst_block.task = task
                    pull_tasks.append(task)
                    all_tasks.append(task)
                    prereqs_tail = list(evict_tasks) + pull_tasks
                    continue

                raise NotImplementedError(f"unsupported action {action!r}")

        if forward_blocks:
            work = forward_cost(
                batch,
                work_per_prefill_token=self.work_per_prefill_token,
                work_per_decode_req=self.work_per_decode_req,
            )
            forward = ForwardTask(work, self.compute_res, forward_blocks)
            self.pool.add(forward, evict_tasks + pull_tasks)
            all_tasks.append(forward)

        return all_tasks

    def apply_batch(self, batch: Batch) -> tuple[list[Request], list[Request]]:
        """Advance state after the batch finishes (vLLM update_from_output)."""
        finished: list[Request] = []
        remote_kv_done: list[Request] = []

        for entry in batch.entries:
            req = entry.req
            if entry.remote_kv:
                self.scheduler.promote_remote_kv_complete(req)
                remote_kv_done.append(req)
                continue

            if req.pending_block_hash is not None:
                req.block_hashes.append(req.pending_block_hash)
                req.pending_block_hash = None
                req.num_computed_blocks += 1
            else:
                req.num_computed_blocks += len(entry.block_hashes)

            if req.num_computed_blocks >= req.blocks_target():
                if req.pd == RequestPD.PREFILL and self.hold_kv_on_complete:
                    self.scheduler.finish_prefill_held(req)
                else:
                    self.scheduler.finish_request(req)
                finished.append(req)

        return finished, remote_kv_done

    def release_held_kv(self, req_id: str) -> None:
        self._local().free_request(req_id)

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
