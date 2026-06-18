from effect_interpreter import BatchExecutor, EngineRuntime
from kv_controller import KVController
from memory import Memory
from plan import BatchWork, ScheduleResult, SimContext
from policies import (
    HBMOnly,
    LookupPolicy,
    PlacementPolicy,
    RetentionPolicy,
    UnboundedRetention,
)
from request import Request, RequestPD, RequestStatus
from resource import BandwidthResource, ComputeResource
from scheduler import Scheduler
from tasks import Task, TaskPool


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
        transfer_links: dict[str, BandwidthResource] | None = None,
        write_links: dict[str, BandwidthResource] | None = None,
        work_per_block: float = 1.0,
        work_per_transfer: float | None = None,
        work_per_store: float | None = None,
        work_per_evict: float | None = None,
        block_size: int = 1,
        max_num_seqs: int = 10_000,
        max_num_batched_tokens: int = 10_000,
        enable_chunked_prefill: bool = False,
        remote_kv_wait: bool = False,
        hold_kv_on_complete: bool = False,
        work_per_prefill_token: float | None = None,
        work_per_decode_req: float | None = None,
        placement_policy: PlacementPolicy | None = None,
        retention_policy: RetentionPolicy | None = None,
        sync_evict: bool = True,
    ):
        self.engine_id = engine_id
        self.pool = pool
        self.memories = memories
        self.local_memory = local_memory
        self.policy = policy
        self.controller = KVController(policy)
        self.compute_res = compute_res
        self.bandwidth_res = bandwidth_res
        if transfer_links is None and bandwidth_res is not None and policy.pull_sources:
            transfer_links = {src: bandwidth_res for src in policy.pull_sources}
        self.transfer_links = transfer_links or {}
        self.write_links = write_links or {}
        self.work_per_block = work_per_block
        self.work_per_transfer = (
            work_per_transfer if work_per_transfer is not None else work_per_block
        )
        self.work_per_store = (
            work_per_store if work_per_store is not None else work_per_transfer
        )
        self.work_per_evict = work_per_evict if work_per_evict is not None else 0.0
        self.sync_evict = sync_evict
        self.block_size = block_size
        self.work_per_prefill_token = (
            work_per_prefill_token if work_per_prefill_token is not None else work_per_block
        )
        self.work_per_decode_req = (
            work_per_decode_req if work_per_decode_req is not None else work_per_block
        )
        self.hold_kv_on_complete = hold_kv_on_complete
        self.placement_policy = placement_policy or HBMOnly()
        self.retention_policy = retention_policy or UnboundedRetention()
        self.placement_policy.bind_retention(self.retention_policy)
        self._peak_duplicate_count = 0

        policy.bind_resources(
            compute_res=self.compute_res,
            transfer_links=self.transfer_links,
            work_per_transfer=self.work_per_transfer,
            work_per_block=self.work_per_block,
            work_per_prefill_token=self.work_per_prefill_token,
            work_per_decode_req=self.work_per_decode_req,
            block_size=self.block_size,
        )

        self.scheduler = Scheduler(
            self.controller,
            memories,
            local_memory,
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
        self.scheduler.remote_kv_wait = enabled

    def planning_context(self, now: float) -> SimContext:
        return SimContext.capture(
            now=now,
            local_memory=self.local_memory,
            block_size=self.block_size,
            compute_res=self.compute_res,
            transfer_links=self.transfer_links,
        )

    def schedule_request(self, req: Request) -> None:
        self.scheduler.add_request(req)

    def release_arrivals(self, now: float) -> None:
        self.scheduler.release_arrivals(now)

    def next_arrival(self) -> float | None:
        return self.scheduler.next_arrival()

    def schedule_batch(self, now: float) -> ScheduleResult:
        return self.scheduler.schedule(
            now,
            engine_id=self.engine_id,
            ctx=self.planning_context(now),
        )

    def make_work(self, scheduled: ScheduleResult, *, batch_id: int, now: float) -> BatchWork:
        return BatchWork.from_schedule(
            scheduled,
            batch_id=batch_id,
            engine_id=self.engine_id,
            scheduled_at=now,
        )

    def execute_work(self, work: BatchWork) -> list[Task]:
        rt = self._runtime()
        result = BatchExecutor(rt).execute(work)
        self._peak_duplicate_count = result.peak_duplicate_count
        return result.tasks

    def apply_work(self, work: BatchWork, now: float) -> tuple[list[Request], list[Request]]:
        """Advance request state after all batch tasks complete."""
        finished: list[Request] = []
        remote_kv_done: list[Request] = []

        for entry in work.entries:
            req = entry.req
            if entry.remote_kv:
                self.scheduler.promote_remote_kv_complete(
                    req, now=now, engine_id=self.engine_id
                )
                remote_kv_done.append(req)
                continue

            if req.pending_block_hash is not None:
                req.block_hashes.append(req.pending_block_hash)
                req.pending_block_hash = None
                req.num_computed_blocks += 1
            else:
                advanced = sum(
                    1
                    for block_hash in entry.block_hashes
                    if entry.plan.blocks.get(block_hash) != "wait"
                )
                req.num_computed_blocks += advanced

            if req.num_computed_blocks >= req.blocks_target():
                if req.pd == RequestPD.PREFILL and self.hold_kv_on_complete:
                    self.scheduler.finish_prefill_held(req, now=now)
                else:
                    self.scheduler.finish_request(req, now=now)
                finished.append(req)

        return finished, remote_kv_done

    def release_held_kv(self, req_id: str) -> None:
        self.memories[self.local_memory].free_request(req_id)

    def _runtime(self) -> EngineRuntime:
        return EngineRuntime(
            engine_id=self.engine_id,
            pool=self.pool,
            memories=self.memories,
            local_memory=self.local_memory,
            scheduler=self.scheduler,
            compute_res=self.compute_res,
            transfer_links=self.transfer_links,
            write_links=self.write_links,
            placement_policy=self.placement_policy,
            retention_policy=self.retention_policy,
            work_per_block=self.work_per_block,
            work_per_transfer=self.work_per_transfer,
            work_per_store=self.work_per_store,
            work_per_evict=self.work_per_evict,
            work_per_prefill_token=self.work_per_prefill_token,
            work_per_decode_req=self.work_per_decode_req,
            sync_evict=self.sync_evict,
            pull_sources=self.policy.pull_sources,
            peak_duplicate_count=self._peak_duplicate_count,
        )
