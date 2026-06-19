from .kv_content import ContentKey
from .execute import BatchRunner
from .memory import KVBlock, Memory, collect_content_copies
from .plan import BatchPlan, StoreOp, WorkEntry, dedupe_batch_evicts
from .policies import EnginePolicies, enrich_entry_plan
from .request import Request, RequestPD, RequestStatus, request_owning_prefix_block
from .resource import BandwidthResource, ComputeResource
from .scheduler import Scheduler
from .tasks import TaskPool


class Engine:
    def __init__(
        self,
        engine_id: str,
        requests: list[Request],
        pool: TaskPool,
        memories: dict[str, Memory],
        policies: EnginePolicies,
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
        retain_hbm_prefix_cache: bool = False,
        store_tiers_on_complete: tuple[str, ...] = (),
        work_per_prefill_token: float | None = None,
        work_per_decode_req: float | None = None,
        sync_evict: bool = True,
    ):
        self.engine_id = engine_id
        self.pool = pool
        self.memories = memories
        self.policies = policies
        self.local_memory = policies.schedule.local_memory
        self.compute_res = compute_res
        self.bandwidth_res = bandwidth_res
        schedule = policies.schedule
        if transfer_links is None and bandwidth_res is not None and schedule.pull_sources:
            transfer_links = {src: bandwidth_res for src in schedule.pull_sources}
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
        self.retain_hbm_prefix_cache = retain_hbm_prefix_cache
        self.store_tiers_on_complete = tuple(store_tiers_on_complete)
        self._peak_duplicate_count = 0
        self._next_batch_id = 0

        self.scheduler = Scheduler(
            schedule,
            memories,
            self.local_memory,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            block_size=block_size,
            enable_chunked_prefill=enable_chunked_prefill,
            remote_kv_wait=remote_kv_wait,
        )
        for req in requests:
            self.scheduler.add_request(req)
        self.scheduler.set_release_handler(self._release_request_kv)

    def _release_request_kv(
        self,
        req: Request,
        *,
        preempted: bool,
        now: float,
    ) -> None:
        stored = True
        if not preempted and self.store_tiers_on_complete:
            stored = self.policies.effects.store_prefix_on_complete(
                self.memories,
                req=req,
                now=now,
                tier_keys=self.store_tiers_on_complete,
            )
        retain_hashes: set[str] | None = None
        prefix_hashes = set(req.block_hashes[: req.prefix_block_count])
        if self.retain_hbm_prefix_cache and not preempted:
            retain_hashes = prefix_hashes
        elif (
            not preempted
            and self.store_tiers_on_complete
            and not stored
        ):
            retain_hashes = prefix_hashes
        self.memories[self.local_memory].free_request(
            req.req_id,
            retain_hashes=retain_hashes,
        )

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

    def schedule_request(self, req: Request) -> None:
        self.scheduler.add_request(req)

    def release_arrivals(self, now: float) -> None:
        self.scheduler.release_arrivals(now)

    def next_arrival(self) -> float | None:
        return self.scheduler.next_arrival()

    def _known_requests(self) -> list[Request]:
        return [
            *self.scheduler.waiting,
            *self.scheduler.running,
            *self.scheduler.completed,
        ]

    def _resolve_spill_req(self, block_hash: str) -> Request | None:
        return request_owning_prefix_block(block_hash, self._known_requests())

    def _track_duplicates(
        self, req: Request | None, tier_key: str, block_hash: str
    ) -> None:
        del tier_key
        content = ContentKey.from_slot(block_hash)
        count = len(
            collect_content_copies(
                self.memories,
                list(self.memories.keys()),
                content,
                req=req,
            )
        )
        self._peak_duplicate_count = max(self._peak_duplicate_count, count)

    def _on_hbm_resident(self, block: KVBlock, req: Request, now: float) -> None:
        self.policies.effects.on_hbm_resident(self.memories, block, req, now)
        self._track_duplicates(req, self.local_memory, block.hash)

    def _on_tier_resident(
        self, tier_key: str, block: KVBlock, req: Request, now: float
    ) -> None:
        self.policies.effects.on_tier_resident(
            self.memories, tier_key, block, req, now
        )
        self._track_duplicates(req, tier_key, block.hash)

    def _after_pull(
        self, src_key: str, block_hash: str, req: Request, now: float
    ) -> None:
        del now
        self.policies.effects.after_pull(
            self.memories, src_key, block_hash, req
        )

    def _on_hbm_evict(self, victim: KVBlock, spill_req: Request, now: float) -> None:
        self.policies.effects.on_hbm_evict(self.memories, victim, spill_req, now)

    def _enrich_scheduled(self, scheduled) -> None:
        known = self._known_requests()
        for entry in scheduled.entries:
            enrich_entry_plan(
                entry.plan,
                entry=entry,
                effects=self.policies.effects,
                memories=self.memories,
                known_requests=known,
            )

    def _execute_plan(self, plan: BatchPlan) -> None:
        BatchRunner(self, plan.scheduled_at).run(plan)

    def try_schedule_and_execute(self, now: float) -> BatchPlan | None:
        """Admit → plan → enrich → execute; returns ``None`` when idle."""
        scheduled = self.scheduler.schedule(
            now,
            engine_id=self.engine_id,
        )
        if not scheduled.entries:
            return None
        self._enrich_scheduled(scheduled)
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        plan = BatchPlan(
            batch_id=batch_id,
            engine_id=self.engine_id,
            scheduled_at=now,
            entries=list(scheduled.entries),
            preempted=list(scheduled.preempted),
            total_num_scheduled_tokens=scheduled.total_num_scheduled_tokens,
        )
        dedupe_batch_evicts(plan)
        self._execute_plan(plan)
        return plan

    def apply_plan(self, plan: BatchPlan, now: float) -> tuple[list[Request], list[Request]]:
        """Advance request state after all batch tasks complete."""
        finished: list[Request] = []
        remote_kv_done: list[Request] = []

        for entry in plan.entries:
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
        req = next((r for r in self.completed if r.req_id == req_id), None)
        if req is None:
            self.memories[self.local_memory].free_request(req_id)
            return
        self._release_request_kv(req, preempted=False, now=0.0)
