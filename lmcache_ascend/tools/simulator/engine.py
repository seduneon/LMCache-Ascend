from .kv_content import ContentKey
from .execute import BatchRunner
from .lookup import LookupPolicy
from .memory import KVBlock, Memory, collect_content_copies
from .placement import HBMOnly, PlacementPolicy
from .plan import BatchPlan, RetentionProfile, StoreOp, WorkEntry
from .request import Request, RequestPD, RequestStatus, request_owning_prefix_block
from .resource import BandwidthResource, ComputeResource
from .retention import (
    ConsumeOnPull,
    GlobalCopyCap,
    RetentionPolicy,
    SingleCopyPerTier,
    UnboundedRetention,
)
from .scheduler import Scheduler
from .tasks import TaskPool


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
        self.retention_profile = _retention_profile(self.retention_policy)
        self._peak_duplicate_count = 0
        self._next_batch_id = 0

        self.scheduler = Scheduler(
            policy,
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
        self.placement_policy.place_copy(
            self.memories,
            local_memory=self.local_memory,
            block=block,
            req=req,
            now=now,
        )
        self.retention_policy.on_block_resident(
            self.memories,
            tier_key=self.local_memory,
            block=block,
            now=now,
            req=req,
        )
        self._track_duplicates(req, self.local_memory, block.hash)

    def _on_tier_resident(
        self, tier_key: str, block: KVBlock, req: Request, now: float
    ) -> None:
        self.retention_policy.on_block_resident(
            self.memories,
            tier_key=tier_key,
            block=block,
            now=now,
            req=req,
        )
        self._track_duplicates(req, tier_key, block.hash)

    def _after_pull(
        self, src_key: str, block_hash: str, req: Request, now: float
    ) -> None:
        self.retention_policy.after_pull(
            self.memories,
            src_key=src_key,
            dst_key=self.local_memory,
            block_hash=block_hash,
            now=now,
            req=req,
        )

    def _on_hbm_evict(self, victim: KVBlock, spill_req: Request, now: float) -> None:
        self.placement_policy.spill_on_evict(
            self.memories,
            local_memory=self.local_memory,
            block=victim,
            now=now,
            req=spill_req,
        )

    def _expand_async_stores(self, entry: WorkEntry, block_hash: str) -> list[StoreOp]:
        return self.placement_policy.plan_async_stores(
            self.memories,
            local_memory=self.local_memory,
            block_hash=block_hash,
            req=entry.req,
        )

    def _expand_spill_stores(self, entry: WorkEntry, victim: KVBlock) -> list[StoreOp]:
        spill_req = self._resolve_spill_req(victim.hash)
        if spill_req is None:
            return []
        ops = self.placement_policy.plan_spill_stores(
            self.memories,
            local_memory=self.local_memory,
            block_hash=victim.hash,
            req=spill_req,
        )
        if ops:
            entry.plan.spill_store_ops[id(victim)] = ops
        return ops

    def _execute_plan(self, plan: BatchPlan) -> None:
        BatchRunner(self, plan.scheduled_at).run(plan)

    def try_schedule_and_execute(self, now: float) -> BatchPlan | None:
        """Admit → plan → execute in one call; returns ``None`` when idle."""
        scheduled = self.scheduler.schedule(
            now,
            engine_id=self.engine_id,
        )
        if not scheduled.entries:
            return None
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        plan = BatchPlan(
            batch_id=batch_id,
            engine_id=self.engine_id,
            scheduled_at=now,
            entries=list(scheduled.entries),
            preempted=list(scheduled.preempted),
            total_num_scheduled_tokens=scheduled.total_num_scheduled_tokens,
            retention=self.retention_profile,
        )
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
        self.memories[self.local_memory].free_request(req_id)


def _retention_profile(policy: RetentionPolicy) -> RetentionProfile:
    if isinstance(policy, ConsumeOnPull):
        return RetentionProfile(kind="consume_on_pull")
    if isinstance(policy, SingleCopyPerTier):
        return RetentionProfile(kind="single_copy")
    if isinstance(policy, GlobalCopyCap):
        return RetentionProfile(
            kind="global_cap",
            max_total=policy.max_total,
            tier_keys=tuple(policy.tier_keys),
            per_tier_cap=policy._per_tier_cap,
        )
    return RetentionProfile(kind="unbounded")
