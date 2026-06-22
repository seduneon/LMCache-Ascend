from .connector import TierCacheConnector
from .engine_config import LifecycleSpec
from .execute import BatchRunner
from .memory import KVBlock, Memory
from .plan import BatchPlan, dedupe_batch_evicts
from .policies import EnginePolicies
from .request import Request, RequestPD, RequestStatus
from .resource import BandwidthResource, ComputeResource
from .scheduler import Scheduler
from .tasks import TaskPool
from .tier import Tier, TierGraph


def _merge_tier_graph(graph: TierGraph, memories: dict[str, Memory]) -> TierGraph:
    tiers = dict(graph.tiers)
    for key, mem in memories.items():
        if key not in tiers:
            tiers[key] = Tier(key, mem, graph.eviction_for(key))
    return TierGraph(tiers=tiers)


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
        retain_prefix_cache: bool = False,
        store_tiers_on_complete: tuple[str, ...] = (),
        work_per_prefill_token: float | None = None,
        work_per_decode_req: float | None = None,
        sync_evict: bool = True,
        interconnect: BandwidthResource | None = None,
        event_trace=None,
    ):
        self.engine_id = engine_id
        self.pool = pool
        self.memories = memories
        merged_graph = _merge_tier_graph(policies.effects.config.graph, memories)
        if set(merged_graph.tiers) != set(policies.effects.config.graph.tiers):
            policies = EnginePolicies.with_graph(policies, merged_graph)
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
        self.retain_prefix_cache = retain_prefix_cache
        self.store_tiers_on_complete = tuple(store_tiers_on_complete)
        self.interconnect = interconnect
        self.event_trace = event_trace
        self._next_batch_id = 0
        if event_trace is not None:
            for tier in merged_graph.tiers.values():
                tier_key = tier.key

                def _tier_evict_cb(
                    now: float, block: KVBlock, *, key: str = tier_key
                ) -> None:
                    event_trace.on_evict(
                        now=now,
                        engine_id=engine_id,
                        tier=key,
                        block_hash=block.hash,
                        policy="tier",
                    )

                tier.on_tier_evict = _tier_evict_cb

        lifecycle = LifecycleSpec(
            hold_kv_on_complete=hold_kv_on_complete,
            retain_prefix_cache=retain_prefix_cache,
            store_on_complete=tuple(store_tiers_on_complete),
        )
        self.cache = TierCacheConnector(
            engine_id=engine_id,
            policies=policies,
            memories=memories,
            graph=merged_graph,
            lifecycle=lifecycle,
            local_tier=self.local_memory,
            compute_res=compute_res,
            transfer_links=self.transfer_links,
            write_links=self.write_links,
            work_per_transfer=self.work_per_transfer,
            work_per_prefill_token=self.work_per_prefill_token,
            work_per_decode_req=self.work_per_decode_req,
            interconnect=interconnect,
            event_trace=event_trace,
        )

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
        self.scheduler.set_release_handler(self.cache.release_request_kv)

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
        if self.event_trace is not None:
            self.event_trace.on_request_phase(
                now=req.arrival_time,
                engine_id=self.engine_id,
                req_id=req.req_id,
                phase=f"{req.pd.value}_admitted",
            )

    def release_arrivals(self, now: float) -> None:
        self.scheduler.release_arrivals(now)

    def next_arrival(self) -> float | None:
        return self.scheduler.next_arrival()

    def known_requests(self) -> list[Request]:
        return [
            *self.scheduler.waiting,
            *self.scheduler.running,
            *self.scheduler.completed,
        ]

    def _execute_plan(self, plan: BatchPlan) -> None:
        BatchRunner(self, plan.scheduled_at).run(plan)

    def try_schedule_and_execute(self, now: float) -> BatchPlan | None:
        """Admit → plan → enrich → execute; returns ``None`` when idle."""
        self.scheduler.set_cost_context(self.cache.build_cost_context(now))
        if self.event_trace is not None:
            self.scheduler.set_trace_writer(
                self.event_trace, engine_id=self.engine_id
            )
        scheduled = self.scheduler.schedule(
            now,
            engine_id=self.engine_id,
        )
        if not scheduled.entries:
            return None
        self.cache.enrich_plan(scheduled, known_requests=self.known_requests())
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
                if self.event_trace is not None:
                    self.event_trace.on_request_phase(
                        now=now,
                        engine_id=self.engine_id,
                        req_id=req.req_id,
                        phase=f"{req.pd.value}_complete",
                    )

        return finished, remote_kv_done

    def release_held_kv(self, req_id: str) -> None:
        req = next((r for r in self.completed if r.req_id == req_id), None)
        if req is None:
            self.memories[self.local_memory].free_request(req_id)
            return
        self.cache.release_request_kv(req, preempted=False, now=0.0)
