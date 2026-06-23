from __future__ import annotations

from typing import TYPE_CHECKING

from .connector import TierCacheConnector
from .engine_config import EngineLinks, EngineRuntime, WorkModel
from simulator.policy.eviction import LRUEviction
from .execute import BatchRunner
from simulator.core.memory import KVBlock, Memory
from .plan import BatchPlan, dedupe_batch_evicts
from simulator.policy.policies import EnginePolicies
from simulator.core.request import Request, RequestPD, RequestStatus
from simulator.core.resource import BandwidthResource, ComputeResource
from .scheduler import Scheduler
from .tasks import TaskPool
from simulator.model.tier import Tier, TierGraph

if TYPE_CHECKING:
    from simulator.observability.event_trace import EventTraceWriter


def _merge_tier_graph(graph: TierGraph, memories: dict[str, Memory]) -> TierGraph:
    tiers = dict(graph.tiers)
    for key, mem in memories.items():
        if key not in tiers:
            try:
                eviction = graph.eviction_for(key)
            except KeyError:
                eviction = LRUEviction()
            tiers[key] = Tier(key, mem, eviction)
    return TierGraph(tiers=tiers)


def _install_tier_evict_trace(
    graph: TierGraph,
    *,
    engine_id: str,
    event_trace: EventTraceWriter,
) -> None:
    for tier in graph.tiers.values():
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


class Engine:
    def __init__(
        self,
        engine_id: str,
        requests: list[Request],
        pool: TaskPool,
        memories: dict[str, Memory],
        policies: EnginePolicies,
        links: EngineLinks,
        *,
        work: WorkModel = WorkModel(),
        runtime: EngineRuntime = EngineRuntime(),
        event_trace: EventTraceWriter | None = None,
    ):
        self.engine_id = engine_id
        self.pool = pool
        self.memories = memories
        merged_graph = _merge_tier_graph(policies.effects.config.graph, memories)
        if set(merged_graph.tiers) != set(policies.effects.config.graph.tiers):
            policies = EnginePolicies.with_graph(policies, merged_graph)
        self.policies = policies
        self.local_memory = policies.schedule.local_memory
        schedule = policies.schedule

        self._work = work
        self._links = links
        self._runtime = runtime
        self.event_trace = event_trace
        self._next_batch_id = 0

        if event_trace is not None:
            _install_tier_evict_trace(
                merged_graph, engine_id=engine_id, event_trace=event_trace
            )

        self.cache = TierCacheConnector(
            engine_id=engine_id,
            policies=policies,
            memories=memories,
            graph=merged_graph,
            lifecycle=policies.config.lifecycle,
            local_tier=self.local_memory,
            compute_res=links.compute_res,
            transfer_links=links.transfer_links,
            write_links=links.write_links,
            work_per_transfer=work.resolved_transfer(),
            work_per_prefill_token=work.resolved_prefill_token(),
            work_per_decode_req=work.resolved_decode_req(),
            interconnect=links.interconnect,
            event_trace=event_trace,
        )

        self.scheduler = Scheduler(
            schedule,
            memories,
            self.local_memory,
            max_num_seqs=runtime.max_num_seqs,
            max_num_batched_tokens=runtime.max_num_batched_tokens,
            block_size=runtime.block_size,
            enable_chunked_prefill=runtime.enable_chunked_prefill,
            remote_kv_wait=runtime.remote_kv_wait,
        )
        for req in requests:
            self.scheduler.add_request(req)
        self.scheduler.set_release_handler(self.cache.release_request_kv)

    @property
    def work(self) -> WorkModel:
        return self._work

    @property
    def links(self) -> EngineLinks:
        return self._links

    @property
    def runtime(self) -> EngineRuntime:
        return self._runtime

    @property
    def compute_res(self) -> ComputeResource:
        return self._links.compute_res

    @property
    def bandwidth_res(self) -> BandwidthResource | None:
        return self._links.bandwidth_res

    @property
    def transfer_links(self) -> dict[str, BandwidthResource]:
        return self._links.transfer_links

    @property
    def write_links(self) -> dict[str, BandwidthResource]:
        return self._links.write_links

    @property
    def interconnect(self) -> BandwidthResource | None:
        return self._links.interconnect

    @property
    def work_per_block(self) -> float:
        return self._work.per_block

    @property
    def work_per_transfer(self) -> float:
        return self._work.resolved_transfer()

    @property
    def work_per_store(self) -> float:
        return self._work.resolved_store()

    @property
    def work_per_evict(self) -> float:
        return self._work.per_evict

    @property
    def work_per_prefill_token(self) -> float:
        return self._work.resolved_prefill_token()

    @property
    def work_per_decode_req(self) -> float:
        return self._work.resolved_decode_req()

    @property
    def sync_evict(self) -> bool:
        return self._runtime.sync_evict

    @property
    def block_size(self) -> int:
        return self._runtime.block_size

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
    def hold_kv_on_complete(self) -> bool:
        return self.policies.config.lifecycle.hold_kv_on_complete

    @hold_kv_on_complete.setter
    def hold_kv_on_complete(self, value: bool) -> None:
        from dataclasses import replace

        lifecycle = replace(
            self.policies.config.lifecycle, hold_kv_on_complete=value
        )
        config = replace(self.policies.config, lifecycle=lifecycle)
        graph = self.policies.effects.config.graph
        self.policies = EnginePolicies.from_config(config, graph)
        self.cache.lifecycle = lifecycle

    @property
    def retain_prefix_cache(self) -> bool:
        return self.policies.config.lifecycle.retain_prefix_cache

    @property
    def store_tiers_on_complete(self) -> tuple[str, ...]:
        return self.policies.config.lifecycle.store_on_complete

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
                if req.is_prefill() and self.hold_kv_on_complete:
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
            return
        self.cache.release_request_kv(req, preempted=False, now=0.0)
