"""Shared fixtures for simulator unit and critical tests."""

from __future__ import annotations

from simulator.runtime.engine import Engine
from simulator.runtime.engine_config import EngineLinks, EngineRuntime, WorkModel
from simulator.core.resource import BandwidthResource, ComputeResource
from simulator.policy.eviction import LRUEviction
from simulator.policy.schedule import ScheduleConfig, SchedulePolicy
from simulator.core.memory import BlockState, Memory
from simulator.runtime.plan import BatchPlan, WorkEntry
from simulator.policy.policies import EnginePolicies
from simulator.runtime.tasks import Task
from simulator.model.tier import Tier, TierGraph, graph_from_memories


class SimpleTask(Task):
    def on_start(self) -> None:
        pass

    def on_end(self) -> None:
        pass


def make_plan(
    entries: list[WorkEntry],
    *,
    batch_id: int = 0,
    engine_id: str = "e0",
    scheduled_at: float = 0.0,
) -> BatchPlan:
    return BatchPlan(
        batch_id=batch_id,
        engine_id=engine_id,
        scheduled_at=scheduled_at,
        entries=entries,
    )


def execute_plan(eng: Engine, plan: BatchPlan) -> list[Task]:
    eng._execute_plan(plan)
    return [t for t in eng.pool.tasks if t.batch_id == plan.batch_id]


def make_resident(memory: Memory, block_hash: str, req_id: str = "producer") -> None:
    memory.append_reserved(block_hash, req_id)
    block = memory.find_reserved_for(block_hash, req_id)
    assert block is not None
    block.state = BlockState.RESIDENT


def _graph_for(
    local_memory: str,
    memories: dict[str, Memory] | None = None,
) -> TierGraph:
    if memories:
        return graph_from_memories(memories)
    return TierGraph(
        tiers={
            local_memory: Tier(
                local_memory,
                Memory(size=100, name=local_memory),
                LRUEviction(),
            )
        }
    )


def schedule_compute(
    local_memory: str,
    *,
    eviction_policy=None,
):
    return SchedulePolicy(
        ScheduleConfig(
            local_memory=local_memory,
            local_eviction=eviction_policy or LRUEviction(),
        )
    )


def schedule_pull(
    local_memory: str,
    pull_sources: list[str] | tuple[str, ...],
    *,
    eviction_policy=None,
):
    return SchedulePolicy(
        ScheduleConfig(
            local_memory=local_memory,
            pull_sources=tuple(pull_sources),
            pull_mode="ordered_pull",
            local_eviction=eviction_policy or LRUEviction(),
        )
    )


def policies_compute(
    local_memory: str,
    *,
    memories: dict[str, Memory] | None = None,
    **kwargs,
) -> EnginePolicies:
    return EnginePolicies.compute_only(
        local_memory,
        graph=_graph_for(local_memory, memories),
        **kwargs,
    )


def policies_pull(
    local_memory: str,
    sources: list[str] | tuple[str, ...],
    *,
    memories: dict[str, Memory] | None = None,
    **kwargs,
) -> EnginePolicies:
    return EnginePolicies.ordered_pull(
        local_memory,
        sources,
        graph=_graph_for(local_memory, memories),
        **kwargs,
    )


def make_engine(
    engine_id: str,
    requests: list,
    pool,
    memories: dict[str, Memory],
    policies: EnginePolicies,
    compute_res: ComputeResource,
    bandwidth_res: BandwidthResource | None = None,
    *,
    work: WorkModel | None = None,
    links: EngineLinks | None = None,
    runtime: EngineRuntime | None = None,
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
    work_per_prefill_token: float | None = None,
    work_per_decode_req: float | None = None,
    sync_evict: bool = True,
    interconnect: BandwidthResource | None = None,
    event_trace=None,
) -> Engine:
    """Build an ``Engine`` from scalar test knobs (maps to config bundles)."""
    schedule = policies.schedule
    if work is None:
        work = WorkModel(
            per_block=work_per_block,
            per_transfer=work_per_transfer,
            per_store=work_per_store,
            per_evict=0.0 if work_per_evict is None else work_per_evict,
            per_prefill_token=work_per_prefill_token,
            per_decode_req=work_per_decode_req,
        )
    if runtime is None:
        runtime = EngineRuntime(
            block_size=block_size,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            remote_kv_wait=remote_kv_wait,
            sync_evict=sync_evict,
        )
    if links is None:
        resolved_transfer = transfer_links
        if resolved_transfer is None and bandwidth_res is not None and schedule.pull_sources:
            resolved_transfer = {
                src: bandwidth_res for src in schedule.pull_sources
            }
        links = EngineLinks(
            compute_res=compute_res,
            bandwidth_res=bandwidth_res,
            transfer_links=dict(resolved_transfer or {}),
            write_links=dict(write_links or {}),
            interconnect=interconnect,
        )
    return Engine(
        engine_id,
        requests,
        pool,
        memories,
        policies,
        links,
        work=work,
        runtime=runtime,
        event_trace=event_trace,
    )
