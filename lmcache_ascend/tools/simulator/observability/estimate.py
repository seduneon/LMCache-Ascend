"""Forward/recompute work units and pull-vs-compute cost estimation."""

from __future__ import annotations

from dataclasses import dataclass, field

from simulator.core.kv_content import storage_key, tier_has_block, tier_has_inflight, transfer_work
from simulator.core.memory import Memory
from simulator.runtime.plan import BatchPlan, WorkEntry
from simulator.core.request import Request, RequestPD
from simulator.core.resource import BandwidthResource, ComputeResource, QueueDiscipline, Resource
from simulator.runtime.tasks import TaskStatus


def is_prefix_block(req: Request, block_hash: str) -> bool:
    """True for prompt/prefix KV blocks (excludes decode output ``blk:`` slots)."""
    if req.pd == RequestPD.PREFILL:
        return True
    return not block_hash.startswith("blk:")


def prefill_work(num_tokens: int, work_per_prefill_token: float) -> float:
    return work_per_prefill_token * num_tokens


def decode_work(work_per_decode_req: float) -> float:
    return work_per_decode_req


def entry_has_compute(entry: WorkEntry) -> bool:
    return any(action == "compute" for action in entry.plan.blocks.values())


def record_entry_metrics(entry: WorkEntry) -> None:
    """Update per-request counters from a scheduled work entry."""
    metrics = entry.req.metrics
    metrics.evictions += len(entry.plan.evicts)
    for block_hash in entry.block_hashes:
        action = entry.plan.blocks.get(block_hash)
        prefix = is_prefix_block(entry.req, block_hash)
        if action == "compute":
            metrics.computes += 1
            if prefix:
                metrics.prefix_computes += 1
        elif action == "wait":
            metrics.remote_waits += 1
        elif isinstance(action, tuple) and action[0] == "pull":
            metrics.pulls += 1
            if prefix:
                metrics.prefix_pulls += 1
                if action[1].endswith(":dram"):
                    metrics.prefix_dram_pulls += 1
        elif action is None:
            metrics.local_hits += 1
            if prefix:
                metrics.prefix_local_hits += 1
    if entry_has_compute(entry):
        metrics.forward_steps += 1


def entry_forward_work(
    req: Request,
    *,
    num_scheduled_tokens: int,
    has_compute: bool,
    work_per_prefill_token: float,
    work_per_decode_req: float,
) -> float:
    if not has_compute:
        return 0.0
    if req.is_prefill_chunk():
        return prefill_work(num_scheduled_tokens, work_per_prefill_token)
    return decode_work(work_per_decode_req)


def batch_forward_work(
    work: BatchPlan,
    *,
    work_per_prefill_token: float,
    work_per_decode_req: float,
) -> float:
    """Total forward work for all compute in a scheduled batch."""
    total = 0.0
    for entry in work.entries:
        total += entry_forward_work(
            entry.req,
            num_scheduled_tokens=entry.num_scheduled_tokens,
            has_compute=entry_has_compute(entry),
            work_per_prefill_token=work_per_prefill_token,
            work_per_decode_req=work_per_decode_req,
        )
    return total


@dataclass(frozen=True)
class BlockCostEstimate:
    """Estimated seconds per action for one block resolution."""

    pull: dict[str, float] = field(default_factory=dict)
    compute: float = 0.0
    wait: dict[str, float] = field(default_factory=dict)


@dataclass
class CostContext:
    """Bundle resources needed to estimate pull/compute/wait at admit time."""

    memories: dict[str, Memory]
    transfer_links: dict[str, BandwidthResource]
    compute_res: ComputeResource
    work_per_transfer: float
    work_per_prefill_token: float
    work_per_decode_req: float
    now: float = 0.0
    interconnect: Resource | None = None

    def estimate_pull(self, src_key: str, req: Request, block_hash: str) -> float:
        link = self.transfer_links.get(src_key)
        if link is None:
            return float("inf")
        src_mem = self.memories[src_key]
        work = self.work_per_transfer * transfer_work(
            src_mem, req, [block_hash]
        )
        if hasattr(link, "estimate_contention_works"):
            queued = link.estimate_contention_works()
        else:
            queued = link.queued_load() + 1
        if getattr(link, "queue_discipline", None) == QueueDiscipline.FIFO_TAIL:
            pull_time = link.latency + work / max(link.speed(1), 1e-12) * queued
        else:
            pull_time = link.time_for(work, works=queued)
        if self.interconnect is not None:
            ic = self.interconnect
            if hasattr(ic, "estimate_contention_works"):
                ic_queued = ic.estimate_contention_works()
            elif getattr(ic, "queue_discipline", None) == QueueDiscipline.FIFO_TAIL:
                ic_queued = ic.queue_position()
            else:
                ic_queued = ic.queued_load() + 1
            if getattr(ic, "queue_discipline", None) == QueueDiscipline.FIFO_TAIL:
                pull_time += ic.latency + work / max(ic.speed(1), 1e-12) * ic_queued
            else:
                pull_time += ic.time_for(work, works=ic_queued)
        return pull_time

    def estimate_compute(self, req: Request, block_hash: str) -> float:
        del block_hash
        if req.is_prefill_chunk():
            work = prefill_work(1, self.work_per_prefill_token)
        else:
            work = decode_work(self.work_per_decode_req)
        queued = self.compute_res.queued_load() + 1
        return self.compute_res.time_for(work, works=queued)

    def estimate_wait(
        self, src_key: str, req: Request, block_hash: str
    ) -> float | None:
        src = self.memories[src_key]
        if not tier_has_inflight(src, req, block_hash):
            return None
        key = storage_key(req, block_hash, src.chunk_blocks)
        inflight = src.inflight_incoming(key)
        if inflight is None:
            return None
        task = inflight.task
        if task is None:
            return self.estimate_pull(src_key, req, block_hash)
        if task.status == TaskStatus.RUNNING:
            return max(0.0, task.estimated_end() - self.now)
        return self.estimate_pull(src_key, req, block_hash)

    def estimate_block(
        self,
        pull_sources: list[str],
        block_hash: str,
        *,
        req: Request,
    ) -> BlockCostEstimate:
        pull: dict[str, float] = {}
        wait: dict[str, float] = {}
        for src_key in pull_sources:
            src = self.memories[src_key]
            if tier_has_block(src, req, block_hash):
                pull[src_key] = self.estimate_pull(src_key, req, block_hash)
            elif tier_has_inflight(src, req, block_hash):
                eta = self.estimate_wait(src_key, req, block_hash)
                if eta is not None:
                    wait[src_key] = eta
        compute = self.estimate_compute(req, block_hash)
        return BlockCostEstimate(pull=pull, compute=compute, wait=wait)
