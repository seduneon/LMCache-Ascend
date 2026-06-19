"""Forward and recompute work units (single source of truth for cost math)."""

from __future__ import annotations

from .plan import BatchPlan, WorkEntry
from .request import Request, RequestPD


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
