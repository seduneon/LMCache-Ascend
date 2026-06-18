"""Forward and recompute work units (single source of truth for cost math)."""

from __future__ import annotations

from .plan import BatchPlan, WorkEntry
from .request import Request


def prefill_work(num_tokens: int, work_per_prefill_token: float) -> float:
    return work_per_prefill_token * num_tokens


def decode_work(work_per_decode_req: float) -> float:
    return work_per_decode_req


def block_recompute_work(
    req: Request | None,
    *,
    block_size: int,
    work_per_prefill_token: float,
    work_per_decode_req: float,
    work_per_block: float,
) -> float:
    """Work for one block-level recompute decision during lookup."""
    if req is None:
        return work_per_block
    if req.is_prefill_chunk():
        return prefill_work(block_size, work_per_prefill_token)
    return decode_work(work_per_decode_req)


def entry_has_compute(entry: WorkEntry) -> bool:
    return any(action == "compute" for action in entry.plan.blocks.values())


def record_entry_metrics(entry: WorkEntry) -> None:
    """Update per-request counters from a scheduled work entry."""
    metrics = entry.req.metrics
    metrics.evictions += len(entry.plan.evicts)
    for block_hash in entry.block_hashes:
        action = entry.plan.blocks.get(block_hash)
        if action == "compute":
            metrics.computes += 1
        elif action == "wait":
            metrics.remote_waits += 1
        elif isinstance(action, tuple) and action[0] == "pull":
            metrics.pulls += 1
        elif action is None:
            metrics.local_hits += 1
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
