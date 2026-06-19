from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable


class RequestPD(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"


class RequestStatus(StrEnum):
    PENDING = "pending"
    WAITING = "waiting"
    WAITING_REMOTE_KV = "waiting_remote_kv"
    RUNNING = "running"
    COMPLETE = "complete"


@dataclass
class RequestMetrics:
    """Per-request counters and phase timings (simulation clock)."""

    released_at: float | None = None
    finished_at: float | None = None

    queue_time: float = 0.0
    remote_kv_time: float = 0.0
    run_time: float = 0.0

    evictions: int = 0
    pulls: int = 0
    computes: int = 0
    local_hits: int = 0
    prefix_pulls: int = 0
    prefix_computes: int = 0
    prefix_local_hits: int = 0
    prefix_dram_pulls: int = 0
    remote_waits: int = 0
    preemptions: int = 0
    remote_kv_admits: int = 0
    forward_steps: int = 0

    engine_id: str | None = None
    _phase: str = field(default="pending", repr=False)
    _phase_since: float | None = field(default=None, repr=False)

    def _close_phase(self, now: float) -> None:
        if self._phase_since is None:
            return
        elapsed = now - self._phase_since
        if self._phase == "waiting":
            self.queue_time += elapsed
        elif self._phase == "remote_kv":
            self.remote_kv_time += elapsed
        elif self._phase == "running":
            self.run_time += elapsed
        self._phase_since = None

    def enter_waiting(self, now: float) -> None:
        self._close_phase(now)
        if self.released_at is None:
            self.released_at = now
        self._phase = "waiting"
        self._phase_since = now

    def enter_remote_kv(self, now: float, *, engine_id: str | None = None) -> None:
        self._close_phase(now)
        if engine_id is not None:
            self.engine_id = engine_id
        self.remote_kv_admits += 1
        self._phase = "remote_kv"
        self._phase_since = now

    def enter_running(self, now: float, *, engine_id: str | None = None) -> None:
        self._close_phase(now)
        if engine_id is not None:
            self.engine_id = engine_id
        self._phase = "running"
        self._phase_since = now

    def finish(self, now: float) -> None:
        self._close_phase(now)
        self.finished_at = now
        self._phase = "done"
        self._phase_since = None

    @property
    def latency(self) -> float | None:
        if self.released_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.released_at


@dataclass
class Request:
    req_id: str
    arrival_time: float
    block_hashes: list[str]
    pd: RequestPD
    status: RequestStatus
    max_output_blocks: int = 0
    num_computed_blocks: int = 0
    prefix_block_count: int = 0
    pending_block_hash: str | None = field(default=None, repr=False)
    prefill_engine_id: str | None = None
    kv_held_for_transfer: bool = False
    metrics: RequestMetrics = field(default_factory=RequestMetrics)

    @property
    def num_preemptions(self) -> int:
        return self.metrics.preemptions

    def total_blocks(self) -> int:
        return self.prefix_block_count + self.max_output_blocks

    def blocks_target(self) -> int:
        if self.pd == RequestPD.PREFILL:
            return self.prefix_block_count
        return self.total_blocks()

    def is_prefill_chunk(self) -> bool:
        """vLLM: num_computed_tokens < prompt length (here: prefix blocks)."""
        return self.num_computed_blocks < self.prefix_block_count


def request_owning_prefix_block(
    block_hash: str, requests: Iterable[Request]
) -> Request | None:
    """Find the request that owns a prefix block (for spill-on-evict)."""
    for req in requests:
        if block_hash in req.block_hashes[: req.prefix_block_count]:
            return req
    return None
