"""Synthetic PD workloads for stress tests and policy sweeps."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .request import Request, RequestPD, RequestStatus


@dataclass(frozen=True)
class WorkloadConfig:
    num_requests: int = 64
    shared_pool_size: int = 48
    min_prefix_blocks: int = 6
    max_prefix_blocks: int = 14
    min_output_blocks: int = 3
    max_output_blocks: int = 8
    arrival_spacing: float = 0.02
    arrival_jitter: float = 0.01
    seed: int = 42
    trace_path: str | None = None
    trace_offset: int = 0
    trace_time_scale: float = 0.001
    tokens_per_block: int = 512
    hash_prefix: str = "mooncake"


def build_workload(cfg: WorkloadConfig) -> tuple[list[Request], list[str]]:
    """Synthetic generator or Mooncake trace replay, depending on cfg."""
    if cfg.trace_path is not None:
        from .mooncake_trace import load_mooncake_trace

        return load_mooncake_trace(cfg)
    return generate_prefill_workload(cfg)


def generate_prefill_workload(
    cfg: WorkloadConfig,
) -> tuple[list[Request], list[str]]:
    """Build prefill requests with overlapping shared prefixes and unique tails."""
    rng = random.Random(cfg.seed)
    shared_pool = [f"shared:{i}" for i in range(cfg.shared_pool_size)]

    requests: list[Request] = []
    for i in range(cfg.num_requests):
        prefix_len = rng.randint(cfg.min_prefix_blocks, cfg.max_prefix_blocks)
        shared_count = rng.randint(prefix_len // 3, (2 * prefix_len) // 3)
        shared_count = max(1, min(shared_count, prefix_len - 1))

        blocks: list[str] = []
        start = rng.randint(0, cfg.shared_pool_size - 1)
        for j in range(shared_count):
            blocks.append(shared_pool[(start + j) % cfg.shared_pool_size])
        for j in range(prefix_len - shared_count):
            blocks.append(f"req{i:04d}:u{j}")

        arrival = i * cfg.arrival_spacing + rng.uniform(0.0, cfg.arrival_jitter)
        output_blocks = rng.randint(cfg.min_output_blocks, cfg.max_output_blocks)
        requests.append(
            Request(
                f"r{i:04d}",
                arrival,
                blocks,
                RequestPD.PREFILL,
                RequestStatus.PENDING,
                max_output_blocks=output_blocks,
            )
        )

    return requests, shared_pool
