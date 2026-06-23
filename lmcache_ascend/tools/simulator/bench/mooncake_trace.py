"""Load Mooncake FAST'25 synthetic trace into simulator requests."""

from __future__ import annotations

import json
import math
from pathlib import Path

from simulator.core.request import Request, RequestPD, RequestStatus
from .workload import WorkloadConfig

DEFAULT_TRACE_PATH = (
    Path(__file__).resolve().parent.parent / "traces" / "synthetic_trace.jsonl"
)
MOONCAKE_TRACE_URL = (
    "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/"
    "FAST25-release/traces/synthetic_trace.jsonl"
)
MOONCAKE_TOKENS_PER_BLOCK = 512


def request_prefix_blocks(req: Request) -> int:
    """Prefix block count before the scheduler normalizes ``prefix_block_count``."""
    if req.prefix_block_count:
        return req.prefix_block_count
    return len(req.block_hashes)


def request_block_footprint(req: Request) -> int:
    """Peak HBM slots one request can hold on a single engine (prefix + decode)."""
    return request_prefix_blocks(req) + req.max_output_blocks


def is_hbm_admittable(req: Request, hbm_size: int) -> bool:
    """True if the request peak footprint fits in one engine's HBM."""
    return request_block_footprint(req) <= hbm_size


def partition_by_hbm(
    requests: list[Request],
    hbm_size: int,
) -> tuple[list[Request], list[Request]]:
    """Split requests into admittable vs physically oversized for ``hbm_size``."""
    admittable: list[Request] = []
    rejected: list[Request] = []
    for req in requests:
        if is_hbm_admittable(req, hbm_size):
            admittable.append(req)
        else:
            rejected.append(req)
    return admittable, rejected


def load_mooncake_trace(cfg: WorkloadConfig) -> tuple[list[Request], list[str]]:
    """Build prefill requests from a Mooncake JSONL trace slice."""
    if cfg.trace_path is None:
        raise ValueError("trace_path is required for Mooncake trace replay")

    path = Path(cfg.trace_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"trace file not found: {path} "
            f"(download from {MOONCAKE_TRACE_URL})"
        )

    requests: list[Request] = []
    shared: set[str] = set()
    line_index = 0
    loaded = 0

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if line_index < cfg.trace_offset:
                line_index += 1
                continue
            if loaded >= cfg.num_requests:
                break

            record = json.loads(raw)
            hash_ids: list[int] = record["hash_ids"]
            block_hashes = [f"{cfg.hash_prefix}:{h}" for h in hash_ids]
            shared.update(block_hashes)

            output_tokens = int(record["output_length"])
            max_output_blocks = max(
                1,
                math.ceil(output_tokens / cfg.tokens_per_block),
            )
            arrival = float(record["timestamp"]) * cfg.trace_time_scale

            requests.append(
                Request(
                    f"trace:{line_index}",
                    arrival,
                    block_hashes,
                    RequestPD.PREFILL,
                    RequestStatus.PENDING,
                    max_output_blocks=max_output_blocks,
                    prefix_block_count=len(block_hashes),
                )
            )
            loaded += 1
            line_index += 1

    if loaded < cfg.num_requests:
        raise ValueError(
            f"trace {path} has only {loaded} records after offset "
            f"{cfg.trace_offset}, need {cfg.num_requests}"
        )

    return requests, sorted(shared)
