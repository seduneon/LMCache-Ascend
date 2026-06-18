"""Cross-engine simulation events (explicit dataflow instead of side effects)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class DecodeSpawn:
    req_id: str
    arrival_time: float
    prefix_blocks: tuple[str, ...]
    max_output_blocks: int
    prefix_block_count: int
    prefill_engine_id: str
    decode_engine_id: str


@dataclass(frozen=True)
class KvRelease:
    req_id: str
    prefill_engine_id: str


SimEvent = Union[DecodeSpawn, KvRelease]
