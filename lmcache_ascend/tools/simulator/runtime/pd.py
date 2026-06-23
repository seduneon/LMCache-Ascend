from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Union

if TYPE_CHECKING:
    from simulator.runtime.engine import Engine


@dataclass
class PDConfig:
    """Read-mode PD: late decode spawn, D pre-alloc + WAITING_REMOTE_KV, deferred P release."""

    mode: Literal["read"] = "read"
    spawn_map: dict[str, str] = field(default_factory=dict)
    hold_kv_on_complete: bool | None = None

    def validate_and_apply(self, engines: dict[str, Engine]) -> None:
        if self.mode != "read":
            raise NotImplementedError(f"PD mode {self.mode!r} is not implemented")

        for prefill_id, decode_id in self.spawn_map.items():
            if prefill_id not in engines:
                raise ValueError(f"PD spawn_map prefill engine {prefill_id!r} not found")
            if decode_id not in engines:
                raise ValueError(f"PD spawn_map decode engine {decode_id!r} not found")

            prefill = engines[prefill_id]
            decode = engines[decode_id]
            if not decode.policies.schedule.pull_sources:
                raise ValueError(
                    f"decode engine {decode_id!r} needs pull_sources for PD read mode"
                )

            hold = (
                True
                if self.hold_kv_on_complete is None
                else self.hold_kv_on_complete
            )
            prefill.hold_kv_on_complete = hold
            decode.remote_kv_wait = True


# --- merged from events.py (cross-engine simulation events) ---


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
