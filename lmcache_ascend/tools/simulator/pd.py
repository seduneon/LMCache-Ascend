from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .engine import Engine


@dataclass
class PDConfig:
    """Read-mode PD: late decode spawn, D pre-alloc + WAITING_REMOTE_KV, deferred P release."""

    mode: Literal["read"] = "read"
    spawn_map: dict[str, str] = field(default_factory=dict)

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
            if not decode.policy.pull_sources:
                raise ValueError(
                    f"decode engine {decode_id!r} needs pull_sources for PD read mode"
                )

            prefill.hold_kv_on_complete = True
            decode.remote_kv_wait = True
