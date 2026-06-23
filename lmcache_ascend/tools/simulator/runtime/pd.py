from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Union

from simulator.model.layout import EngineRole
from simulator.policy.registry import PolicyContext
from simulator.policy.routing import ROUTING, RoutingPolicy

if TYPE_CHECKING:
    from simulator.runtime.engine import Engine


@dataclass
class PDConfig:
    """Read-mode PD: late decode spawn, D pre-alloc + WAITING_REMOTE_KV, deferred P release."""

    mode: Literal["read"] = "read"
    routing: RoutingPolicy | None = None
    spawn_map: dict[str, str] = field(default_factory=dict)
    engine_role: dict[str, EngineRole] = field(default_factory=dict)
    hold_kv_on_complete: bool | None = None

    def resolved_routing(self) -> RoutingPolicy:
        if self.routing is not None:
            return self.routing
        if self.spawn_map:
            return ROUTING.create(
                "bijection", PolicyContext(params={"map": self.spawn_map})
            )
        raise ValueError("PDConfig requires routing or spawn_map")

    def resolved_engine_role(self, engines: dict[str, Engine]) -> dict[str, EngineRole]:
        if self.engine_role:
            return dict(self.engine_role)
        if self.spawn_map:
            roles: dict[str, EngineRole] = {
                prefill_id: "prefill" for prefill_id in self.spawn_map
            }
            for decode_id in set(self.spawn_map.values()):
                roles[decode_id] = "decode"
            return roles
        raise ValueError("PDConfig requires engine_role or spawn_map")

    def validate_and_apply(self, engines: dict[str, Engine]) -> None:
        if self.mode != "read":
            raise NotImplementedError(f"PD mode {self.mode!r} is not implemented")

        roles = self.resolved_engine_role(engines)
        hold = True if self.hold_kv_on_complete is None else self.hold_kv_on_complete

        from .roles import ROLES

        for eng_id, eng in engines.items():
            role = roles.get(eng_id)
            if role is None:
                continue
            ROLES.create(role).apply(eng, hold=hold)


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
