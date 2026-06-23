"""Engine role profiles: prefill vs decode behavior for PD simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from simulator.core.request import Request
from simulator.policy.registry import PolicyContext, Registry
from simulator.policy.routing import RoutingPolicy

from .pd import DecodeSpawn, KvRelease, SimEvent

if TYPE_CHECKING:
    from simulator.runtime.engine import Engine

ROLES = Registry["EngineRoleProfile"]("engine_role")


class EngineRoleProfile(ABC):
    """Strategy for engine-role-specific PD wiring and commit dispatch."""

    pull_mode: Literal["compute_only", "ordered_pull"]
    remote_kv_wait: bool
    wants_pull_sources: bool

    @abstractmethod
    def apply(self, eng: Engine, *, hold: bool) -> None:
        """Apply PD read-mode settings to an engine instance."""

    @abstractmethod
    def events_on_commit(
        self,
        req: Request,
        eng_id: str,
        *,
        routing: RoutingPolicy | None,
        now: float,
    ) -> list[SimEvent]:
        """Cross-engine events emitted when a request finishes on this engine."""


class PrefillProfile(EngineRoleProfile):
    pull_mode = "compute_only"
    remote_kv_wait = False
    wants_pull_sources = False

    def apply(self, eng: Engine, *, hold: bool) -> None:
        eng.hold_kv_on_complete = hold

    def events_on_commit(
        self,
        req: Request,
        eng_id: str,
        *,
        routing: RoutingPolicy | None,
        now: float,
    ) -> list[SimEvent]:
        if routing is None or not req.is_prefill():
            return []
        decode_id = routing.route(req, eng_id)
        return [
            DecodeSpawn(
                req_id=req.req_id,
                arrival_time=now,
                prefix_blocks=tuple(req.block_hashes[: req.prefix_block_count]),
                max_output_blocks=req.max_output_blocks,
                prefix_block_count=req.prefix_block_count,
                prefill_engine_id=eng_id,
                decode_engine_id=decode_id,
            )
        ]


class DecodeProfile(EngineRoleProfile):
    pull_mode = "ordered_pull"
    remote_kv_wait = True
    wants_pull_sources = True

    def apply(self, eng: Engine, *, hold: bool) -> None:
        del hold
        if not eng.policies.schedule.pull_sources:
            raise ValueError(
                f"decode engine {eng.engine_id!r} needs pull_sources for PD read mode"
            )
        eng.remote_kv_wait = True

    def events_on_commit(
        self,
        req: Request,
        eng_id: str,
        *,
        routing: RoutingPolicy | None,
        now: float,
    ) -> list[SimEvent]:
        del eng_id, routing, now
        if not req.is_decode() or req.prefill_engine_id is None:
            return []
        return [
            KvRelease(req_id=req.req_id, prefill_engine_id=req.prefill_engine_id)
        ]


@ROLES.register("prefill")
def _prefill(_ctx: PolicyContext) -> EngineRoleProfile:
    return PrefillProfile()


@ROLES.register("decode")
def _decode(_ctx: PolicyContext) -> EngineRoleProfile:
    return DecodeProfile()
