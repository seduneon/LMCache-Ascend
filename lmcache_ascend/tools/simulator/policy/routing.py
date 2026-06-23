"""Routing policies: map finished prefill requests to decode engines (xPyD proxy)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from simulator.core.request import Request
from .registry import PolicyContext, Registry

ROUTING = Registry["RoutingPolicy"]("routing")


class RoutingPolicy(ABC):
    @abstractmethod
    def route(self, req: Request, prefill_id: str) -> str:
        """Choose decode engine for a finished prefill request."""


class BijectionRouting(RoutingPolicy):
    def __init__(self, mapping: dict[str, str]):
        self._mapping = dict(mapping)

    def route(self, req: Request, prefill_id: str) -> str:
        del req
        if prefill_id not in self._mapping:
            raise KeyError(f"no decode route for prefill {prefill_id!r}")
        return self._mapping[prefill_id]


class RoundRobinRouting(RoutingPolicy):
    def __init__(self, decode_ids: tuple[str, ...]):
        if not decode_ids:
            raise ValueError("decode_ids must be non-empty")
        self._decode_ids = decode_ids
        self._cursor = 0

    def route(self, req: Request, prefill_id: str) -> str:
        del req, prefill_id
        decode_id = self._decode_ids[self._cursor % len(self._decode_ids)]
        self._cursor += 1
        return decode_id


class HashRouting(RoutingPolicy):
    def __init__(self, decode_ids: tuple[str, ...]):
        if not decode_ids:
            raise ValueError("decode_ids must be non-empty")
        self._decode_ids = decode_ids

    def route(self, req: Request, prefill_id: str) -> str:
        del prefill_id
        key = getattr(req, "conversation_id", None) or req.req_id
        idx = hash(key) % len(self._decode_ids)
        return self._decode_ids[idx]


@ROUTING.register("bijection")
def _bijection(ctx: PolicyContext) -> RoutingPolicy:
    mapping = ctx.params.get("map") or ctx.params.get("spawn_map") or {}
    return BijectionRouting(dict(mapping))


@ROUTING.register("round_robin")
def _round_robin(ctx: PolicyContext) -> RoutingPolicy:
    decode_ids = tuple(ctx.params.get("decode_ids", ()))
    return RoundRobinRouting(decode_ids)


@ROUTING.register("hash")
def _hash(ctx: PolicyContext) -> RoutingPolicy:
    decode_ids = tuple(ctx.params.get("decode_ids", ()))
    return HashRouting(decode_ids)
