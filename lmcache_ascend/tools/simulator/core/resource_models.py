"""Optional bandwidth/compute resource models (registry-backed)."""

from __future__ import annotations

from simulator.policy.registry import PolicyContext, Registry

from .resource import BandwidthResource, LinearShareResource, QueueDiscipline, Resource

RESOURCE = Registry["Resource"]("resource")


class SaturatingResource(LinearShareResource):
    """Throughput rises with concurrency up to ``max_concurrency``, then saturates."""

    def __init__(
        self,
        base_speed: float,
        *,
        max_concurrency: int = 4,
        latency: float = 0.0,
        queue_discipline: QueueDiscipline = QueueDiscipline.EVEN_SHARE,
    ):
        super().__init__(base_speed, latency=latency)
        self.max_concurrency = max(1, max_concurrency)
        self.queue_discipline = queue_discipline

    def speed(self, works: int | None = None) -> float:
        n = self._running if works is None else works
        if n <= 0:
            return 0.0
        effective = min(n, self.max_concurrency)
        return self._base_speed / effective

    def estimate_contention_works(self, *, include_self: bool = True) -> int:
        base = self.queued_load() + (1 if include_self else 0)
        if self.queue_discipline == QueueDiscipline.FIFO_TAIL:
            return max(1, base)
        return max(1, base)


class FixedLatencyOpResource(LinearShareResource):
    """Per-operation fixed latency plus bandwidth term (SSD IOPS / RDMA setup)."""

    def __init__(
        self,
        base_speed: float,
        *,
        op_latency: float = 0.0,
        read_speed: float | None = None,
        write_speed: float | None = None,
        latency: float = 0.0,
        queue_discipline: QueueDiscipline = QueueDiscipline.EVEN_SHARE,
    ):
        super().__init__(base_speed, latency=latency)
        self.op_latency = op_latency
        self.read_speed = read_speed if read_speed is not None else base_speed
        self.write_speed = write_speed if write_speed is not None else base_speed
        self.queue_discipline = queue_discipline

    def speed(self, works: int | None = None) -> float:
        n = self._running if works is None else works
        if n <= 0:
            return 0.0
        return self._base_speed / n

    def time_for(self, work: float, works: int | None = None) -> float:
        queued = self.estimate_contention_works(
            include_self=works is not None and works > self.queued_load()
        )
        if self.queue_discipline == QueueDiscipline.FIFO_TAIL:
            return self.latency + self.op_latency * queued + work / max(
                self._base_speed, 1e-12
            )
        rate = self.speed(works if works is not None else queued)
        if rate <= 0:
            return float("inf")
        return self.latency + self.op_latency + work / rate

    def estimate_contention_works(self, *, include_self: bool = True) -> int:
        base = self.queued_load() + (1 if include_self else 0)
        return max(1, base)


def _linear(ctx: PolicyContext) -> Resource:
    params = ctx.params
    return LinearShareResource(
        float(params.get("base_speed", 32.0)),
        latency=float(params.get("latency", 0.0)),
    )


def _saturating(ctx: PolicyContext) -> Resource:
    params = ctx.params
    discipline = params.get("queue_discipline", QueueDiscipline.EVEN_SHARE)
    if isinstance(discipline, str):
        discipline = QueueDiscipline(discipline)
    return SaturatingResource(
        float(params.get("base_speed", 32.0)),
        max_concurrency=int(params.get("max_concurrency", 4)),
        latency=float(params.get("latency", 0.01)),
        queue_discipline=discipline,
    )


def _fixed_latency_op(ctx: PolicyContext) -> Resource:
    params = ctx.params
    discipline = params.get("queue_discipline", QueueDiscipline.EVEN_SHARE)
    if isinstance(discipline, str):
        discipline = QueueDiscipline(discipline)
    base = float(params.get("base_speed", 8.0))
    return FixedLatencyOpResource(
        base,
        op_latency=float(params.get("op_latency", 0.001)),
        read_speed=params.get("read_speed"),
        write_speed=params.get("write_speed"),
        latency=float(params.get("latency", 0.0)),
        queue_discipline=discipline,
    )


RESOURCE.register("linear")(_linear)
RESOURCE.register("saturating")(_saturating)
RESOURCE.register("fixed_latency_op")(_fixed_latency_op)


def medium_resource_kind(tier_key: str) -> str:
    """Default resource model kind by tier suffix."""
    if tier_key.endswith(":ssd"):
        return "fixed_latency_op"
    if tier_key.endswith(":dram"):
        return "saturating"
    return "linear"


def build_bandwidth_resource(
    kind: str,
    *,
    base_speed: float,
    latency: float = 0.01,
    **params: object,
) -> BandwidthResource:
    merged = {"base_speed": base_speed, "latency": latency, **params}
    resource = RESOURCE.create(kind, PolicyContext(params=merged))
    assert isinstance(resource, LinearShareResource)
    return resource
