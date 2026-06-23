from abc import ABC, abstractmethod
from enum import StrEnum


class QueueDiscipline(StrEnum):
    """How queued consumers affect latency estimates (see resource_models)."""

    EVEN_SHARE = "even_share"
    FIFO_TAIL = "fifo_tail"


class Resource(ABC):
    def __init__(
        self,
        latency: float = 0.0,
        *,
        queue_discipline: QueueDiscipline = QueueDiscipline.EVEN_SHARE,
    ):
        self._running = 0
        self._scheduled = 0
        self.latency = latency
        self.queue_discipline = queue_discipline

    @property
    def running(self) -> int:
        return self._running

    @property
    def scheduled(self) -> int:
        return self._scheduled

    def queued_load(self) -> int:
        """Running plus reserved (pool-pending) consumers."""
        return self._running + self._scheduled

    def queue_depth(self) -> int:
        """Pending consumers (scheduled but not yet running)."""
        return self._scheduled

    def queue_position(self, *, include_self: bool = True) -> int:
        """FIFO position at tail: consumers ahead plus optional self."""
        ahead = self._scheduled + self._running
        return ahead + (1 if include_self else 0)

    def estimate_contention_works(self, *, include_self: bool = True) -> int:
        """Effective concurrent consumers for ``time_for`` estimates."""
        if self.queue_discipline == QueueDiscipline.FIFO_TAIL:
            return self.queue_position(include_self=include_self)
        return self.queued_load() + (1 if include_self else 0)

    def schedule(self) -> None:
        """Reserve this resource for a task not yet running."""
        self._scheduled += 1

    def start(self) -> None:
        """Move one reserved slot to running."""
        if self._scheduled <= 0:
            raise RuntimeError("resource.start() called without schedule()")
        self._scheduled -= 1
        self._running += 1

    def finish(self) -> None:
        """Release one running slot."""
        if self._running <= 0:
            raise RuntimeError("resource.finish() called without a running task")
        self._running -= 1

    @abstractmethod
    def speed(self, works: int | None = None) -> float:
        """Effective throughput with ``works`` concurrent consumers (default: ``self.running``)."""
        pass

    def time_for(self, work: float, works: int | None = None) -> float:
        """Wall time to complete ``work`` at ``speed(works)``."""
        rate = self.speed(works)
        if rate <= 0:
            return float("inf")
        return self.latency + work / rate


class LinearShareResource(Resource):
    """Fair-shared linear throughput (compute and bandwidth share the same model)."""

    def __init__(self, base_speed: float, latency: float = 0.0, **kwargs: object):
        del kwargs
        super().__init__(latency=latency)
        self._base_speed = base_speed

    @property
    def base_speed(self) -> float:
        return self._base_speed

    def speed(self, works: int | None = None) -> float:
        n = self._running if works is None else works
        if n <= 0:
            return 0.0
        return self._base_speed / n


ComputeResource = LinearShareResource
BandwidthResource = LinearShareResource
