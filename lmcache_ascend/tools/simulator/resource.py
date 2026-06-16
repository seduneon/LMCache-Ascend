from abc import ABC, abstractmethod


class Resource(ABC):
    def __init__(self, latency: float = 0.0):
        self._running = 0
        self._scheduled = 0
        self.latency = latency

    @property
    def running(self) -> int:
        return self._running

    @property
    def scheduled(self) -> int:
        return self._scheduled

    @property
    def works(self) -> int:
        """Running tasks (alias kept for ``speed()`` and existing tests)."""
        return self._running

    def queued_load(self) -> int:
        """Running plus reserved (pool-pending) consumers."""
        return self._running + self._scheduled

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


class ComputeResource(Resource):
    def __init__(self, base_speed: float, latency: float = 0.0):
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


class BandwidthResource(Resource):
    def __init__(self, base_speed: float, latency: float = 0.0):
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
