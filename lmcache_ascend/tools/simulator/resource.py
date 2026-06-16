from abc import ABC, abstractmethod


class Resource(ABC):
    def __init__(self, latency: float = 0.0):
        self.works = 0
        self.latency = latency

    def add(self):
        self.works += 1

    def remove(self):
        self.works -= 1

    @abstractmethod
    def speed(self, works: int | None = None) -> float:
        """Effective throughput with ``works`` concurrent consumers (default: ``self.works``)."""
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
        n = self.works if works is None else works
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
        n = self.works if works is None else works
        if n <= 0:
            return 0.0
        return self._base_speed / n
