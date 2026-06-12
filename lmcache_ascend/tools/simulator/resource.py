from abc import ABC, abstractmethod


class Resource(ABC):
    def __init__(self, latency: float = 0.0):
        self.works = 0
        self.latency = latency

    def add(self):
        self.works += 1

    def remove(self):
        self.works -= 1

    @property
    @abstractmethod
    def speed(self) -> float:
        pass

    @property
    @abstractmethod
    def base_speed(self) -> float:
        pass

    def share_time(self, work: float, sharers: int) -> float:
        """Wall time for ``work`` split evenly among ``sharers`` consumers."""
        if self.base_speed <= 0 or sharers <= 0:
            return float("inf")
        return self.latency + work * sharers / self.base_speed


class ComputeResource(Resource):
    def __init__(self, base_speed: float, latency: float = 0.0):
        super().__init__(latency=latency)
        self._base_speed = base_speed

    @property
    def base_speed(self) -> float:
        return self._base_speed

    @property
    def speed(self) -> float:
        return 0.0 if self.works == 0 else self._base_speed / self.works


class BandwidthResource(Resource):
    def __init__(self, base_speed: float, latency: float = 0.0):
        super().__init__(latency=latency)
        self._base_speed = base_speed

    @property
    def base_speed(self) -> float:
        return self._base_speed

    @property
    def speed(self) -> float:
        return 0.0 if self.works == 0 else self._base_speed / self.works
