from abc import ABC, abstractmethod


class Resource(ABC):
    def __init__(self):
        self.works = 0

    def add(self):
        self.works += 1

    def remove(self):
        self.works -= 1

    @property
    @abstractmethod
    def speed(self) -> float:
        pass

    def time_takes(self, value: float) -> float:
        return value / self.speed


class ComputeResource(Resource):
    def __init__(self, base_speed: float):
        super().__init__()
        self.base_speed = base_speed

    @property
    def speed(self) -> float:
        return 0.0 if self.works == 0 else self.base_speed / self.works
