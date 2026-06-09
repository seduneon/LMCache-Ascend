from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from request import Request
from memory import Memory

@dataclass
class LookupResult():
    evicts: list[str] = field(default_factory=list)
    blocks: dict[str, Literal["compute"] | tuple[Literal["pull"], str]] = field(default_factory=dict)

class LookupPolicy(ABC):
    @abstractmethod
    def lookup(self, memories: dict[str, Memory], request: Request) -> LookupResult | None:
        pass