"""Generic policy registry: name -> factory with shared construction context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from simulator.model.tier import TierGraph

T = TypeVar("T")


@dataclass(frozen=True)
class PolicyContext:
    graph: TierGraph | None = None
    seed: int = 0
    params: dict = field(default_factory=dict)


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._factories: dict[str, Callable[[PolicyContext], T]] = {}

    def register(self, name: str):
        def deco(fn: Callable[[PolicyContext], T]):
            self._factories[name] = fn
            return fn

        return deco

    def create(self, name: str, ctx: PolicyContext | None = None) -> T:
        if name not in self._factories:
            raise KeyError(
                f"unknown {self.kind}: {name!r} (have {sorted(self._factories)})"
            )
        return self._factories[name](ctx or PolicyContext())

    def names(self) -> list[str]:
        return sorted(self._factories)
