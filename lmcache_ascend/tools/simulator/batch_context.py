from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchContext:
    batch_id: int
    engine_id: str
