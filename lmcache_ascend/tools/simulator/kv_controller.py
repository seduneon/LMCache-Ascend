"""KV planning facade: schedule-time decisions in one place."""

from __future__ import annotations

from typing import TYPE_CHECKING

from plan import EntryPlan, SimContext
from policies import LookupPolicy

if TYPE_CHECKING:
    from memory import Memory
    from request import Request


class KVController:
    """Wraps ``LookupPolicy``; future home for joint placement/eviction/lookup planning."""

    def __init__(self, policy: LookupPolicy):
        self.policy = policy

    def begin_batch(self) -> None:
        self.policy.begin_allocate_batch()

    def plan_blocks(
        self,
        memories: dict[str, Memory],
        block_hashes: list[str],
        *,
        req: Request | None = None,
        allow_compute: bool = True,
        block_size: int = 1,
        ctx: SimContext | None = None,
    ) -> EntryPlan | None:
        del ctx  # wired at capture time; cost policies read live resources today
        result = self.policy.lookup(
            memories,
            block_hashes,
            allow_compute=allow_compute,
            req=req,
            block_size=block_size,
        )
        if result is None:
            return None
        return EntryPlan(evicts=list(result.evicts), blocks=dict(result.blocks))
