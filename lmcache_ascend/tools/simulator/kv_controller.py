"""KV planning: lookup + placement + retention at schedule time (authoritative plan)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .chunk_hash import chunk_key_for_hbm_block
from .memory import BlockState, KVBlock
from .plan import EntryPlan, SimContext, StoreOp
from .policies import LookupPolicy, PlacementPolicy, RetentionPolicy
from .task_outcomes import TaskOutcome

if TYPE_CHECKING:
    from .memory import Memory
    from .request import Request


class KVController:
    """Schedule-time entry point: emits a fully-specified ``EntryPlan`` per request."""

    def __init__(
        self,
        lookup: LookupPolicy,
        *,
        local_memory: str,
        placement: PlacementPolicy | None = None,
        retention: RetentionPolicy | None = None,
    ):
        self.lookup = lookup
        self.local_memory = local_memory
        self.placement = placement
        self.retention = retention
        self.policy = lookup

    def begin_batch(self) -> None:
        self.lookup.begin_allocate_batch()

    def plan_blocks(
        self,
        memories: dict[str, Memory],
        block_hashes: list[str],
        *,
        req: Request | None = None,
        allow_compute: bool = True,
        block_size: int = 1,
        ctx: SimContext | None = None,
        known_requests: list[Request] | None = None,
    ) -> EntryPlan | None:
        result = self.lookup.lookup(
            memories,
            block_hashes,
            allow_compute=allow_compute,
            req=req,
            block_size=block_size,
            ctx=ctx,
        )
        if result is None:
            return None

        plan = result
        self._plan_hbm_effects(memories, block_hashes, plan, req=req)
        self._plan_spills(plan, known_requests or [])
        return plan

    def _plan_hbm_effects(
        self,
        memories: dict[str, Memory],
        block_hashes: list[str],
        plan: EntryPlan,
        *,
        req: Request | None,
    ) -> None:
        if req is None:
            return
        for block_hash in block_hashes:
            action = plan.blocks.get(block_hash)
            if action not in ("compute",) and not (
                isinstance(action, tuple) and action[0] == "pull"
            ):
                continue
            plan.resident_outcomes[block_hash] = TaskOutcome(
                kind="hbm_resident",
                req_id=req.req_id,
                block_hash=block_hash,
            )
            if isinstance(action, tuple) and action[0] == "pull":
                plan.pull_outcomes[block_hash] = TaskOutcome(
                    kind="pull_complete",
                    req_id=req.req_id,
                    block_hash=block_hash,
                    src_key=action[1],
                )
            if self.placement is None:
                continue
            plan.hbm_mirror_tiers[block_hash] = self.placement.mirror_tier_keys()

    @staticmethod
    def _req_for_block(block_hash: str, known_requests: list[Request]) -> Request | None:
        for req in known_requests:
            if block_hash in req.block_hashes[: req.prefix_block_count]:
                return req
        return None

    def _plan_spills(
        self,
        plan: EntryPlan,
        known_requests: list[Request],
    ) -> None:
        if self.placement is None:
            return
        for victim in plan.evicts:
            spill_req = self._req_for_block(victim.hash, known_requests)
            plan.spill_reqs[id(victim)] = spill_req
            if spill_req is None:
                continue
