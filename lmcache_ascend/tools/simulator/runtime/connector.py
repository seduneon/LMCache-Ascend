"""Cache subsystem facade: lifecycle hooks and cost context for the engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from simulator.observability.estimate import CostContext
from simulator.policy.effects import EffectPolicy
from simulator.policy.config import LifecycleSpec
from simulator.core.memory import KVBlock, Memory, collect_content_copies
from .plan import BatchPlan, WorkEntry
from simulator.policy.policies import EnginePolicies, enrich_entry_plan
from simulator.core.request import Request, request_owning_prefix_block
from simulator.core.resource import BandwidthResource, ComputeResource
from simulator.policy.schedule import SchedulePolicy
from simulator.model.tier import TierGraph

if TYPE_CHECKING:
    from simulator.observability.event_trace import EventTraceWriter


class TierCacheConnector:
    """Tier-centric cache boundary (vLLM connector-shaped facade)."""

    def __init__(
        self,
        *,
        engine_id: str,
        policies: EnginePolicies,
        memories: dict[str, Memory],
        graph: TierGraph,
        lifecycle: LifecycleSpec,
        local_tier: str,
        compute_res: ComputeResource,
        transfer_links: dict[str, BandwidthResource],
        write_links: dict[str, BandwidthResource],
        work_per_transfer: float,
        work_per_prefill_token: float,
        work_per_decode_req: float,
        interconnect: BandwidthResource | None = None,
        event_trace: EventTraceWriter | None = None,
    ):
        self.engine_id = engine_id
        self.event_trace = event_trace
        self.policies = policies
        self.memories = memories
        self.graph = graph
        self.lifecycle = lifecycle
        self.local_tier = local_tier
        self.compute_res = compute_res
        self.transfer_links = transfer_links
        self.write_links = write_links
        self.work_per_transfer = work_per_transfer
        self.work_per_prefill_token = work_per_prefill_token
        self.work_per_decode_req = work_per_decode_req
        self.interconnect = interconnect
        self._peak_duplicate_count = 0

    @property
    def schedule(self) -> SchedulePolicy:
        return self.policies.schedule

    @property
    def effects(self) -> EffectPolicy:
        return self.policies.effects

    def build_cost_context(self, now: float) -> CostContext:
        return CostContext(
            memories=self.memories,
            transfer_links=self.transfer_links,
            compute_res=self.compute_res,
            work_per_transfer=self.work_per_transfer,
            work_per_prefill_token=self.work_per_prefill_token,
            work_per_decode_req=self.work_per_decode_req,
            now=now,
            interconnect=self.interconnect,
        )

    def enrich_plan(
        self,
        scheduled,
        *,
        known_requests: list[Request],
    ) -> None:
        for entry in scheduled.entries:
            enrich_entry_plan(
                entry.plan,
                entry=entry,
                effects=self.effects,
                memories=self.memories,
                known_requests=known_requests,
            )

    def on_local_resident(self, block: KVBlock, req: Request, now: float) -> None:
        self.effects.on_local_resident(self.memories, block, req, now)
        self._track_duplicates(req, block.hash)

    def on_tier_resident(
        self, tier_key: str, block: KVBlock, req: Request, now: float
    ) -> None:
        self.effects.on_tier_resident(self.memories, tier_key, block, req, now)
        self._track_duplicates(req, block.hash)

    def after_pull(self, src_key: str, block_hash: str, req: Request) -> None:
        self.effects.after_pull(self.memories, src_key, block_hash, req)

    def on_local_evict(self, victim: KVBlock, spill_req: Request, now: float) -> None:
        if self.event_trace is not None:
            self.event_trace.on_evict(
                now=now,
                engine_id=self.engine_id,
                tier=self.local_tier,
                block_hash=victim.hash,
                policy="local",
            )
        self.effects.on_local_evict(self.memories, victim, spill_req, now)

    def release_request_kv(
        self,
        req: Request,
        *,
        preempted: bool,
        now: float,
    ) -> None:
        stored = True
        if not preempted and self.lifecycle.store_on_complete:
            stored = self.effects.store_prefix_on_complete(
                self.memories,
                req=req,
                now=now,
                tier_keys=self.lifecycle.store_on_complete,
            )
        retain_hashes: set[str] | None = None
        prefix_hashes = set(req.block_hashes[: req.prefix_block_count])
        if self.lifecycle.retain_prefix_cache and not preempted:
            retain_hashes = prefix_hashes
        elif (
            not preempted
            and self.lifecycle.store_on_complete
            and not stored
        ):
            retain_hashes = prefix_hashes
        self.memories[self.local_tier].free_request(
            req.req_id,
            retain_hashes=retain_hashes,
        )

    def resolve_spill_req(
        self, block_hash: str, known_requests: list[Request]
    ) -> Request | None:
        return request_owning_prefix_block(block_hash, known_requests)

    def _track_duplicates(self, req: Request | None, block_hash: str) -> None:
        from simulator.core.kv_content import ContentKey

        content = ContentKey.from_slot(block_hash)
        count = len(
            collect_content_copies(
                self.memories,
                list(self.memories.keys()),
                content,
                req=req,
            )
        )
        self._peak_duplicate_count = max(self._peak_duplicate_count, count)

    @property
    def peak_duplicate_count(self) -> int:
        return self._peak_duplicate_count
