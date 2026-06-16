from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from memory import KVBlock, Memory
from request import Request

if TYPE_CHECKING:
    from resource import BandwidthResource, ComputeResource

BlockAction = Literal["compute"] | tuple[Literal["pull"], str]
BlockActions = dict[str, BlockAction]

_LOCAL = Literal["local"]
BlockResolution = BlockAction | _LOCAL | None


@dataclass
class LookupResult:
    evicts: list[KVBlock] = field(default_factory=list)
    blocks: BlockActions = field(default_factory=dict)


class EvictionPolicy(ABC):
    @abstractmethod
    def pick_victims(self, hbm: Memory, count: int, exclude: set[str]) -> list[KVBlock]:
        pass

    def plan(
        self, local: Memory, slots_needed: int, exclude: set[str]
    ) -> list[KVBlock] | None:
        deficit = slots_needed - local.free_size()
        if deficit <= 0:
            return []
        evicts = self.pick_victims(local, deficit, exclude)
        if len(evicts) < deficit:
            return None
        return evicts


class FirstAvailableEviction(EvictionPolicy):
    def pick_victims(self, hbm: Memory, count: int, exclude: set[str]) -> list[KVBlock]:
        victims: list[KVBlock] = []
        for block_hash, copies in hbm.blocks.items():
            if block_hash in exclude:
                continue
            for block in copies:
                if not hbm.can_evict_block(block):
                    continue
                victims.append(block)
                if len(victims) >= count:
                    return victims
        return victims


def local_satisfied(local: Memory, block_hash: str) -> bool:
    return (
        local.best_resident(block_hash) is not None
        or local.inflight_incoming(block_hash) is not None
    )


def slots_needed(actions: BlockActions) -> int:
    return sum(
        1
        for action in actions.values()
        if action == "compute" or (isinstance(action, tuple) and action[0] == "pull")
    )


def recompute_work(
    req: Request | None,
    *,
    block_size: int,
    work_per_prefill_token: float,
    work_per_decode_req: float,
    work_per_block: float,
) -> float:
    """Work units for one recompute decision (matches ``engine.forward_cost`` per block)."""
    if req is None:
        return work_per_block
    if req.is_prefill_chunk():
        return work_per_prefill_token * block_size
    return work_per_decode_req


def first_resident_pull_source(
    memories: dict[str, Memory],
    pull_sources: list[str],
    block_hash: str,
) -> str | None:
    for src_key in pull_sources:
        src = memories[src_key]
        if src.inflight_incoming(block_hash) is not None:
            return None
        if src.best_resident(block_hash) is not None:
            return src_key
    return None


class LookupPolicy(ABC):
    """Resolve per-block actions: local hit, pull from a tier, or recompute."""

    def __init__(
        self,
        local_memory: str,
        eviction_policy: EvictionPolicy | None = None,
    ):
        self.local_memory = local_memory
        self.eviction_policy = eviction_policy or FirstAvailableEviction()

    @property
    @abstractmethod
    def pull_sources(self) -> list[str]:
        pass

    @abstractmethod
    def resolve_block(
        self,
        memories: dict[str, Memory],
        block_hash: str,
        *,
        allow_compute: bool,
        req: Request | None = None,
        block_size: int = 1,
    ) -> BlockResolution:
        pass

    def bind_resources(
        self,
        *,
        compute_res: ComputeResource | None,
        transfer_links: dict[str, BandwidthResource] | None,
        work_per_transfer: float,
        work_per_block: float,
        work_per_prefill_token: float | None = None,
        work_per_decode_req: float | None = None,
        block_size: int = 1,
    ) -> None:
        """Optional hook for Engine to wire runtime resources (cost-based policies)."""

    def begin_allocate_batch(self) -> None:
        """Reset per-batch reservation state before ``schedule()`` allocations."""

    def lookup(
        self, memories: dict[str, Memory], block_hashes: list[str]
    ) -> LookupResult | None:
        actions = self.resolve_actions(memories, block_hashes)
        if actions is None:
            return None
        evicts = self.eviction_policy.plan(
            memories[self.local_memory], slots_needed(actions), set(block_hashes)
        )
        if evicts is None:
            return None
        return LookupResult(evicts=evicts, blocks=actions)

    def resolve_actions(
        self,
        memories: dict[str, Memory],
        block_hashes: list[str],
        *,
        allow_compute: bool = True,
        req: Request | None = None,
        block_size: int = 1,
    ) -> BlockActions | None:
        actions: BlockActions = {}
        for block_hash in block_hashes:
            resolution = self.resolve_block(
                memories,
                block_hash,
                allow_compute=allow_compute,
                req=req,
                block_size=block_size,
            )
            if resolution is None:
                return None
            if resolution == "local":
                continue
            actions[block_hash] = resolution
        return actions


class ComputeOnlyLookupPolicy(LookupPolicy):
    """Local hit or recompute. No remote tiers."""

    @property
    def pull_sources(self) -> list[str]:
        return []

    def resolve_block(
        self,
        memories: dict[str, Memory],
        block_hash: str,
        *,
        allow_compute: bool,
        req: Request | None = None,
        block_size: int = 1,
    ) -> BlockResolution:
        local = memories[self.local_memory]
        if local_satisfied(local, block_hash):
            return "local"
        if allow_compute:
            return "compute"
        return None


class OrderedPullLookupPolicy(LookupPolicy):
    """First ``pull_sources`` entry with a resident copy, else recompute."""

    def __init__(
        self,
        local_memory: str,
        pull_sources: list[str],
        eviction_policy: EvictionPolicy | None = None,
    ):
        super().__init__(local_memory, eviction_policy)
        self._pull_sources = list(pull_sources)

    @property
    def pull_sources(self) -> list[str]:
        return self._pull_sources

    def resolve_block(
        self,
        memories: dict[str, Memory],
        block_hash: str,
        *,
        allow_compute: bool,
        req: Request | None = None,
        block_size: int = 1,
    ) -> BlockResolution:
        local = memories[self.local_memory]
        if local_satisfied(local, block_hash):
            return "local"

        src_key = first_resident_pull_source(memories, self._pull_sources, block_hash)
        if src_key is not None:
            return ("pull", src_key)

        if allow_compute:
            return "compute"
        return None


class CostBasedPullLookupPolicy(LookupPolicy):
    """Pick min-cost pull source vs recompute using per-link bandwidth queues."""

    def __init__(
        self,
        local_memory: str,
        pull_sources: list[str],
        eviction_policy: EvictionPolicy | None = None,
    ):
        super().__init__(local_memory, eviction_policy)
        self._pull_sources = list(pull_sources)
        self._compute_res: ComputeResource | None = None
        self._transfer_links: dict[str, BandwidthResource] = {}
        self._work_per_transfer = 1.0
        self._work_per_block = 1.0
        self._work_per_prefill_token = 1.0
        self._work_per_decode_req = 1.0
        self._block_size = 1
        self._pending_pulls: dict[str, int] = {}
        self._forward_reserved_in_alloc = False
        self._alloc_req: Request | None = None
        self._alloc_block_size = 1

    @property
    def pull_sources(self) -> list[str]:
        return self._pull_sources

    def bind_resources(
        self,
        *,
        compute_res: ComputeResource | None,
        transfer_links: dict[str, BandwidthResource] | None,
        work_per_transfer: float,
        work_per_block: float,
        work_per_prefill_token: float | None = None,
        work_per_decode_req: float | None = None,
        block_size: int = 1,
    ) -> None:
        self._compute_res = compute_res
        self._transfer_links = transfer_links or {}
        self._work_per_transfer = work_per_transfer
        self._work_per_block = work_per_block
        self._work_per_prefill_token = (
            work_per_prefill_token
            if work_per_prefill_token is not None
            else work_per_block
        )
        self._work_per_decode_req = (
            work_per_decode_req if work_per_decode_req is not None else work_per_block
        )
        self._block_size = block_size

    def begin_allocate_batch(self) -> None:
        self._pending_pulls = {}
        self._forward_reserved_in_alloc = False

    def resolve_actions(
        self,
        memories: dict[str, Memory],
        block_hashes: list[str],
        *,
        allow_compute: bool = True,
        req: Request | None = None,
        block_size: int = 1,
    ) -> BlockActions | None:
        self._alloc_req = req
        self._alloc_block_size = block_size
        return super().resolve_actions(
            memories,
            block_hashes,
            allow_compute=allow_compute,
            req=req,
            block_size=block_size,
        )

    def _pull_time(self, link: BandwidthResource, src_key: str) -> float:
        pending = self._pending_pulls.get(src_key, 0)
        return link.time_for(
            self._work_per_transfer,
            link.queued_load() + pending + 1,
        )

    def _compute_time(self) -> float:
        if self._forward_reserved_in_alloc:
            return 0.0
        compute_res = self._compute_res
        assert compute_res is not None
        work = recompute_work(
            self._alloc_req,
            block_size=self._alloc_block_size,
            work_per_prefill_token=self._work_per_prefill_token,
            work_per_decode_req=self._work_per_decode_req,
            work_per_block=self._work_per_block,
        )
        pending_forward = 1
        return compute_res.time_for(
            work,
            compute_res.queued_load() + pending_forward,
        )

    def _note_resolution(self, resolution: BlockResolution) -> None:
        if isinstance(resolution, tuple) and resolution[0] == "pull":
            src_key = resolution[1]
            self._pending_pulls[src_key] = self._pending_pulls.get(src_key, 0) + 1
        elif resolution == "compute":
            self._forward_reserved_in_alloc = True

    def resolve_block(
        self,
        memories: dict[str, Memory],
        block_hash: str,
        *,
        allow_compute: bool,
        req: Request | None = None,
        block_size: int = 1,
    ) -> BlockResolution:
        local = memories[self.local_memory]
        if local_satisfied(local, block_hash):
            return "local"

        if not self._transfer_links or self._compute_res is None:
            src_key = first_resident_pull_source(
                memories, self._pull_sources, block_hash
            )
            if src_key is not None:
                resolution: BlockResolution = ("pull", src_key)
                self._note_resolution(resolution)
                return resolution
            if allow_compute:
                resolution = "compute"
                self._note_resolution(resolution)
                return resolution
            return None

        pull_candidates: list[tuple[float, int, str]] = []

        for order, src_key in enumerate(self._pull_sources):
            src = memories[src_key]
            if src.inflight_incoming(block_hash) is not None:
                return None
            if src.best_resident(block_hash) is None:
                continue

            link = self._transfer_links.get(src_key)
            if link is None:
                continue
            pull_candidates.append(
                (
                    self._pull_time(link, src_key),
                    order,
                    src_key,
                )
            )

        best_pull: tuple[float, int, str] | None = (
            min(pull_candidates, key=lambda item: (item[0], item[1]))
            if pull_candidates
            else None
        )

        if best_pull is None:
            if allow_compute:
                resolution = "compute"
                self._note_resolution(resolution)
                return resolution
            return None

        if not allow_compute:
            resolution = ("pull", best_pull[2])
            self._note_resolution(resolution)
            return resolution

        t_compute = self._compute_time()
        t_pull = best_pull[0]

        if t_pull < t_compute:
            resolution = ("pull", best_pull[2])
        elif t_compute < t_pull:
            resolution = "compute"
        else:
            resolution = ("pull", best_pull[2])
        self._note_resolution(resolution)
        return resolution
