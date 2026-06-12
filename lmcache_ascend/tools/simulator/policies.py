from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from memory import KVBlock, Memory

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


class LookupPolicy:
    def __init__(
        self,
        local_memory: str,
        pull_sources: list[str] | None = None,
        eviction_policy: EvictionPolicy | None = None,
    ):
        self.local_memory = local_memory
        self.pull_sources = pull_sources or []
        self.eviction_policy = eviction_policy or FirstAvailableEviction()
        self._compute_res: ComputeResource | None = None
        self._transfer_links: dict[str, BandwidthResource] = {}
        self._work_per_transfer = 1.0
        self._work_per_block = 1.0
        self._use_cost_model = False

    def bind_cost_model(
        self,
        *,
        compute_res: ComputeResource | None,
        transfer_links: dict[str, BandwidthResource] | None,
        work_per_transfer: float,
        work_per_block: float,
    ) -> None:
        self._compute_res = compute_res
        self._transfer_links = transfer_links or {}
        self._work_per_transfer = work_per_transfer
        self._work_per_block = work_per_block
        self._use_cost_model = bool(self._transfer_links and compute_res is not None)

    def lookup(self, memories: dict[str, Memory], block_hashes: list[str]) -> LookupResult | None:
        local = memories[self.local_memory]
        actions = self.resolve_actions(memories, block_hashes)
        if actions is None:
            return None
        evicts = self.eviction_policy.plan(
            local, self.slots_needed(actions), set(block_hashes)
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
    ) -> BlockActions | None:
        actions: BlockActions = {}
        for block_hash in block_hashes:
            resolution = self._resolve_block(memories, block_hash, allow_compute=allow_compute)
            if resolution is None:
                return None
            if resolution == "local":
                continue
            actions[block_hash] = resolution
        return actions

    def _resolve_block(
        self,
        memories: dict[str, Memory],
        block_hash: str,
        *,
        allow_compute: bool,
    ) -> BlockResolution:
        local = memories[self.local_memory]
        if self.local_satisfied(local, block_hash):
            return "local"

        if not self._use_cost_model:
            return self._resolve_block_legacy(memories, block_hash, allow_compute=allow_compute)

        return self._resolve_block_cost(memories, block_hash, allow_compute=allow_compute)

    def _resolve_block_legacy(
        self,
        memories: dict[str, Memory],
        block_hash: str,
        *,
        allow_compute: bool,
    ) -> BlockResolution:
        for src_key in self.pull_sources:
            src = memories[src_key]
            if src.inflight_incoming(block_hash) is not None:
                return None
            if src.best_resident(block_hash) is not None:
                return ("pull", src_key)

        if allow_compute:
            return "compute"
        return None

    def _resolve_block_cost(
        self,
        memories: dict[str, Memory],
        block_hash: str,
        *,
        allow_compute: bool,
    ) -> BlockResolution:
        pull_candidates: list[tuple[float, int, str]] = []

        for order, src_key in enumerate(self.pull_sources):
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
                    link.share_time(self._work_per_transfer, link.works + 1),
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
                return "compute"
            return None

        if not allow_compute:
            return ("pull", best_pull[2])

        assert self._compute_res is not None
        compute_res = self._compute_res
        t_compute = compute_res.share_time(self._work_per_block, compute_res.works + 1)
        t_pull = best_pull[0]

        if t_pull < t_compute:
            return ("pull", best_pull[2])
        if t_compute < t_pull:
            return "compute"
        return ("pull", best_pull[2])

    def local_satisfied(self, local: Memory, block_hash: str) -> bool:
        return (
            local.best_resident(block_hash) is not None
            or local.inflight_incoming(block_hash) is not None
        )

    def slots_needed(self, actions: BlockActions) -> int:
        return sum(
            1
            for action in actions.values()
            if action == "compute" or (isinstance(action, tuple) and action[0] == "pull")
        )
