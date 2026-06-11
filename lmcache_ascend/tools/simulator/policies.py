from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from memory import KVBlock, Memory

BlockAction = Literal["compute"] | tuple[Literal["pull"], str]
BlockActions = dict[str, BlockAction]


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

    def lookup(self, memories: dict[str, Memory], block_hashes: list[str]) -> LookupResult | None:
        local = memories[self.local_memory]
        actions = self._resolve_actions(memories, block_hashes)
        if actions is None:
            return None
        evicts = self.eviction_policy.plan(
            local, self._slots_needed(actions), set(block_hashes)
        )
        if evicts is None:
            return None
        return LookupResult(evicts=evicts, blocks=actions)

    def _resolve_actions(
        self, memories: dict[str, Memory], block_hashes: list[str]
    ) -> BlockActions | None:
        local = memories[self.local_memory]
        actions: BlockActions = {}

        for block_hash in block_hashes:
            if self._local_satisfied(local, block_hash):
                continue

            pulled = False
            for src_key in self.pull_sources:
                src = memories[src_key]
                if src.inflight_incoming(block_hash) is not None:
                    return None
                if src.best_resident(block_hash) is not None:
                    actions[block_hash] = ("pull", src_key)
                    pulled = True
                    break

            if not pulled:
                actions[block_hash] = "compute"

        return actions

    def _local_satisfied(self, local: Memory, block_hash: str) -> bool:
        return (
            local.best_resident(block_hash) is not None
            or local.inflight_incoming(block_hash) is not None
        )

    def _slots_needed(self, actions: BlockActions) -> int:
        return sum(
            1
            for action in actions.values()
            if action == "compute" or (isinstance(action, tuple) and action[0] == "pull")
        )
