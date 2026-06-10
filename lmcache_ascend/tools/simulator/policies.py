from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from memory import KVBlock, Memory


@dataclass
class LookupResult:
    evicts: list[KVBlock] = field(default_factory=list)
    blocks: dict[str, Literal["compute"] | tuple[Literal["pull"], str]] = field(
        default_factory=dict
    )


class LookupPolicy(ABC):
    @abstractmethod
    def lookup(self, memories: dict[str, Memory], block_hashes: list[str]) -> LookupResult | None:
        pass


def _pick_victims(hbm: Memory, count: int, exclude: set[str]) -> list[KVBlock]:
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


class ComputeAllLookup(LookupPolicy):
    def lookup(self, memories: dict[str, Memory], block_hashes: list[str]) -> LookupResult | None:
        hbm = memories["hbm"]
        blocks: dict[str, Literal["compute"]] = {}
        exclude = set(block_hashes)

        for block_hash in block_hashes:
            if hbm.best_resident(block_hash) is not None:
                continue
            if hbm.inflight_incoming(block_hash) is not None:
                continue
            blocks[block_hash] = "compute"

        needed = len(blocks)
        deficit = needed - hbm.free_size()
        if deficit < 0:
            deficit = 0

        evicts: list[KVBlock] = []
        if deficit > 0:
            evicts = _pick_victims(hbm, deficit, exclude)
            if len(evicts) < deficit:
                return None

        return LookupResult(evicts=evicts, blocks=blocks)
