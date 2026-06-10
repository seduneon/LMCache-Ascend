from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from memory import BlockState, Memory


@dataclass
class LookupResult:
    evicts: list[str] = field(default_factory=list)
    blocks: dict[str, Literal["compute"] | tuple[Literal["pull"], str]] = field(
        default_factory=dict
    )


class LookupPolicy(ABC):
    @abstractmethod
    def lookup(self, memories: dict[str, Memory], block_hashes: list[str]) -> LookupResult | None:
        pass


def _pick_victims(hbm: Memory, count: int, exclude: set[str]) -> list[str]:
    victims: list[str] = []
    for block_hash, block in hbm.blocks.items():
        if block_hash in exclude:
            continue
        if not hbm.can_evict(block_hash):
            continue
        victims.append(block_hash)
        if len(victims) >= count:
            break
    return victims


class ComputeAllLookup(LookupPolicy):
    def lookup(self, memories: dict[str, Memory], block_hashes: list[str]) -> LookupResult | None:
        hbm = memories["hbm"]
        blocks: dict[str, Literal["compute"]] = {}
        exclude = set(block_hashes)

        for block_hash in block_hashes:
            state = hbm.state_of(block_hash)
            if state == BlockState.RESIDENT:
                continue
            if state == BlockState.EVICTING:
                return None
            if state in (BlockState.LOADING, BlockState.RESERVED):
                continue
            blocks[block_hash] = "compute"

        needed = len(blocks)
        deficit = needed - hbm.free_size()
        if deficit < 0:
            deficit = 0

        evicts: list[str] = []
        if deficit > 0:
            evicts = _pick_victims(hbm, deficit, exclude)
            if len(evicts) < deficit:
                return None

        return LookupResult(evicts=evicts, blocks=blocks)
