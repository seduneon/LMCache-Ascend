from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from memory import BlockState, Memory

@dataclass
class LookupResult():
    evicts: list[str] = field(default_factory=list)
    blocks: dict[str, Literal["compute"] | tuple[Literal["pull"], str]] = field(default_factory=dict)

class LookupPolicy(ABC):
    @abstractmethod
    def lookup(self, memories: dict[str, Memory], block_hashes: list[str]) -> LookupResult | None:
        pass

class ComputeAllLookup(LookupPolicy):
    def lookup(self, memories: dict[str, Memory], block_hashes: list[str]) -> LookupResult | None:
        hbm = memories["hbm"]
        blocks: dict[str, Literal["compute"]] = {}

        for block_hash in block_hashes:
            state = hbm.state_of(block_hash)
            if state == BlockState.RESIDENT:
                continue
            if state in (BlockState.LOADING, BlockState.EVICTING):
                return None
            blocks[block_hash] = "compute"

        if hbm.free_size() < len(blocks):
            return None

        return LookupResult(evicts=[], blocks=blocks)

