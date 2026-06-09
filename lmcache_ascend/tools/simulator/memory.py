from enum import StrEnum
from tasks import Task

class BlockState(StrEnum):
    EVICTING = "evicting"
    RESIDENT = "resident"
    LOADING = "loading"

class KVBlock:
    def __init__(self, hash: str, state: BlockState):
        self.hash = hash
        self.state = state
        self.task: Task | None = None

class Memory:
    def __init__(self, size: int, name: str):
        self.name = name
        self.size = size
        self.blocks: dict[str, KVBlock] = {}
    
    def state_of(self, block_hash: str) -> BlockState | None:
        b = self.blocks.get(block_hash)
        return b.state if b else None

    def used_size(self) -> int:
        return len(self.blocks)
    
    def free_size(self) -> int:
        return self.size - self.used_size()
    
    def update(self, block_hash: str, state: BlockState, task: Task | None = None):
        self.blocks[block_hash] = KVBlock(block_hash, state, task)
    
    def remove(self, block_hash: str):
        self.blocks.pop(block_hash)
    
    def find(self, block_hash: str) -> KVBlock | None:
        return self.blocks.get(block_hash)
    
    def list(self) -> list[KVBlock]:
        return list(self.blocks.values())