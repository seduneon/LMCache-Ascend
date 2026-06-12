from dataclasses import dataclass, field
from enum import StrEnum


class RequestPD(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"


class RequestStatus(StrEnum):
    PENDING = "pending"
    WAITING = "waiting"
    WAITING_REMOTE_KV = "waiting_remote_kv"
    RUNNING = "running"
    COMPLETE = "complete"


@dataclass
class Request:
    req_id: str
    arrival_time: float
    block_hashes: list[str]
    pd: RequestPD
    status: RequestStatus
    max_output_blocks: int = 0
    num_computed_blocks: int = 0
    prefix_block_count: int = 0
    num_preemptions: int = 0
    pending_block_hash: str | None = field(default=None, repr=False)
    prefill_engine_id: str | None = None
    kv_held_for_transfer: bool = False

    def total_blocks(self) -> int:
        return self.prefix_block_count + self.max_output_blocks

    def blocks_target(self) -> int:
        if self.pd == RequestPD.PREFILL:
            return self.prefix_block_count
        return self.total_blocks()

    def is_prefill_chunk(self) -> bool:
        """vLLM: num_computed_tokens < prompt length (here: prefix blocks)."""
        return self.num_computed_blocks < self.prefix_block_count
