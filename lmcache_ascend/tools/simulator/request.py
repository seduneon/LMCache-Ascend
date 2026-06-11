from dataclasses import dataclass, field
from enum import StrEnum


class RequestPD(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"


class RequestStatus(StrEnum):
    PENDING = "pending"
    WAITING = "waiting"
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

    def total_blocks(self) -> int:
        return self.prefix_block_count + self.max_output_blocks

    def blocks_target(self) -> int:
        if self.pd == RequestPD.PREFILL:
            return self.prefix_block_count
        return self.total_blocks()
