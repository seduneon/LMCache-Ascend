from dataclasses import dataclass
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
    tokens: int