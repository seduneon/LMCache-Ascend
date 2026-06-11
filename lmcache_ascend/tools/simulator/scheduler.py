import heapq
from collections import deque
from dataclasses import dataclass, field

from policies import LookupPolicy, LookupResult
from request import Request, RequestPD, RequestStatus


@dataclass
class BatchEntry:
    req: Request
    block_hashes: list[str]
    result: LookupResult


@dataclass
class Batch:
    entries: list[BatchEntry] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)


class Scheduler:
    """vLLM-style batch builder: RUNNING first, then WAITING; preempt only while allocating."""

    def __init__(
        self,
        policy: LookupPolicy,
        memories: dict,
        local_memory: str,
        max_blocks_per_step: int = 10_000,
    ):
        self.policy = policy
        self.memories = memories
        self.local_memory = local_memory
        self.max_blocks_per_step = max_blocks_per_step

        self.pending: list[tuple[float, str, Request]] = []
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.completed: list[Request] = []

    def _local(self):
        return self.memories[self.local_memory]

    def add_request(self, req: Request) -> None:
        req.status = RequestStatus.PENDING
        heapq.heappush(self.pending, (req.arrival_time, req.req_id, req))

    def release_arrivals(self, now: float) -> None:
        while self.pending and self.pending[0][0] <= now:
            _, _, req = heapq.heappop(self.pending)
            req.status = RequestStatus.WAITING
            self.waiting.append(req)

    def next_arrival(self) -> float | None:
        return self.pending[0][0] if self.pending else None

    def schedule(self) -> Batch:
        batch = Batch()
        scheduled_ids: set[str] = set()
        block_budget = self.max_blocks_per_step

        idx = 0
        while idx < len(self.running) and block_budget > 0:
            req = self.running[idx]
            block_hashes = self._blocks_for_running(req)
            if not block_hashes:
                idx += 1
                continue

            result = self._allocate_running(
                req, block_hashes, scheduled_ids, batch.preempted
            )
            if result is None:
                idx += 1
                continue

            batch.entries.append(BatchEntry(req, block_hashes, result))
            scheduled_ids.add(req.req_id)
            block_budget -= self.policy.slots_needed(result.blocks)
            idx += 1

        while self.waiting and block_budget > 0:
            req = self.waiting[0]
            req.prefix_block_count = len(req.block_hashes)
            block_hashes = list(req.block_hashes)

            result = self.policy.lookup(self.memories, block_hashes)
            if result is None:
                break

            slots = self.policy.slots_needed(result.blocks)
            if slots > block_budget:
                break

            self.waiting.popleft()
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            batch.entries.append(BatchEntry(req, block_hashes, result))
            scheduled_ids.add(req.req_id)
            block_budget -= slots

        return batch

    def _blocks_for_running(self, req: Request) -> list[str]:
        if req.pd != RequestPD.DECODE:
            return []
        if req.num_computed_blocks >= req.blocks_target():
            return []
        if req.num_computed_blocks < req.prefix_block_count:
            return []

        block_hash = f"blk:{req.req_id}:{req.num_computed_blocks}"
        req.pending_block_hash = block_hash
        return [block_hash]

    def _allocate_running(
        self,
        req: Request,
        block_hashes: list[str],
        scheduled_ids: set[str],
        preempted: list[Request],
    ) -> LookupResult | None:
        local = self._local()
        actions = {h: "compute" for h in block_hashes}
        exclude = set(block_hashes)

        while True:
            evicts = self.policy.eviction_policy.plan(
                local, self.policy.slots_needed(actions), exclude
            )
            if evicts is not None:
                return LookupResult(evicts=evicts, blocks=actions)

            victim = self._pick_preemption_victim(req, scheduled_ids)
            if victim is None:
                return None

            self._preempt_request(victim)
            preempted.append(victim)
            if victim.req_id == req.req_id:
                return None

    def _pick_preemption_victim(
        self, protected: Request, scheduled_ids: set[str]
    ) -> Request | None:
        others = [
            r
            for r in self.running
            if r.req_id != protected.req_id and r.req_id not in scheduled_ids
        ]
        if others:
            return others[-1]
        if protected.req_id not in scheduled_ids and protected in self.running:
            return protected
        return None

    def _preempt_request(self, req: Request) -> None:
        """Free KV and return to waiting. No task cancel — batches drain before re-schedule."""
        assert req in self.running
        self._local().free_request(req.req_id)

        req.num_computed_blocks = 0
        req.pending_block_hash = None
        req.block_hashes = req.block_hashes[: req.prefix_block_count]
        req.num_preemptions += 1
        req.status = RequestStatus.WAITING

        self.running.remove(req)
        self.waiting.appendleft(req)

    def finish_request(self, req: Request) -> None:
        req.status = RequestStatus.COMPLETE
        self._local().release_request(req.req_id)
        self.running.remove(req)
        self.completed.append(req)
