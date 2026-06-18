from __future__ import annotations

import heapq
from collections import deque

from .kv_controller import KVController
from .memory import Memory
from .plan import EntryPlan, ScheduleResult, SimContext, WorkEntry
from .lookup import LookupPolicy, local_satisfied
from .request import Request, RequestPD, RequestStatus


class Scheduler:
    """vLLM-style batch builder: RUNNING first, then WAITING; preempt only while allocating."""

    def __init__(
        self,
        controller_or_policy: KVController | LookupPolicy,
        memories: dict[str, Memory],
        local_memory: str,
        *,
        max_num_seqs: int = 10_000,
        max_num_batched_tokens: int = 10_000,
        block_size: int = 1,
        enable_chunked_prefill: bool = False,
        remote_kv_wait: bool = False,
    ):
        if isinstance(controller_or_policy, KVController):
            self.controller = controller_or_policy
        else:
            self.controller = KVController(controller_or_policy)
        self.policy = self.controller.policy
        self.memories = memories
        self.local_memory = local_memory
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.block_size = block_size
        self.enable_chunked_prefill = enable_chunked_prefill
        self.remote_kv_wait = remote_kv_wait

        self.pending: list[tuple[float, str, Request]] = []
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.completed: list[Request] = []

    def _local(self) -> Memory:
        return self.memories[self.local_memory]

    def add_request(self, req: Request) -> None:
        req.status = RequestStatus.PENDING
        heapq.heappush(self.pending, (req.arrival_time, req.req_id, req))

    def _ensure_prefix_block_count(self, req: Request) -> None:
        if req.prefix_block_count == 0 and req.block_hashes:
            req.prefix_block_count = len(req.block_hashes)

    def release_arrivals(self, now: float) -> None:
        while self.pending and self.pending[0][0] <= now:
            _, _, req = heapq.heappop(self.pending)
            req.status = RequestStatus.WAITING
            self._ensure_prefix_block_count(req)
            req.metrics.enter_waiting(now)
            self.waiting.append(req)

    def next_arrival(self) -> float | None:
        return self.pending[0][0] if self.pending else None

    def _remaining_prefill_tokens(self, req: Request) -> int:
        return (req.prefix_block_count - req.num_computed_blocks) * self.block_size

    def _blocks_for_prefill_chunk(self, req: Request, num_new_tokens: int) -> list[str]:
        start = req.num_computed_blocks
        if start >= req.prefix_block_count:
            return []
        num_blocks = (num_new_tokens + self.block_size - 1) // self.block_size
        num_blocks = min(num_blocks, req.prefix_block_count - start)
        return list(req.block_hashes[start : start + num_blocks])

    def _num_new_tokens_running(self, req: Request, token_budget: int) -> int:
        if req.num_computed_blocks >= req.blocks_target():
            return 0
        if req.is_prefill_chunk():
            remaining = self._remaining_prefill_tokens(req)
            if remaining <= 0:
                return 0
            return min(remaining, token_budget)
        if req.pd != RequestPD.DECODE:
            return 0
        remaining = (req.blocks_target() - req.num_computed_blocks) * self.block_size
        if remaining <= 0:
            return 0
        return min(self.block_size, remaining, token_budget)

    def _waiting_prefix_tokens(self, req: Request) -> int:
        return self._remaining_prefill_tokens(req)

    @staticmethod
    def _entry_scheduled_tokens(entry: WorkEntry) -> int:
        if entry.remote_kv:
            return 0
        if not any(action == "compute" for action in entry.plan.blocks.values()):
            return 0
        return entry.num_scheduled_tokens

    def _active_request_count(self) -> int:
        remote_kv = sum(
            1 for req in self.waiting if req.status == RequestStatus.WAITING_REMOTE_KV
        )
        return len(self.running) + remote_kv

    def _prefix_block_hashes(self, req: Request) -> list[str]:
        return list(req.block_hashes[: req.prefix_block_count])

    def _needs_remote_kv(self, req: Request) -> bool:
        if not self.remote_kv_wait:
            return False
        if req.pd != RequestPD.DECODE:
            return False
        if req.status == RequestStatus.WAITING_REMOTE_KV:
            return False
        if req.num_computed_blocks >= req.prefix_block_count:
            return False
        if not self.policy.pull_sources:
            return False
        local = self._local()
        for block_hash in self._prefix_block_hashes(req):
            if local_satisfied(local, block_hash):
                continue
            return True
        return False

    def _try_admit_remote_kv(
        self,
        req: Request,
        scheduled: ScheduleResult,
        scheduled_ids: set[str],
        *,
        now: float | None = None,
        engine_id: str | None = None,
        ctx: SimContext | None = None,
    ) -> bool:
        block_hashes = self._prefix_block_hashes(req)
        if not block_hashes:
            return False

        plan = self._allocate_blocks(
            req,
            block_hashes,
            scheduled_ids,
            scheduled.preempted,
            pull_only=True,
            now=now,
            ctx=ctx,
        )
        if plan is None:
            return False

        entry = WorkEntry(
            req,
            block_hashes,
            plan,
            num_scheduled_tokens=0,
            remote_kv=True,
        )
        req.status = RequestStatus.WAITING_REMOTE_KV
        if now is not None:
            req.metrics.enter_remote_kv(now, engine_id=engine_id)
        scheduled.entries.append(entry)
        scheduled_ids.add(req.req_id)
        return True

    def promote_remote_kv_complete(
        self, req: Request, *, now: float | None = None, engine_id: str | None = None
    ) -> None:
        assert req.status == RequestStatus.WAITING_REMOTE_KV
        self.waiting.remove(req)
        req.status = RequestStatus.RUNNING
        req.num_computed_blocks = req.prefix_block_count
        if now is not None:
            req.metrics.enter_running(now, engine_id=engine_id)
        self.running.append(req)

    def finish_prefill_held(self, req: Request, *, now: float | None = None) -> None:
        """Prefill compute done; keep KV resident until decode acknowledges transfer."""
        if req not in self.running:
            return
        assert req.pd == RequestPD.PREFILL
        req.status = RequestStatus.COMPLETE
        req.kv_held_for_transfer = True
        if now is not None:
            req.metrics.finish(now)
        self.running.remove(req)
        self.completed.append(req)

    def schedule(
        self,
        now: float | None = None,
        *,
        engine_id: str | None = None,
        ctx: SimContext | None = None,
    ) -> ScheduleResult:
        self.controller.begin_batch()
        scheduled = ScheduleResult()
        scheduled_ids: set[str] = set()
        token_budget = self.max_num_batched_tokens

        idx = 0
        while idx < len(self.running) and token_budget > 0:
            req = self.running[idx]
            num_new_tokens = self._num_new_tokens_running(req, token_budget)
            if num_new_tokens == 0:
                idx += 1
                continue

            block_hashes = self._blocks_for_running(req, num_new_tokens)
            if not block_hashes:
                idx += 1
                continue

            plan = self._allocate_blocks(
                req, block_hashes, scheduled_ids, scheduled.preempted, now=now, ctx=ctx
            )
            if plan is None:
                idx += 1
                continue

            entry = WorkEntry(req, block_hashes, plan, num_new_tokens)
            scheduled.entries.append(entry)
            scheduled_ids.add(req.req_id)
            tokens = self._entry_scheduled_tokens(entry)
            token_budget -= tokens
            scheduled.total_num_scheduled_tokens += tokens
            idx += 1

        if not scheduled.preempted:
            remote_kv_rotations = 0
            max_rotations = len(self.waiting)

            while self.waiting:
                if self._active_request_count() >= self.max_num_seqs:
                    break

                req = self.waiting[0]
                if req.status == RequestStatus.WAITING_REMOTE_KV:
                    if remote_kv_rotations >= max_rotations:
                        break
                    self.waiting.popleft()
                    self.waiting.append(req)
                    remote_kv_rotations += 1
                    continue

                if self._needs_remote_kv(req):
                    if self._try_admit_remote_kv(
                        req,
                        scheduled,
                        scheduled_ids,
                        now=now,
                        engine_id=engine_id,
                        ctx=ctx,
                    ):
                        break
                    break

                if token_budget <= 0:
                    break

                self._ensure_prefix_block_count(req)
                prefix_tokens = self._waiting_prefix_tokens(req)

                if not self.enable_chunked_prefill and prefix_tokens > token_budget:
                    break

                num_new_tokens = (
                    min(prefix_tokens, token_budget)
                    if self.enable_chunked_prefill
                    else prefix_tokens
                )

                block_hashes = self._blocks_for_prefill_chunk(req, num_new_tokens)
                if not block_hashes:
                    break

                plan = self._allocate_blocks(
                    req, block_hashes, scheduled_ids, scheduled.preempted, now=now, ctx=ctx
                )
                if plan is None:
                    break

                tokens = self._entry_scheduled_tokens(
                    WorkEntry(req, block_hashes, plan, num_new_tokens)
                )
                if tokens > token_budget:
                    break

                self.waiting.popleft()
                req.status = RequestStatus.RUNNING
                if now is not None:
                    req.metrics.enter_running(now, engine_id=engine_id)
                self.running.append(req)
                entry = WorkEntry(req, block_hashes, plan, num_new_tokens)
                scheduled.entries.append(entry)
                scheduled_ids.add(req.req_id)
                token_budget -= tokens
                scheduled.total_num_scheduled_tokens += tokens

        return scheduled

    def _blocks_for_running(self, req: Request, num_new_tokens: int) -> list[str]:
        if req.is_prefill_chunk():
            return self._blocks_for_prefill_chunk(req, num_new_tokens)
        if req.pd != RequestPD.DECODE:
            return []
        if req.num_computed_blocks >= req.blocks_target():
            return []

        block_hash = f"blk:{req.req_id}:{req.num_computed_blocks}"
        req.pending_block_hash = block_hash
        return [block_hash]

    def _allocate_blocks(
        self,
        req: Request,
        block_hashes: list[str],
        scheduled_ids: set[str],
        preempted: list[Request],
        *,
        pull_only: bool = False,
        now: float | None = None,
        ctx: SimContext | None = None,
    ) -> EntryPlan | None:
        while True:
            plan = self.controller.plan_blocks(
                self.memories,
                block_hashes,
                allow_compute=not pull_only,
                req=req,
                block_size=self.block_size,
                ctx=ctx,
                known_requests=[
                    *self.waiting,
                    *self.running,
                    *self.completed,
                ],
            )
            if plan is not None:
                return plan

            victim = self._pick_preemption_victim(req, scheduled_ids)
            if victim is None:
                return None

            self._preempt_request(victim, now=now)
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

    def _preempt_request(self, req: Request, *, now: float | None = None) -> None:
        """Free KV and return to waiting. No task cancel — batches drain before re-schedule."""
        assert req in self.running
        self._local().free_request(req.req_id)

        req.num_computed_blocks = 0
        req.pending_block_hash = None
        req.block_hashes = req.block_hashes[: req.prefix_block_count]
        req.metrics.preemptions += 1
        req.status = RequestStatus.WAITING
        if now is not None:
            req.metrics.enter_waiting(now)

        self.running.remove(req)
        self.waiting.appendleft(req)

    def finish_request(self, req: Request, *, now: float | None = None) -> None:
        if req not in self.running:
            return
        req.status = RequestStatus.COMPLETE
        if now is not None:
            req.metrics.finish(now)
        self._local().free_request(req.req_id)
        self.running.remove(req)
        self.completed.append(req)
