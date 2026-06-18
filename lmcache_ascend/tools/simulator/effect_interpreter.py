"""Execute ``BatchPlan`` mechanically: reservations, tasks, planned outcome effects."""

from __future__ import annotations

from dataclasses import dataclass

from .chunk_hash import (
    chunk_key_for_hbm_block,
    chunk_transfer_work,
    group_pull_blocks,
    pull_dedupe_key,
    tier_covers_hbm_block,
    tier_inflight_hbm_block,
)
from .content_key import ContentKey
from .cost_model import batch_forward_work, record_entry_metrics
from .memory import BlockState, KVBlock, Memory, collect_content_copies
from .lookup import first_resident_pull_source
from .plan import BatchPlan, RetentionProfile, StoreOp, WorkEntry
from .request import Request
from .resource import BandwidthResource, ComputeResource
from .task_outcomes import TaskOutcome
from .tasks import BatchLoadTask, EvictTask, ForwardTask, StoreTask, Task, TaskPool


@dataclass
class ExecuteContext:
    """Per-batch execute resources (no policy objects)."""

    pool: TaskPool
    memories: dict[str, Memory]
    local_memory: str
    compute_res: ComputeResource
    transfer_links: dict[str, BandwidthResource]
    write_links: dict[str, BandwidthResource]
    work_per_transfer: float
    work_per_store: float
    work_per_evict: float
    work_per_prefill_token: float
    work_per_decode_req: float
    sync_evict: bool
    retention: RetentionProfile
    pull_sources: list[str]
    scheduled_at: float
    on_hbm_evict: object
    on_hbm_resident: object
    on_tier_resident: object
    after_pull: object
    expand_async_stores: object  # Callable[[WorkEntry, str], list[StoreOp]]
    expand_spill: object  # Callable[[WorkEntry, KVBlock], list[StoreOp]]
    peak_duplicate_count: int = 0


class OutcomeApplicator:
    """Apply pre-planned completion effects via engine-provided hooks."""

    def __init__(
        self,
        ctx: ExecuteContext,
        work: BatchPlan,
        *,
        on_hbm_resident,
        on_tier_resident,
        after_pull,
    ):
        self._ctx = ctx
        self._req_by_id = {entry.req.req_id: entry.req for entry in work.entries}
        self._spill_removals: dict[int, KVBlock] = {}
        self._on_hbm_resident = on_hbm_resident
        self._on_tier_resident = on_tier_resident
        self._after_pull = after_pull

    def spill_removals(self) -> dict[int, KVBlock]:
        return self._spill_removals

    def __call__(self, outcome: TaskOutcome, block: KVBlock, now: float) -> None:
        req = self._req_by_id.get(outcome.req_id)
        if outcome.kind == "hbm_evict_before":
            if req is not None:
                self._ctx.on_hbm_evict(block, req, now)
            return
        if outcome.kind == "hbm_resident" and req is not None:
            self._on_hbm_resident(block, req, now)
        elif outcome.kind == "tier_resident" and req is not None and outcome.tier_key:
            self._on_tier_resident(outcome.tier_key, block, req, now)
        elif (
            outcome.kind == "pull_complete"
            and req is not None
            and outcome.src_key is not None
        ):
            self._on_hbm_resident(block, req, now)
            self._after_pull(
                outcome.src_key, block.hash, req, now
            )
        elif outcome.kind == "spill_complete" and req is not None and outcome.tier_key:
            self._on_tier_resident(outcome.tier_key, block, req, now)
            victim = self._spill_removals.pop(id(block), None)
            if victim is not None:
                self._ctx.memories[self._ctx.local_memory].remove_block(victim)


class BatchExecutor:
    """Translate ``BatchPlan`` into ``TaskPool`` tasks without policy calls."""

    def __init__(self, ctx: ExecuteContext):
        self._ctx = ctx

    def execute(self, work: BatchPlan) -> None:
        applicator = OutcomeApplicator(
            self._ctx,
            work,
            on_hbm_resident=self._ctx.on_hbm_resident,
            on_tier_resident=self._ctx.on_tier_resident,
            after_pull=self._ctx.after_pull,
        )
        local = self._local()
        all_tasks: list[Task] = []
        evict_tasks: list[Task] = []
        pull_tasks: list[Task] = []
        forward_blocks: list[KVBlock] = []
        forward_outcomes: dict[int, TaskOutcome] = {}
        shared_pulls: dict[tuple[str, ContentKey], BatchLoadTask] = {}

        for entry in work.entries:
            record_entry_metrics(entry)
            self._reserve(entry, now=work.scheduled_at)

            for victim in entry.plan.evicts:
                spill_req = entry.plan.spill_reqs.get(id(victim))
                if self._ctx.sync_evict:
                    self._sync_evict(
                        victim,
                        entry,
                        spill_req=spill_req,
                        applicator=applicator,
                        all_tasks=all_tasks,
                        batch_id=work.batch_id,
                    )
                else:
                    outcome = TaskOutcome(
                        kind="hbm_evict_before",
                        req_id=spill_req.req_id if spill_req is not None else "",
                        block_hash=victim.hash,
                    )
                    task = EvictTask(
                        self._ctx.work_per_evict,
                        self._ctx.compute_res,
                        local,
                        victim,
                        outcome=outcome,
                        outcome_handler=applicator,
                    )
                    self._add_task(task, [], work.batch_id, all_tasks)
                    evict_tasks.append(task)

            for src_key, group_hashes in group_pull_blocks(
                entry.block_hashes,
                entry.plan.blocks,
                entry.req,
                self._ctx.memories,
            ):
                self._schedule_pull(
                    entry,
                    src_key,
                    group_hashes,
                    evict_tasks,
                    work.batch_id,
                    all_tasks,
                    pull_tasks,
                    shared_pulls,
                    applicator,
                )

            for block_hash in entry.block_hashes:
                if entry.plan.blocks.get(block_hash) == "compute":
                    dst = local.find_reserved_for(block_hash, entry.req.req_id)
                    if dst is None:
                        raise RuntimeError(
                            f"no reserved block for {block_hash!r} "
                            f"request {entry.req.req_id!r}"
                        )
                    forward_blocks.append(dst)
                    forward_outcomes[id(dst)] = entry.plan.resident_outcomes.get(
                        block_hash,
                        TaskOutcome(
                            kind="hbm_resident",
                            req_id=entry.req.req_id,
                            block_hash=block_hash,
                        ),
                    )

        if forward_blocks:
            forward = ForwardTask(
                batch_forward_work(
                    work,
                    work_per_prefill_token=self._ctx.work_per_prefill_token,
                    work_per_decode_req=self._ctx.work_per_decode_req,
                ),
                self._ctx.compute_res,
                forward_blocks,
                forward_outcomes,
                outcome_handler=applicator,
            )
            self._add_task(forward, evict_tasks + pull_tasks, work.batch_id, all_tasks)
            seen_store: set[tuple[str, ContentKey]] = set()
            for entry in work.entries:
                for block_hash in entry.block_hashes:
                    if entry.plan.blocks.get(block_hash) != "compute":
                        continue
                    local_block = local.find_reserved_for(
                        block_hash, entry.req.req_id
                    )
                    if local_block is None:
                        continue
                    self._schedule_stores(
                        self._store_ops(entry, block_hash),
                        req=entry.req,
                        prereqs=[forward],
                        batch_id=work.batch_id,
                        all_tasks=all_tasks,
                        seen_store=seen_store,
                        applicator=applicator,
                    )

    def _local(self) -> Memory:
        return self._ctx.memories[self._ctx.local_memory]

    def _add_task(
        self,
        task: Task,
        prereqs: list[Task],
        batch_id: int,
        all_tasks: list[Task],
    ) -> None:
        self._ctx.pool.add(task, prereqs, batch_id=batch_id)
        all_tasks.append(task)

    def _reserve(self, entry: WorkEntry, *, now: float) -> None:
        local = self._local()
        for block_hash in entry.block_hashes:
            action = entry.plan.blocks.get(block_hash)
            if action == "wait":
                self._attach_remote_wait(entry.req, block_hash)
                continue
            if block_hash in entry.plan.blocks:
                local.append_reserved(block_hash, entry.req.req_id)
                continue
            resident = local.best_resident(block_hash)
            if resident is not None:
                resident.holders.add(entry.req.req_id)
                local.touch(resident, now)
                continue
            inflight = local.inflight_incoming(block_hash)
            if inflight is not None:
                inflight.holders.add(entry.req.req_id)
                continue
            if first_resident_pull_source(
                self._ctx.memories,
                self._ctx.pull_sources,
                block_hash,
                req=entry.req,
            ) is None:
                self._attach_remote_wait(entry.req, block_hash)

    def _attach_remote_wait(self, req: Request, block_hash: str) -> None:
        for src_key in self._ctx.pull_sources:
            src = self._ctx.memories[src_key]
            if tier_covers_hbm_block(src, req, block_hash):
                continue
            if tier_inflight_hbm_block(src, req, block_hash):
                storage_key = chunk_key_for_hbm_block(req, block_hash, src.chunk_blocks)
                inflight = src.inflight_incoming(storage_key)
                if inflight is not None:
                    inflight.holders.add(req.req_id)

    def _sync_evict(
        self,
        victim: KVBlock,
        entry: WorkEntry,
        *,
        spill_req: Request | None,
        applicator: OutcomeApplicator,
        all_tasks: list[Task],
        batch_id: int,
    ) -> None:
        if spill_req is not None:
            self._ctx.on_hbm_evict(victim, spill_req, self._ctx.scheduled_at)
        spill_ops = entry.plan.spill_store_ops.get(id(victim), [])
        if spill_req is not None and not spill_ops:
            spill_ops = self._ctx.expand_spill(entry, victim)
        if spill_req is not None and spill_ops:
            for op in spill_ops:
                tier_block = self._tier_block_for_store(op)
                if tier_block is None:
                    continue
                applicator._spill_removals[id(tier_block)] = victim
                outcome = TaskOutcome(
                    kind="spill_complete",
                    req_id=spill_req.req_id,
                    block_hash=op.storage_key,
                    tier_key=op.tier_key,
                    remove_hbm_hash=victim.hash,
                )
                self._append_store(
                    op,
                    req=spill_req,
                    hbm_hashes=[victim.hash],
                    prereqs=[],
                    batch_id=batch_id,
                    all_tasks=all_tasks,
                    outcome=outcome,
                    applicator=applicator,
                )
            return
        self._local().remove_block(victim)

    def _tier_block_for_store(self, op: StoreOp) -> KVBlock | None:
        tier = self._ctx.memories[op.tier_key]
        for block in tier.get(op.storage_key):
            if block.state in (BlockState.RESERVED, BlockState.LOADING):
                return block
        return None

    def _append_store(
        self,
        op: StoreOp,
        *,
        req: Request,
        hbm_hashes: list[str],
        prereqs: list[Task],
        batch_id: int,
        all_tasks: list[Task],
        outcome: TaskOutcome,
        applicator: OutcomeApplicator,
    ) -> bool:
        tier_block = self._tier_block_for_store(op)
        if tier_block is None:
            return False
        link = self._ctx.write_links.get(op.tier_key)
        if link is None:
            return False
        tier = self._ctx.memories[op.tier_key]
        store = StoreTask(
            self._ctx.work_per_store * chunk_transfer_work(tier, req, hbm_hashes),
            link,
            tier,
            tier_block,
            outcome=outcome,
            outcome_handler=applicator,
        )
        self._add_task(store, prereqs, batch_id, all_tasks)
        return True

    def _store_ops(self, entry: WorkEntry, block_hash: str) -> list[StoreOp]:
        ops = entry.plan.store_ops.get(block_hash)
        if ops is None:
            ops = self._ctx.expand_async_stores(entry, block_hash)
            entry.plan.store_ops[block_hash] = ops
        return ops

    def _schedule_stores(
        self,
        ops: list[StoreOp],
        *,
        req: Request,
        prereqs: list[Task],
        batch_id: int,
        all_tasks: list[Task],
        seen_store: set[tuple[str, ContentKey]],
        applicator: OutcomeApplicator,
    ) -> None:
        for op in ops:
            key = (op.tier_key, op.content)
            if key in seen_store:
                continue
            seen_store.add(key)
            outcome = TaskOutcome(
                kind="tier_resident",
                req_id=req.req_id,
                block_hash=op.hbm_block_hash,
                tier_key=op.tier_key,
            )
            self._append_store(
                op,
                req=req,
                hbm_hashes=[op.hbm_block_hash],
                prereqs=prereqs,
                batch_id=batch_id,
                all_tasks=all_tasks,
                outcome=outcome,
                applicator=applicator,
            )

    def _schedule_pull(
        self,
        entry: WorkEntry,
        src_key: str,
        group_hashes: list[str],
        prereqs: list[Task],
        batch_id: int,
        all_tasks: list[Task],
        pull_tasks: list[Task],
        shared_pulls: dict[tuple[str, ContentKey], BatchLoadTask],
        applicator: OutcomeApplicator,
    ) -> None:
        local = self._local()
        link = self._ctx.transfer_links.get(src_key)
        if link is None:
            raise RuntimeError(f"transfer link required for pull source {src_key!r}")
        src_mem = self._ctx.memories[src_key]
        dedupe_key = pull_dedupe_key(
            src_key, entry.req, group_hashes[0], self._ctx.memories
        )

        dst_blocks: list[KVBlock] = []
        pull_outcomes: dict[int, TaskOutcome] = {}
        for block_hash in group_hashes:
            dst = local.find_reserved_for(block_hash, entry.req.req_id)
            if dst is None:
                raise RuntimeError(
                    f"no reserved block for {block_hash!r} request {entry.req.req_id!r}"
                )
            dst_blocks.append(dst)
            pull_outcomes[id(dst)] = entry.plan.pull_outcomes.get(
                block_hash,
                TaskOutcome(
                    kind="pull_complete",
                    req_id=entry.req.req_id,
                    block_hash=block_hash,
                    src_key=src_key,
                ),
            )

        existing = shared_pulls.get(dedupe_key)
        if existing is not None:
            for dst in dst_blocks:
                existing.add_block(dst, pull_outcomes[id(dst)])
            return

        task = BatchLoadTask(
            self._ctx.work_per_transfer
            * chunk_transfer_work(src_mem, entry.req, group_hashes),
            link,
            local,
            [dst_blocks[0]],
            {id(dst_blocks[0]): pull_outcomes[id(dst_blocks[0])]},
            outcome_handler=applicator,
        )
        for dst in dst_blocks[1:]:
            task.add_block(dst, pull_outcomes[id(dst)])

        self._add_task(task, prereqs, batch_id, all_tasks)
        for block in dst_blocks:
            block.task = task
        shared_pulls[dedupe_key] = task
        pull_tasks.append(task)

        seen_store: set[tuple[str, ContentKey]] = set()
        for block_hash in group_hashes:
            self._schedule_stores(
                self._store_ops(entry, block_hash),
                req=entry.req,
                prereqs=[task],
                batch_id=batch_id,
                all_tasks=all_tasks,
                seen_store=seen_store,
                applicator=applicator,
            )
