"""Execute ``BatchPlan``: reserve slots, build tasks, apply placement/retention on completion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from simulator.core.kv_content import (
    ContentKey,
    group_pulls,
    shared_pull_key,
    storage_key,
    tier_has_block,
    tier_has_inflight,
    transfer_work,
)
from simulator.observability.estimate import batch_forward_work, record_entry_metrics
from simulator.policy.block_resolve import first_resident_pull_source
from simulator.core.memory import BlockState, KVBlock, Memory
from .plan import BatchPlan, EvictPlan, StoreOp, WorkEntry
from simulator.core.request import Request
from .tasks import BatchLoadTask, EvictTask, ForwardTask, StoreTask, Task

if TYPE_CHECKING:
    from .engine import Engine

BlockCallback = Callable[[KVBlock, float], None]


class BatchRunner:
    """Turn a scheduled ``BatchPlan`` into ``TaskPool`` work for one engine."""

    def __init__(self, engine: Engine, scheduled_at: float):
        self._engine = engine
        self._at = scheduled_at
        self._spill_removals: dict[int, KVBlock] = {}

    def run(self, plan: BatchPlan) -> None:
        local = self._local()
        evict_tasks: list[Task] = []
        pull_tasks: list[Task] = []
        forward_blocks: list[KVBlock] = []
        forward_callbacks: dict[int, BlockCallback] = {}
        shared_pulls: dict[tuple[str, ContentKey], BatchLoadTask] = {}

        for entry in plan.entries:
            record_entry_metrics(entry)
            self._reserve(entry)

            for evict_plan in entry.plan.evicts:
                victim = evict_plan.block
                spill_req = self._engine.cache.resolve_spill_req(
                    victim.hash, self._engine.known_requests()
                )
                if self._engine.sync_evict:
                    self._sync_evict(
                        evict_plan,
                        entry,
                        spill_req=spill_req,
                        batch_id=plan.batch_id,
                    )
                else:
                    on_evict = (
                        self._local_evict_cb(spill_req)
                        if spill_req is not None
                        else None
                    )
                    task = EvictTask(
                        self._engine.work_per_evict,
                        self._engine.compute_res,
                        local,
                        victim,
                        on_evict_before=on_evict,
                    )
                    self._add_task(task, [], plan.batch_id)
                    evict_tasks.append(task)

            for src_key, group_hashes in group_pulls(
                entry.block_hashes,
                entry.plan.blocks,
                entry.req,
                self._engine.memories,
            ):
                self._schedule_pull(
                    entry,
                    src_key,
                    group_hashes,
                    evict_tasks,
                    plan.batch_id,
                    pull_tasks,
                    shared_pulls,
                )

            for block_hash in entry.block_hashes:
                if entry.plan.blocks.get(block_hash) != "compute":
                    continue
                dst = local.find_reserved_for(block_hash, entry.req.req_id)
                if dst is None:
                    raise RuntimeError(
                        f"no reserved block for {block_hash!r} "
                        f"request {entry.req.req_id!r}"
                    )
                forward_blocks.append(dst)
                forward_callbacks[id(dst)] = self._local_resident_cb(entry.req)

        if not forward_blocks:
            return

        forward = ForwardTask(
            batch_forward_work(
                plan,
                work_per_prefill_token=self._engine.work_per_prefill_token,
                work_per_decode_req=self._engine.work_per_decode_req,
            ),
            self._engine.compute_res,
            forward_blocks,
            forward_callbacks,
        )
        self._add_task(forward, evict_tasks + pull_tasks, plan.batch_id)

        seen_store: set[tuple[str, ContentKey]] = set()
        for entry in plan.entries:
            for block_hash, ops in entry.plan.store_ops.items():
                if not ops:
                    continue
                if local.find_reserved_for(block_hash, entry.req.req_id) is None:
                    continue
                self._schedule_stores(
                    ops,
                    req=entry.req,
                    prereqs=[forward],
                    batch_id=plan.batch_id,
                    seen_store=seen_store,
                )

    def _local(self) -> Memory:
        return self._engine.memories[self._engine.local_memory]

    def _add_task(self, task: Task, prereqs: list[Task], batch_id: int) -> None:
        if self._engine.event_trace is not None:
            task.trace_meta = {
                "engine_id": self._engine.engine_id,
                "batch_id": batch_id,
            }
        self._engine.pool.add(task, prereqs, batch_id=batch_id)

    def _local_resident_cb(self, req: Request) -> BlockCallback:
        cache = self._engine.cache

        def cb(block: KVBlock, now: float) -> None:
            cache.on_local_resident(block, req, now)

        return cb

    def _local_evict_cb(self, req: Request) -> BlockCallback:
        cache = self._engine.cache

        def cb(block: KVBlock, now: float) -> None:
            cache.on_local_evict(block, req, now)

        return cb

    def _pull_complete_cb(self, req: Request, src_key: str) -> BlockCallback:
        cache = self._engine.cache

        def cb(block: KVBlock, now: float) -> None:
            cache.on_local_resident(block, req, now)
            cache.after_pull(src_key, block.hash, req)

        return cb

    def _tier_resident_cb(self, req: Request, tier_key: str) -> BlockCallback:
        cache = self._engine.cache

        def cb(block: KVBlock, now: float) -> None:
            cache.on_tier_resident(tier_key, block, req, now)

        return cb

    def _spill_complete_cb(self, req: Request, tier_key: str) -> BlockCallback:
        cache = self._engine.cache

        def cb(block: KVBlock, now: float) -> None:
            cache.on_tier_resident(tier_key, block, req, now)
            victim = self._spill_removals.pop(id(block), None)
            if victim is not None:
                self._local().remove_block(victim)

        return cb

    def _reserve(self, entry: WorkEntry) -> None:
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
                local.touch(resident, self._at)
                continue
            inflight = local.inflight_incoming(block_hash)
            if inflight is not None:
                inflight.holders.add(entry.req.req_id)
                continue
            if first_resident_pull_source(
                self._engine.memories,
                self._engine.policies.schedule.pull_sources,
                block_hash,
                req=entry.req,
            ) is None:
                self._attach_remote_wait(entry.req, block_hash)

    def _attach_remote_wait(self, req: Request, block_hash: str) -> None:
        for src_key in self._engine.policies.schedule.pull_sources:
            src = self._engine.memories[src_key]
            if tier_has_block(src, req, block_hash):
                continue
            if tier_has_inflight(src, req, block_hash):
                slot = storage_key(req, block_hash, src.chunk_blocks)
                inflight = src.inflight_incoming(slot)
                if inflight is not None:
                    inflight.holders.add(req.req_id)

    def _sync_evict(
        self,
        evict_plan: EvictPlan,
        entry: WorkEntry,
        *,
        spill_req: Request | None,
        batch_id: int,
    ) -> None:
        victim = evict_plan.block
        if spill_req is not None:
            self._engine.cache.on_local_evict(victim, spill_req, self._at)
        spill_ops = evict_plan.store_ops
        if spill_req is not None and spill_ops:
            for op in spill_ops:
                tier_block = self._tier_block_for_store(op)
                if tier_block is None:
                    continue
                self._spill_removals[id(tier_block)] = victim
                self._append_store(
                    op,
                    req=spill_req,
                    hbm_hashes=[victim.hash],
                    prereqs=[],
                    batch_id=batch_id,
                    on_complete=self._spill_complete_cb(spill_req, op.tier_key),
                )
            return
        self._local().remove_block(victim)

    def _tier_block_for_store(self, op: StoreOp) -> KVBlock | None:
        tier = self._engine.memories[op.tier_key]
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
        on_complete: BlockCallback,
    ) -> bool:
        tier_block = self._tier_block_for_store(op)
        if tier_block is None:
            return False
        link = self._engine.write_links.get(op.tier_key)
        if link is None:
            return False
        tier = self._engine.memories[op.tier_key]
        store = StoreTask(
            self._engine.work_per_store * transfer_work(tier, req, hbm_hashes),
            link,
            tier,
            tier_block,
            on_complete=on_complete,
        )
        self._add_task(store, prereqs, batch_id)
        return True

    def _schedule_stores(
        self,
        ops: list[StoreOp],
        *,
        req: Request,
        prereqs: list[Task],
        batch_id: int,
        seen_store: set[tuple[str, ContentKey]],
    ) -> None:
        for op in ops:
            key = (op.tier_key, op.content)
            if key in seen_store:
                continue
            seen_store.add(key)
            self._append_store(
                op,
                req=req,
                hbm_hashes=[op.hbm_block_hash],
                prereqs=prereqs,
                batch_id=batch_id,
                on_complete=self._tier_resident_cb(req, op.tier_key),
            )

    def _schedule_pull(
        self,
        entry: WorkEntry,
        src_key: str,
        group_hashes: list[str],
        prereqs: list[Task],
        batch_id: int,
        pull_tasks: list[Task],
        shared_pulls: dict[tuple[str, ContentKey], BatchLoadTask],
    ) -> None:
        local = self._local()
        link = self._engine.transfer_links.get(src_key)
        if link is None:
            raise RuntimeError(f"transfer link required for pull source {src_key!r}")
        src_mem = self._engine.memories[src_key]
        dedupe_key = shared_pull_key(
            src_key, entry.req, group_hashes[0], self._engine.memories
        )

        dst_blocks: list[KVBlock] = []
        pull_callbacks: dict[int, BlockCallback] = {}
        pull_cb = self._pull_complete_cb(entry.req, src_key)
        for block_hash in group_hashes:
            dst = local.find_reserved_for(block_hash, entry.req.req_id)
            if dst is None:
                raise RuntimeError(
                    f"no reserved block for {block_hash!r} request {entry.req.req_id!r}"
                )
            dst_blocks.append(dst)
            pull_callbacks[id(dst)] = pull_cb

        existing = shared_pulls.get(dedupe_key)
        if existing is not None:
            for dst in dst_blocks:
                existing.add_block(dst, pull_callbacks[id(dst)])
            return

        task = BatchLoadTask(
            self._engine.work_per_transfer
            * transfer_work(src_mem, entry.req, group_hashes),
            link,
            local,
            [dst_blocks[0]],
            {id(dst_blocks[0]): pull_callbacks[id(dst_blocks[0])]},
        )
        for dst in dst_blocks[1:]:
            task.add_block(dst, pull_callbacks[id(dst)])

        self._add_task(task, prereqs, batch_id)
        for block in dst_blocks:
            block.task = task
        shared_pulls[dedupe_key] = task
        pull_tasks.append(task)

        seen_store: set[tuple[str, ContentKey]] = set()
        for block_hash in group_hashes:
            self._schedule_stores(
                entry.plan.store_ops.get(block_hash, []),
                req=entry.req,
                prereqs=[task],
                batch_id=batch_id,
                seen_store=seen_store,
            )
