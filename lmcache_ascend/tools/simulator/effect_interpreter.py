"""Execute ``BatchWork``: reservations, task DAG, resident side effects."""

from __future__ import annotations

from dataclasses import dataclass

from chunk_hash import (
    chunk_key_for_hbm_block,
    chunk_transfer_work,
    group_pull_blocks,
    pull_dedupe_key,
    tier_covers_hbm_block,
    tier_inflight_hbm_block,
)
from content_key import ContentKey
from cost_model import batch_forward_work, record_entry_metrics
from memory import BlockState, KVBlock, Memory, collect_content_copies
from plan import BatchWork, ExecuteResult, WorkEntry
from policies import PlacementPolicy, RetentionPolicy, StoreOp, first_resident_pull_source
from request import Request
from resource import BandwidthResource, ComputeResource
from scheduler import Scheduler
from tasks import BatchLoadTask, EvictTask, ForwardTask, StoreTask, Task, TaskPool


@dataclass
class EngineRuntime:
    """Mutable engine resources the executor reads and updates."""

    engine_id: str
    pool: TaskPool
    memories: dict[str, Memory]
    local_memory: str
    scheduler: Scheduler
    compute_res: ComputeResource
    transfer_links: dict[str, BandwidthResource]
    write_links: dict[str, BandwidthResource]
    placement_policy: PlacementPolicy
    retention_policy: RetentionPolicy
    work_per_block: float
    work_per_transfer: float
    work_per_store: float
    work_per_evict: float
    work_per_prefill_token: float
    work_per_decode_req: float
    sync_evict: bool
    pull_sources: list[str]
    peak_duplicate_count: int = 0


class ResidentEffects:
    """Placement and retention hooks when KV blocks change state."""

    def __init__(self, rt: EngineRuntime):
        self._rt = rt

    def on_hbm_resident(self, block: KVBlock, req: Request, now: float) -> None:
        self._rt.placement_policy.place_copy(
            self._rt.memories,
            local_memory=self._rt.local_memory,
            block=block,
            req=req,
            now=now,
        )
        self._rt.retention_policy.on_block_resident(
            self._rt.memories,
            tier_key=self._rt.local_memory,
            block=block,
            now=now,
            req=req,
        )
        self._track_duplicates(req, self._rt.local_memory, block.hash)

    def on_tier_resident(
        self, tier_key: str, block: KVBlock, req: Request, now: float
    ) -> None:
        self._rt.retention_policy.on_block_resident(
            self._rt.memories,
            tier_key=tier_key,
            block=block,
            now=now,
            req=req,
        )
        self._track_duplicates(req, tier_key, block.hash)

    def on_pull_complete(
        self, block: KVBlock, req: Request, *, src_key: str, now: float
    ) -> None:
        self.on_hbm_resident(block, req, now)
        self._rt.retention_policy.after_pull(
            self._rt.memories,
            src_key=src_key,
            dst_key=self._rt.local_memory,
            block_hash=block.hash,
            now=now,
            req=req,
        )

    def on_hbm_evict(self, block: KVBlock, spill_req: Request | None, now: float) -> None:
        self._rt.placement_policy.spill_on_evict(
            self._rt.memories,
            local_memory=self._rt.local_memory,
            block=block,
            now=now,
            req=spill_req,
        )

    def _track_duplicates(
        self, req: Request | None, tier_key: str, block_hash: str
    ) -> None:
        content = ContentKey.for_storage_key(block_hash)
        count = len(
            collect_content_copies(
                self._rt.memories,
                list(self._rt.memories.keys()),
                content,
                req=req,
            )
        )
        self._rt.peak_duplicate_count = max(self._rt.peak_duplicate_count, count)


class BatchExecutor:
    """Interpret ``BatchWork`` into memory updates and ``TaskPool`` tasks."""

    def __init__(self, rt: EngineRuntime):
        self._rt = rt
        self._effects = ResidentEffects(rt)

    def execute(self, work: BatchWork) -> ExecuteResult:
        local = self._local()
        all_tasks: list[Task] = []
        evict_tasks: list[Task] = []
        pull_tasks: list[Task] = []
        forward_blocks: list[KVBlock] = []
        forward_block_req: dict[int, Request] = {}
        shared_pulls: dict[tuple[str, ContentKey], BatchLoadTask] = {}

        for entry in work.entries:
            record_entry_metrics(entry)
            self._reserve(entry, now=work.scheduled_at)

            for victim in entry.plan.evicts:
                if self._rt.sync_evict:
                    self._sync_evict(
                        victim,
                        work,
                        spill_req=self._req_for_block(victim.hash),
                        all_tasks=all_tasks,
                    )
                else:
                    task = EvictTask(
                        work_left=self._rt.work_per_evict,
                        resource=self._rt.compute_res,
                        memory=local,
                        block=victim,
                        on_before_evict=lambda b, t, v=victim: self._effects.on_hbm_evict(
                            v, self._req_for_block(v.hash), t
                        ),
                    )
                    self._add_task(task, [], work.batch_id, all_tasks)
                    evict_tasks.append(task)

            for src_key, group_hashes in group_pull_blocks(
                entry.block_hashes,
                entry.plan.blocks,
                entry.req,
                self._rt.memories,
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
                    forward_block_req[id(dst)] = entry.req

        if forward_blocks:
            forward = ForwardTask(
                batch_forward_work(
                    work,
                    work_per_prefill_token=self._rt.work_per_prefill_token,
                    work_per_decode_req=self._rt.work_per_decode_req,
                ),
                self._rt.compute_res,
                forward_blocks,
                on_resident=lambda block, t: self._effects.on_hbm_resident(
                    block, forward_block_req[id(block)], t
                ),
            )
            self._add_task(forward, evict_tasks + pull_tasks, work.batch_id, all_tasks)
            seen_store: set[tuple[str, ContentKey]] = set()
            for block in forward_blocks:
                self._schedule_stores_after(
                    forward_block_req[id(block)],
                    block.hash,
                    prereqs=[forward],
                    batch_id=work.batch_id,
                    all_tasks=all_tasks,
                    seen_store=seen_store,
                )

        return ExecuteResult(
            tasks=all_tasks,
            peak_duplicate_count=self._rt.peak_duplicate_count,
        )

    def _local(self) -> Memory:
        return self._rt.memories[self._rt.local_memory]

    def _add_task(
        self,
        task: Task,
        prereqs: list[Task],
        batch_id: int,
        all_tasks: list[Task],
    ) -> None:
        self._rt.pool.add(task, prereqs, batch_id=batch_id)
        all_tasks.append(task)

    def _req_for_block(self, block_hash: str) -> Request | None:
        for req in (
            *self._rt.scheduler.waiting,
            *self._rt.scheduler.running,
            *self._rt.scheduler.completed,
        ):
            if block_hash in req.block_hashes[: req.prefix_block_count]:
                return req
        return None

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
                self._rt.memories, self._rt.pull_sources, block_hash, req=entry.req
            ) is None:
                self._attach_remote_wait(entry.req, block_hash)

    def _attach_remote_wait(self, req: Request, block_hash: str) -> None:
        for src_key in self._rt.pull_sources:
            src = self._rt.memories[src_key]
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
        work: BatchWork,
        *,
        spill_req: Request | None,
        all_tasks: list[Task],
    ) -> None:
        self._effects.on_hbm_evict(victim, spill_req, work.scheduled_at)
        if spill_req is not None:
            spill_stores = self._rt.placement_policy.plan_spill_stores(
                self._rt.memories,
                local_memory=self._rt.local_memory,
                block_hash=victim.hash,
                req=spill_req,
            )
            if spill_stores:
                local = self._local()
                for op in spill_stores:
                    def on_spill_done(
                        b: KVBlock,
                        t: float,
                        tk=op.tier_key,
                        r=spill_req,
                        v=victim,
                    ) -> None:
                        self._effects.on_tier_resident(tk, b, r, t)
                        local.remove_block(v)

                    self._append_store(
                        op,
                        req=spill_req,
                        hbm_hashes=[victim.hash],
                        prereqs=[],
                        batch_id=work.batch_id,
                        all_tasks=all_tasks,
                        on_resident=on_spill_done,
                    )
                return
        self._local().remove_block(victim)

    def _tier_block_for_store(self, op: StoreOp) -> KVBlock | None:
        tier = self._rt.memories[op.tier_key]
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
        on_resident,
    ) -> bool:
        tier_block = self._tier_block_for_store(op)
        if tier_block is None:
            return False
        link = self._rt.write_links.get(op.tier_key)
        if link is None:
            return False
        tier = self._rt.memories[op.tier_key]
        store = StoreTask(
            self._rt.work_per_store * chunk_transfer_work(tier, req, hbm_hashes),
            link,
            tier,
            tier_block,
            on_resident=on_resident,
        )
        self._add_task(store, prereqs, batch_id, all_tasks)
        return True

    def _schedule_stores_after(
        self,
        req: Request,
        hbm_hash: str,
        *,
        prereqs: list[Task],
        batch_id: int,
        all_tasks: list[Task],
        seen_store: set[tuple[str, ContentKey]],
    ) -> None:
        ops = self._rt.placement_policy.plan_async_stores(
            self._rt.memories,
            local_memory=self._rt.local_memory,
            block_hash=hbm_hash,
            req=req,
        )
        for op in ops:
            key = (op.tier_key, op.content)
            if key in seen_store:
                continue
            seen_store.add(key)
            self._append_store(
                op,
                req=req,
                hbm_hashes=[hbm_hash],
                prereqs=prereqs,
                batch_id=batch_id,
                all_tasks=all_tasks,
                on_resident=lambda b, t, tk=op.tier_key, r=req: self._effects.on_tier_resident(
                    tk, b, r, t
                ),
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
    ) -> None:
        local = self._local()
        link = self._rt.transfer_links.get(src_key)
        if link is None:
            raise RuntimeError(f"transfer link required for pull source {src_key!r}")
        src_mem = self._rt.memories[src_key]
        dedupe_key = pull_dedupe_key(src_key, entry.req, group_hashes[0], self._rt.memories)

        dst_blocks: list[KVBlock] = []
        for block_hash in group_hashes:
            dst = local.find_reserved_for(block_hash, entry.req.req_id)
            if dst is None:
                raise RuntimeError(
                    f"no reserved block for {block_hash!r} request {entry.req.req_id!r}"
                )
            dst_blocks.append(dst)

        def pull_callback(req: Request, src: str):
            return lambda block, t, r=req, s=src: self._effects.on_pull_complete(
                block, r, src_key=s, now=t
            )

        existing = shared_pulls.get(dedupe_key)
        if existing is not None:
            for dst in dst_blocks:
                existing.add_block(dst, pull_callback(entry.req, src_key))
            return

        task = BatchLoadTask(
            self._rt.work_per_transfer * chunk_transfer_work(src_mem, entry.req, group_hashes),
            link,
            local,
            [dst_blocks[0]],
            on_resident=pull_callback(entry.req, src_key),
        )
        for dst in dst_blocks[1:]:
            task.add_block(dst, pull_callback(entry.req, src_key))

        self._add_task(task, prereqs, batch_id, all_tasks)
        for block in dst_blocks:
            block.task = task
        shared_pulls[dedupe_key] = task
        pull_tasks.append(task)

        seen_store: set[tuple[str, ContentKey]] = set()
        for block_hash in group_hashes:
            self._schedule_stores_after(
                entry.req,
                block_hash,
                prereqs=[task],
                batch_id=batch_id,
                all_tasks=all_tasks,
                seen_store=seen_store,
            )
