from chunk_hash import (
    anchor_hbm_hash,
    chunk_key_for_hbm_block,
    chunk_transfer_work,
    group_pull_blocks,
    pull_dedupe_key,
    tier_covers_hbm_block,
    tier_inflight_hbm_block,
)
from cost_model import batch_forward_work, entry_has_compute
from memory import BlockState, KVBlock, Memory, collect_content_copies
from policies import (
    HBMOnly,
    LookupPolicy,
    LookupResult,
    PlacementPolicy,
    RetentionPolicy,
    UnboundedRetention,
    first_resident_pull_source,
)
from request import Request, RequestPD, RequestStatus
from resource import BandwidthResource, ComputeResource
from scheduler import Batch, BatchEntry, Scheduler
from tasks import BatchLoadTask, EvictTask, ForwardTask, StoreTask, Task, TaskPool


class Engine:
    def __init__(
        self,
        engine_id: str,
        requests: list[Request],
        pool: TaskPool,
        memories: dict[str, Memory],
        local_memory: str,
        policy: LookupPolicy,
        compute_res: ComputeResource,
        bandwidth_res: BandwidthResource | None = None,
        transfer_links: dict[str, BandwidthResource] | None = None,
        write_links: dict[str, BandwidthResource] | None = None,
        work_per_block: float = 1.0,
        work_per_transfer: float | None = None,
        work_per_store: float | None = None,
        work_per_evict: float | None = None,
        block_size: int = 1,
        max_num_seqs: int = 10_000,
        max_num_batched_tokens: int = 10_000,
        enable_chunked_prefill: bool = False,
        remote_kv_wait: bool = False,
        hold_kv_on_complete: bool = False,
        work_per_prefill_token: float | None = None,
        work_per_decode_req: float | None = None,
        placement_policy: PlacementPolicy | None = None,
        retention_policy: RetentionPolicy | None = None,
        sync_evict: bool = True,
    ):
        self.engine_id = engine_id
        self.pool = pool
        self.memories = memories
        self.local_memory = local_memory
        self.policy = policy
        self.compute_res = compute_res
        self.bandwidth_res = bandwidth_res
        if transfer_links is None and bandwidth_res is not None and policy.pull_sources:
            transfer_links = {src: bandwidth_res for src in policy.pull_sources}
        self.transfer_links = transfer_links or {}
        self.write_links = write_links or {}
        self.work_per_block = work_per_block
        self.work_per_transfer = (
            work_per_transfer if work_per_transfer is not None else work_per_block
        )
        self.work_per_store = (
            work_per_store if work_per_store is not None else work_per_transfer
        )
        self.work_per_evict = work_per_evict if work_per_evict is not None else 0.0
        self.sync_evict = sync_evict
        self.block_size = block_size
        self.work_per_prefill_token = (
            work_per_prefill_token if work_per_prefill_token is not None else work_per_block
        )
        self.work_per_decode_req = (
            work_per_decode_req if work_per_decode_req is not None else work_per_block
        )
        self.hold_kv_on_complete = hold_kv_on_complete
        self.placement_policy = placement_policy or HBMOnly()
        self.retention_policy = retention_policy or UnboundedRetention()
        self.placement_policy.bind_retention(self.retention_policy)
        self._peak_duplicate_count = 0

        policy.bind_resources(
            compute_res=self.compute_res,
            transfer_links=self.transfer_links,
            work_per_transfer=self.work_per_transfer,
            work_per_block=self.work_per_block,
            work_per_prefill_token=self.work_per_prefill_token,
            work_per_decode_req=self.work_per_decode_req,
            block_size=self.block_size,
        )

        self.scheduler = Scheduler(
            policy=policy,
            memories=memories,
            local_memory=local_memory,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            block_size=block_size,
            enable_chunked_prefill=enable_chunked_prefill,
            remote_kv_wait=remote_kv_wait,
        )
        for req in requests:
            self.scheduler.add_request(req)

    @property
    def waiting(self):
        return self.scheduler.waiting

    @property
    def running(self):
        return self.scheduler.running

    @property
    def completed(self):
        return self.scheduler.completed

    @property
    def remote_kv_wait(self) -> bool:
        return self.scheduler.remote_kv_wait

    @remote_kv_wait.setter
    def remote_kv_wait(self, enabled: bool) -> None:
        self.scheduler.remote_kv_wait = enabled

    def _local(self) -> Memory:
        return self.memories[self.local_memory]

    def schedule_request(self, req: Request) -> None:
        self.scheduler.add_request(req)

    def release_arrivals(self, now: float) -> None:
        self.scheduler.release_arrivals(now)

    def next_arrival(self) -> float | None:
        return self.scheduler.next_arrival()

    def schedule(self, now: float) -> Batch:
        return self.scheduler.schedule(now, engine_id=self.engine_id)

    def _record_entry_actions(self, entry: BatchEntry) -> None:
        metrics = entry.req.metrics
        metrics.evictions += len(entry.result.evicts)
        for block_hash in entry.block_hashes:
            action = entry.result.blocks.get(block_hash)
            if action == "compute":
                metrics.computes += 1
            elif action == "wait":
                metrics.remote_waits += 1
            elif isinstance(action, tuple) and action[0] == "pull":
                metrics.pulls += 1
            elif action is None:
                metrics.local_hits += 1
        if entry_has_compute(entry):
            metrics.forward_steps += 1

    def _track_duplicates(self, req: Request | None, tier_key: str, block_hash: str) -> None:
        anchor = (
            anchor_hbm_hash(req, tier_key, block_hash, self.memories)
            if req is not None
            else block_hash
        )
        count = len(
            collect_content_copies(
                self.memories,
                list(self.memories.keys()),
                req,
                anchor,
            )
        )
        self._peak_duplicate_count = max(self._peak_duplicate_count, count)

    def _on_block_resident(self, block: KVBlock, req: Request, now: float) -> None:
        self.placement_policy.place_copy(
            self.memories,
            local_memory=self.local_memory,
            block=block,
            req=req,
            now=now,
        )
        self.retention_policy.on_block_resident(
            self.memories,
            tier_key=self.local_memory,
            block=block,
            now=now,
            req=req,
        )
        self._track_duplicates(req, self.local_memory, block.hash)

    def _on_tier_block_resident(
        self, tier_key: str, block: KVBlock, req: Request, now: float
    ) -> None:
        self.retention_policy.on_block_resident(
            self.memories,
            tier_key=tier_key,
            block=block,
            now=now,
            req=req,
        )
        self._track_duplicates(req, tier_key, block.hash)

    def _on_pull_complete(
        self,
        block: KVBlock,
        req: Request,
        *,
        src_key: str,
        now: float,
    ) -> None:
        self._on_block_resident(block, req, now)
        self.retention_policy.after_pull(
            self.memories,
            src_key=src_key,
            dst_key=self.local_memory,
            block_hash=block.hash,
            now=now,
            req=req,
        )

    def _req_for_block_hash(self, block_hash: str) -> Request | None:
        for req in (
            *self.scheduler.waiting,
            *self.scheduler.running,
            *self.scheduler.completed,
        ):
            prefix = req.block_hashes[: req.prefix_block_count]
            if block_hash in prefix:
                return req
        return None

    def _on_hbm_evict(self, block: KVBlock, spill_req: Request | None, now: float) -> None:
        self.placement_policy.spill_on_evict(
            self.memories,
            local_memory=self.local_memory,
            block=block,
            now=now,
            req=spill_req,
        )

    def _sync_evict_victim(
        self,
        victim: KVBlock,
        now: float,
        *,
        spill_req: Request | None,
        all_tasks: list[Task],
    ) -> None:
        self._on_hbm_evict(victim, spill_req, now)
        if spill_req is not None:
            spill_stores = self.placement_policy.plan_spill_stores(
                self.memories,
                local_memory=self.local_memory,
                block_hash=victim.hash,
                req=spill_req,
            )
            if spill_stores:
                local = self._local()
                for op in spill_stores:
                    tier_block = self._tier_block_for_store(op)
                    if tier_block is None:
                        continue
                    link = self.write_links.get(op.tier_key)
                    if link is None:
                        continue
                    tier = self.memories[op.tier_key]
                    store_work = self.work_per_store * chunk_transfer_work(
                        tier, spill_req, [victim.hash]
                    )

                    def on_spill_done(
                        b: KVBlock,
                        t: float,
                        tk=op.tier_key,
                        r=spill_req,
                        v=victim,
                    ) -> None:
                        self._on_tier_block_resident(tk, b, r, t)
                        local.remove_block(v)

                    store = StoreTask(
                        work_left=store_work,
                        resource=link,
                        tier=tier,
                        tier_block=tier_block,
                        on_resident=on_spill_done,
                    )
                    self.pool.add(store, [])
                    all_tasks.append(store)
                return
        self._local().remove_block(victim)

    def _tier_block_for_store(self, op) -> KVBlock | None:
        tier = self.memories[op.tier_key]
        for block in tier.get(op.content_key):
            if block.state in (BlockState.RESERVED, BlockState.LOADING):
                return block
        return None

    def _schedule_pull_group(
        self,
        entry: BatchEntry,
        src_key: str,
        group_hashes: list[str],
        prereqs: list[Task],
        all_tasks: list[Task],
        pull_tasks: list[Task],
        store_tasks: list[Task],
        shared_pulls: dict[tuple[str, str], BatchLoadTask],
    ) -> None:
        local = self._local()
        link = self.transfer_links.get(src_key)
        if link is None:
            raise RuntimeError(f"transfer link required for pull source {src_key!r}")
        src_mem = self.memories[src_key]
        dedupe_key = pull_dedupe_key(src_key, entry.req, group_hashes[0], self.memories)

        dst_blocks: list[KVBlock] = []
        for block_hash in group_hashes:
            dst = local.find_reserved_for(block_hash, entry.req.req_id)
            if dst is None:
                raise RuntimeError(
                    f"no reserved block for {block_hash!r} request {entry.req.req_id!r}"
                )
            dst_blocks.append(dst)

        def make_callback(req: Request, src: str):
            return lambda block, t, r=req, s=src: self._on_pull_complete(
                block, r, src_key=s, now=t
            )

        existing = shared_pulls.get(dedupe_key)
        if existing is not None:
            for dst, block_hash in zip(dst_blocks, group_hashes):
                existing.add_block(dst, make_callback(entry.req, src_key))
            return

        pull_work = self.work_per_transfer * chunk_transfer_work(
            src_mem, entry.req, group_hashes
        )

        task = BatchLoadTask(
            pull_work,
            link,
            local,
            [dst_blocks[0]],
            on_resident=make_callback(entry.req, src_key),
        )
        for dst in dst_blocks[1:]:
            task.add_block(dst, make_callback(entry.req, src_key))

        pull_task = task
        self.pool.add(task, prereqs)
        for block in dst_blocks:
            block.task = task
        shared_pulls[dedupe_key] = task
        pull_tasks.append(task)
        all_tasks.append(task)

        seen_store: set[tuple[str, str]] = set()
        for block_hash in group_hashes:
            ops = self.placement_policy.plan_async_stores(
                self.memories,
                local_memory=self.local_memory,
                block_hash=block_hash,
                req=entry.req,
            )
            for op in ops:
                dedupe = (op.tier_key, op.content_key)
                if dedupe in seen_store:
                    continue
                seen_store.add(dedupe)
                tier_block = self._tier_block_for_store(op)
                if tier_block is None:
                    continue
                write_link = self.write_links.get(op.tier_key)
                if write_link is None:
                    continue
                tier = self.memories[op.tier_key]
                store_work = self.work_per_store * chunk_transfer_work(
                    tier, entry.req, [block_hash]
                )
                store = StoreTask(
                    work_left=store_work,
                    resource=write_link,
                    tier=tier,
                    tier_block=tier_block,
                    on_resident=lambda b, t, tk=op.tier_key, r=entry.req: self._on_tier_block_resident(
                        tk, b, r, t
                    ),
                )
                self.pool.add(store, [pull_task])
                store_tasks.append(store)
                all_tasks.append(store)

    def execute_batch(self, batch: Batch, now: float = 0.0) -> list[Task]:
        """Reserve memory, sync evict, pulls/stores, then one batched forward."""
        local = self._local()
        all_tasks: list[Task] = []
        evict_tasks: list[Task] = []
        pull_tasks: list[Task] = []
        store_tasks: list[Task] = []
        forward_blocks: list[KVBlock] = []
        forward_block_req: dict[int, Request] = {}
        shared_pulls: dict[tuple[str, str], BatchLoadTask] = {}

        for entry in batch.entries:
            self._record_entry_actions(entry)
            self._reserve(entry.req, entry.result, entry.block_hashes, now=now)

            for victim in entry.result.evicts:
                if self.sync_evict:
                    self._sync_evict_victim(
                        victim,
                        now,
                        spill_req=self._req_for_block_hash(victim.hash),
                        all_tasks=all_tasks,
                    )
                else:
                    task = EvictTask(
                        work_left=self.work_per_evict,
                        resource=self.compute_res,
                        memory=local,
                        block=victim,
                        on_before_evict=lambda b, t, v=victim: self._on_hbm_evict(
                            v, self._req_for_block_hash(v.hash), t
                        ),
                    )
                    self.pool.add(task, [])
                    evict_tasks.append(task)
                    all_tasks.append(task)

            prereqs_tail: list[Task] = list(evict_tasks)

            pull_groups = group_pull_blocks(
                entry.block_hashes,
                entry.result.blocks,
                entry.req,
                self.memories,
            )
            for src_key, group_hashes in pull_groups:
                self._schedule_pull_group(
                    entry,
                    src_key,
                    group_hashes,
                    prereqs_tail,
                    all_tasks,
                    pull_tasks,
                    store_tasks,
                    shared_pulls,
                )

            for block_hash in entry.block_hashes:
                action = entry.result.blocks.get(block_hash)
                if action == "compute":
                    dst_block = local.find_reserved_for(block_hash, entry.req.req_id)
                    if dst_block is None:
                        raise RuntimeError(
                            f"no reserved block for {block_hash!r} "
                            f"request {entry.req.req_id!r}"
                        )
                    forward_blocks.append(dst_block)
                    forward_block_req[id(dst_block)] = entry.req

        if forward_blocks:
            def on_resident(block: KVBlock, t: float) -> None:
                req = forward_block_req[id(block)]
                self._on_block_resident(block, req, t)

            work = batch_forward_work(
                batch,
                work_per_prefill_token=self.work_per_prefill_token,
                work_per_decode_req=self.work_per_decode_req,
            )

            forward = ForwardTask(
                work,
                self.compute_res,
                forward_blocks,
                on_resident=on_resident,
            )
            self.pool.add(forward, evict_tasks + pull_tasks)
            all_tasks.append(forward)

            seen_store: set[tuple[str, str]] = set()
            for block in forward_blocks:
                req = forward_block_req[id(block)]
                ops = self.placement_policy.plan_async_stores(
                    self.memories,
                    local_memory=self.local_memory,
                    block_hash=block.hash,
                    req=req,
                )
                for op in ops:
                    dedupe = (op.tier_key, op.content_key)
                    if dedupe in seen_store:
                        continue
                    seen_store.add(dedupe)
                    tier_block = self._tier_block_for_store(op)
                    if tier_block is None:
                        continue
                    link = self.write_links.get(op.tier_key)
                    if link is None:
                        continue
                    tier = self.memories[op.tier_key]
                    store_work = self.work_per_store * chunk_transfer_work(
                        tier, req, [block.hash]
                    )
                    store = StoreTask(
                        work_left=store_work,
                        resource=link,
                        tier=tier,
                        tier_block=tier_block,
                        on_resident=lambda b, t, tk=op.tier_key, r=req: self._on_tier_block_resident(
                            tk, b, r, t
                        ),
                    )
                    self.pool.add(store, [forward])
                    store_tasks.append(store)
                    all_tasks.append(store)

        return all_tasks

    def apply_batch(self, batch: Batch, now: float) -> tuple[list[Request], list[Request]]:
        """Advance state after the batch finishes (vLLM update_from_output)."""
        finished: list[Request] = []
        remote_kv_done: list[Request] = []

        for entry in batch.entries:
            req = entry.req
            if entry.remote_kv:
                self.scheduler.promote_remote_kv_complete(
                    req, now=now, engine_id=self.engine_id
                )
                remote_kv_done.append(req)
                continue

            if req.pending_block_hash is not None:
                req.block_hashes.append(req.pending_block_hash)
                req.pending_block_hash = None
                req.num_computed_blocks += 1
            else:
                advanced = sum(
                    1
                    for block_hash in entry.block_hashes
                    if entry.result.blocks.get(block_hash) != "wait"
                )
                req.num_computed_blocks += advanced

            if req.num_computed_blocks >= req.blocks_target():
                if req.pd == RequestPD.PREFILL and self.hold_kv_on_complete:
                    self.scheduler.finish_prefill_held(req, now=now)
                else:
                    self.scheduler.finish_request(req, now=now)
                finished.append(req)

        return finished, remote_kv_done

    def release_held_kv(self, req_id: str) -> None:
        self._local().free_request(req_id)

    def _attach_remote_wait(
        self,
        req: Request,
        block_hash: str,
    ) -> None:
        for src_key in self.policy.pull_sources:
            src = self.memories[src_key]
            if tier_covers_hbm_block(src, req, block_hash):
                continue
            if tier_inflight_hbm_block(src, req, block_hash):
                chunk_key = chunk_key_for_hbm_block(req, block_hash, src.chunk_blocks)
                inflight = src.inflight_incoming(chunk_key)
                if inflight is not None:
                    inflight.holders.add(req.req_id)

    def _reserve(
        self,
        req: Request,
        result: LookupResult,
        block_hashes: list[str],
        *,
        now: float = 0.0,
    ) -> None:
        local = self._local()

        for block_hash in block_hashes:
            action = result.blocks.get(block_hash)
            if action == "wait":
                self._attach_remote_wait(req, block_hash)
                continue

            if block_hash in result.blocks:
                local.append_reserved(block_hash, req.req_id)
                continue

            resident = local.best_resident(block_hash)
            if resident is not None:
                resident.holders.add(req.req_id)
                local.touch(resident, now)
                continue

            inflight = local.inflight_incoming(block_hash)
            if inflight is not None:
                inflight.holders.add(req.req_id)
                continue

            if first_resident_pull_source(
                self.memories, self.policy.pull_sources, block_hash, req=req
            ) is None:
                self._attach_remote_wait(req, block_hash)
