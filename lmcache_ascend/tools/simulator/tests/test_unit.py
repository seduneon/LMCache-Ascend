"""Small unit tests for policies, tasks, memory, and scheduler."""

from __future__ import annotations

from chunk_hash import (
    chunk_key_for_hbm_block,
    lmcache_chunk_hash,
    tier_covers_hbm_block,
    transfer_work_units,
)
from memory import BlockState, KVBlock, Memory
from policies import (
    ComputeOnlyLookupPolicy,
    CostBasedPullLookupPolicy,
    ConsumeOnPull,
    FirstAvailableEviction,
    HBMAndDRAM,
    HBMOnly,
    LRUEviction,
    OrderedPullLookupPolicy,
    SingleCopyPerTier,
    UnboundedRetention,
    local_satisfied,
)
from engine import Engine
from request import Request, RequestPD, RequestStatus
from resource import BandwidthResource, ComputeResource
from scheduler import Scheduler
from simulator import Simulator
from tasks import Task, TaskPool, TaskStatus


class _SimpleTask(Task):
    def on_start(self) -> None:
        pass

    def on_end(self) -> None:
        pass


def _make_resident(memory: Memory, block_hash: str, req_id: str = "producer") -> None:
    memory.append_reserved(block_hash, req_id)
    block = memory.find_reserved_for(block_hash, req_id)
    assert block is not None
    block.state = BlockState.RESIDENT


def test_hbm_and_dram_placement_creates_copy() -> None:
    memories = {
        "hbm": Memory(size=2, name="hbm"),
        "dram": Memory(size=10, name="dram"),
    }
    req = Request("r1", 0.0, ["a"], RequestPD.PREFILL, RequestStatus.RUNNING)
    block = memories["hbm"].append_reserved("a", "r1")
    block.state = BlockState.RESIDENT

    HBMAndDRAM("dram").place_copy(
        memories, local_memory="hbm", block=block, req=req, now=1.0
    )

    assert memories["dram"].best_resident("a") is not None

    req2 = Request("r2", 0.0, ["b"], RequestPD.PREFILL, RequestStatus.RUNNING)
    block2 = memories["hbm"].append_reserved("b", "r2")
    block2.state = BlockState.RESIDENT
    HBMOnly().place_copy(
        memories, local_memory="hbm", block=block2, req=req2, now=2.0
    )
    assert memories["dram"].best_resident("b") is None


def test_dram_retains_block_after_hbm_eviction() -> None:
    memories = {
        "hbm": Memory(size=1, name="hbm"),
        "dram": Memory(size=10, name="dram"),
    }
    req = Request("r1", 0.0, ["a"], RequestPD.PREFILL, RequestStatus.RUNNING)
    block = memories["hbm"].append_reserved("a", "r1")
    block.state = BlockState.RESIDENT
    HBMAndDRAM("dram").place_copy(
        memories, local_memory="hbm", block=block, req=req, now=1.0
    )

    memories["hbm"].remove_block(block)
    assert memories["hbm"].best_resident("a") is None

    policy = OrderedPullLookupPolicy(local_memory="hbm", pull_sources=["dram"])
    assert policy.resolve_actions(memories, ["a"], req=req)["a"] == ("pull", "dram")


def test_placement_e2e_pull_from_dram() -> None:
    pool = TaskPool()
    memories = {
        "hbm": Memory(size=2, name="hbm"),
        "dram": Memory(size=10, name="dram"),
    }
    eng = Engine(
        "e0",
        [],
        pool,
        memories,
        "hbm",
        OrderedPullLookupPolicy(local_memory="hbm", pull_sources=["dram"]),
        ComputeResource(base_speed=8.0),
        BandwidthResource(base_speed=8.0),
        work_per_block=1.0,
        placement_policy=HBMAndDRAM("dram"),
    )
    eng.schedule_request(
        Request("r1", 0.0, ["a"], RequestPD.PREFILL, RequestStatus.PENDING)
    )
    eng.schedule_request(
        Request("r2", 1.0, ["b"], RequestPD.PREFILL, RequestStatus.PENDING)
    )
    eng.schedule_request(
        Request("r3", 2.0, ["a"], RequestPD.PREFILL, RequestStatus.PENDING)
    )

    Simulator([eng], pool).run()

    r3 = next(r for r in eng.completed if r.req_id == "r3")
    assert r3.metrics.pulls == 1
    assert memories["dram"].best_resident("a") is not None


def test_dram_lru_eviction_when_tier_full() -> None:
    memories = {
        "hbm": Memory(size=10, name="hbm"),
        "dram": Memory(size=2, name="dram"),
    }
    req = Request("r1", 0.0, ["a"], RequestPD.PREFILL, RequestStatus.RUNNING)
    policy = HBMAndDRAM("dram")

    for name, touch_t in (("a", 1.0), ("b", 2.0), ("c", 3.0)):
        block = KVBlock(name, BlockState.RESIDENT)
        memories["hbm"].append(block)
        policy.place_copy(
            memories, local_memory="hbm", block=block, req=req, now=touch_t
        )

    assert memories["dram"].best_resident("a") is None
    assert memories["dram"].best_resident("b") is not None
    assert memories["dram"].best_resident("c") is not None


def test_spill_on_evict_without_prior_mirror() -> None:
    memories = {
        "hbm": Memory(size=2, name="hbm"),
        "dram": Memory(size=10, name="dram"),
    }
    req = Request("r1", 0.0, ["a"], RequestPD.PREFILL, RequestStatus.RUNNING)
    req.prefix_block_count = 1
    block = KVBlock("a", BlockState.RESIDENT)
    memories["hbm"].append(block)

    HBMAndDRAM("dram").spill_on_evict(
        memories, local_memory="hbm", block=block, now=5.0, req=req
    )
    memories["hbm"].remove_block(block)

    assert memories["hbm"].best_resident("a") is None
    assert memories["dram"].best_resident("a") is not None


def test_lmcache_chunk_hash_groups_aligned_blocks() -> None:
    assert lmcache_chunk_hash(["a"]) == "a"
    assert lmcache_chunk_hash(["a", "b", "c", "d"]) == "chunk:a|b|c|d"


def test_chunk_key_for_hbm_block_aligned_group() -> None:
    req = Request("r1", 0.0, ["a", "b", "c", "d", "e"], RequestPD.PREFILL, RequestStatus.RUNNING)
    req.prefix_block_count = 5
    dram = Memory(size=10, name="dram", chunk_blocks=4)
    assert chunk_key_for_hbm_block(req, "c", dram.chunk_blocks) == "chunk:a|b|c|d"
    assert chunk_key_for_hbm_block(req, "e", dram.chunk_blocks) == "e"


def test_chunked_dram_mirror_and_pull() -> None:
    memories = {
        "hbm": Memory(size=10, name="hbm"),
        "dram": Memory(size=10, name="dram", chunk_blocks=4),
    }
    req = Request("r1", 0.0, ["a", "b", "c", "d"], RequestPD.PREFILL, RequestStatus.RUNNING)
    req.prefix_block_count = 4
    block = KVBlock("b", BlockState.RESIDENT)
    memories["hbm"].append(block)

    HBMAndDRAM("dram").place_copy(
        memories, local_memory="hbm", block=block, req=req, now=1.0
    )
    chunk_key = "chunk:a|b|c|d"
    assert memories["dram"].best_resident(chunk_key) is not None
    assert memories["dram"].best_resident("b") is None

    policy = OrderedPullLookupPolicy(local_memory="hbm", pull_sources=["dram"])
    assert policy.resolve_actions(memories, ["c"], req=req)["c"] == ("pull", "dram")
    assert tier_covers_hbm_block(memories["dram"], req, "c")
    assert transfer_work_units(memories["dram"], req, "c") == 4


def test_chunked_dram_pull_transfer_cost_e2e() -> None:
    pool = TaskPool()
    memories = {
        "hbm": Memory(size=10, name="hbm"),
        "dram": Memory(size=10, name="dram", chunk_blocks=4),
    }
    chunk_key = "chunk:a|b|c|d"
    memories["dram"].append(KVBlock(chunk_key, BlockState.RESIDENT))

    eng = Engine(
        "e0",
        [],
        pool,
        memories,
        "hbm",
        OrderedPullLookupPolicy(local_memory="hbm", pull_sources=["dram"]),
        ComputeResource(base_speed=8.0),
        BandwidthResource(base_speed=4.0),
        work_per_block=1.0,
        work_per_transfer=1.0,
    )
    eng.schedule_request(
        Request("r1", 0.0, ["a", "b", "c", "d"], RequestPD.PREFILL, RequestStatus.PENDING)
    )
    Simulator([eng], pool).run()

    req = eng.completed[0]
    assert req.metrics.pulls == 4
    assert req.metrics.computes == 0


def test_spill_e2e_after_hbm_pressure() -> None:
    pool = TaskPool()
    memories = {
        "hbm": Memory(size=1, name="hbm"),
        "dram": Memory(size=10, name="dram"),
    }
    eng = Engine(
        "e0",
        [],
        pool,
        memories,
        "hbm",
        OrderedPullLookupPolicy(local_memory="hbm", pull_sources=["dram"]),
        ComputeResource(base_speed=8.0),
        BandwidthResource(base_speed=8.0),
        work_per_block=1.0,
        placement_policy=HBMAndDRAM("dram"),
    )
    eng.schedule_request(
        Request("r1", 0.0, ["a"], RequestPD.PREFILL, RequestStatus.PENDING)
    )
    eng.schedule_request(
        Request("r2", 1.0, ["b"], RequestPD.PREFILL, RequestStatus.PENDING)
    )
    eng.schedule_request(
        Request("r3", 2.0, ["a"], RequestPD.PREFILL, RequestStatus.PENDING)
    )

    Simulator([eng], pool).run()

    r3 = next(r for r in eng.completed if r.req_id == "r3")
    assert r3.metrics.pulls == 1
    assert memories["dram"].best_resident("a") is not None


def test_unbounded_retention_allows_duplicate_residents() -> None:
    mem = Memory(size=10, name="hbm")
    mem.append(KVBlock("a", BlockState.RESIDENT))
    mem.append(KVBlock("a", BlockState.RESIDENT))

    UnboundedRetention().on_block_resident(
        {"hbm": mem}, tier_key="hbm", block=mem.resident_copies("a")[-1], now=1.0
    )

    assert len(mem.resident_copies("a")) == 2


def test_single_copy_per_tier_trims_oldest_duplicate() -> None:
    mem = Memory(size=10, name="hbm")
    old = KVBlock("a", BlockState.RESIDENT)
    newer = KVBlock("a", BlockState.RESIDENT)
    mem.append(old)
    mem.append(newer)
    mem.touch(old, 1.0)
    mem.touch(newer, 2.0)

    SingleCopyPerTier().on_block_resident(
        {"hbm": mem}, tier_key="hbm", block=newer, now=2.0
    )

    remaining = mem.resident_copies("a")
    assert len(remaining) == 1
    assert remaining[0] is newer


def test_consume_on_pull_removes_unheld_source() -> None:
    memories = {
        "hbm": Memory(size=10, name="hbm"),
        "dram": Memory(size=10, name="dram"),
    }
    memories["dram"].append(KVBlock("a", BlockState.RESIDENT))

    ConsumeOnPull().after_pull(
        memories,
        src_key="dram",
        dst_key="hbm",
        block_hash="a",
        now=1.0,
    )

    assert memories["dram"].best_resident("a") is None


def test_consume_on_pull_keeps_held_source() -> None:
    memories = {
        "hbm": Memory(size=10, name="hbm"),
        "dram": Memory(size=10, name="dram"),
    }
    held = KVBlock("a", BlockState.RESIDENT, holders={"r1"})
    memories["dram"].append(held)

    ConsumeOnPull().after_pull(
        memories,
        src_key="dram",
        dst_key="hbm",
        block_hash="a",
        now=1.0,
    )

    assert memories["dram"].best_resident("a") is held


def test_consume_on_pull_e2e() -> None:
    pool = TaskPool()
    memories = {
        "hbm": Memory(size=10, name="hbm"),
        "dram": Memory(size=10, name="dram"),
    }
    memories["dram"].append(KVBlock("a", BlockState.RESIDENT))

    eng = Engine(
        "e0",
        [Request("r1", 0.0, ["a"], RequestPD.PREFILL, RequestStatus.PENDING)],
        pool,
        memories,
        "hbm",
        OrderedPullLookupPolicy(local_memory="hbm", pull_sources=["dram"]),
        ComputeResource(base_speed=8.0),
        BandwidthResource(base_speed=8.0),
        work_per_block=1.0,
        hold_kv_on_complete=True,
        retention_policy=ConsumeOnPull(),
    )

    Simulator([eng], pool).run()

    assert memories["hbm"].best_resident("a") is not None
    assert memories["dram"].best_resident("a") is None


def test_lru_eviction_picks_oldest_touch() -> None:
    mem = Memory(size=4, name="hbm")
    old = KVBlock("a", BlockState.RESIDENT)
    mid = KVBlock("b", BlockState.RESIDENT)
    recent = KVBlock("c", BlockState.RESIDENT)
    for block, t in ((old, 1.0), (mid, 5.0), (recent, 9.0)):
        mem.append(block)
        mem.touch(block, t)

    policy = LRUEviction()
    victims = policy.pick_victims(mem, 2, exclude=set())
    assert [v.hash for v in victims] == ["a", "b"]


def test_lru_eviction_skips_held_and_excluded() -> None:
    mem = Memory(size=4, name="hbm")
    held = KVBlock("held", BlockState.RESIDENT, holders={"r1"})
    old = KVBlock("a", BlockState.RESIDENT)
    newer = KVBlock("b", BlockState.RESIDENT)
    mem.append(held)
    mem.append(old)
    mem.append(newer)
    mem.touch(old, 1.0)
    mem.touch(newer, 2.0)

    policy = LRUEviction()
    victims = policy.pick_victims(mem, 1, exclude={"b"})
    assert victims == [old]


def test_lru_eviction_under_allocate_pressure() -> None:
    memories = {"hbm": Memory(size=2, name="hbm")}
    policy = ComputeOnlyLookupPolicy(local_memory="hbm")
    assert isinstance(policy.eviction_policy, LRUEviction)
    sched = Scheduler(policy, memories, "hbm")

    for name, touch_t in (("old", 1.0), ("mid", 5.0)):
        block = KVBlock(name, BlockState.RESIDENT)
        memories["hbm"].append(block)
        memories["hbm"].touch(block, touch_t)

    req = Request("r1", 0.0, ["new"], RequestPD.PREFILL, RequestStatus.RUNNING)
    req.prefix_block_count = 1
    sched.running.append(req)

    result = sched._allocate_blocks(req, ["new"], set(), [], now=10.0)
    assert result is not None
    assert len(result.evicts) == 1
    assert result.evicts[0].hash == "old"


def test_lookup_compute() -> None:
    memories = {"hbm": Memory(size=10, name="hbm")}
    policy = ComputeOnlyLookupPolicy(local_memory="hbm")
    result = policy.lookup(memories, ["a"])
    assert result is not None
    assert result.blocks == {"a": "compute"}
    assert result.evicts == []


def test_pull_only_rejects_compute_fallback() -> None:
    memories = {
        "src": Memory(size=10, name="src"),
        "dst": Memory(size=10, name="dst"),
    }
    _make_resident(memories["src"], "a")
    _make_resident(memories["src"], "b")
    policy = OrderedPullLookupPolicy(local_memory="dst", pull_sources=["src"])

    assert policy.resolve_actions(memories, ["a", "b", "c"])["c"] == "compute"
    assert policy.resolve_actions(memories, ["a", "b", "c"], allow_compute=False) is None


def test_task_prereq_ordering() -> None:
    pool = TaskPool()
    res = ComputeResource(base_speed=1.0)

    first = _SimpleTask(1.0, res)
    second = _SimpleTask(1.0, res)
    pool.add(second, [first])
    pool.add(first, [])

    pool.start_ready(0.0)
    assert first.status == TaskStatus.RUNNING
    assert second.status == TaskStatus.PENDING

    first.advance_to(1.0)
    first.finish()
    pool.start_ready(1.0)
    assert second.status == TaskStatus.RUNNING


def test_prefix_block_count_on_arrival() -> None:
    memories = {"hbm": Memory(size=10, name="hbm")}
    sched = Scheduler(ComputeOnlyLookupPolicy("hbm"), memories, "hbm")
    req = Request("r1", 1.0, ["a", "b", "c"], RequestPD.PREFILL, RequestStatus.PENDING)
    sched.add_request(req)
    sched.release_arrivals(1.0)
    assert req.prefix_block_count == 3
    assert req.status == RequestStatus.WAITING


def test_finish_frees_kv() -> None:
    memories = {"hbm": Memory(size=10, name="hbm")}
    sched = Scheduler(ComputeOnlyLookupPolicy("hbm"), memories, "hbm")
    req = Request("r1", 0.0, ["a"], RequestPD.PREFILL, RequestStatus.RUNNING)
    req.prefix_block_count = 1
    _make_resident(memories["hbm"], "a", "r1")
    sched.running.append(req)

    sched.finish_request(req)
    assert memories["hbm"].used_size() == 0
    assert req in sched.completed


def test_cost_model_picks_faster_pull_source() -> None:
    memories = {
        "fast": Memory(size=10, name="fast"),
        "slow": Memory(size=10, name="slow"),
        "dst": Memory(size=10, name="dst"),
    }
    _make_resident(memories["fast"], "a")
    _make_resident(memories["slow"], "a")

    fast_link = BandwidthResource(base_speed=10.0)
    slow_link = BandwidthResource(base_speed=1.0)
    policy = CostBasedPullLookupPolicy(local_memory="dst", pull_sources=["slow", "fast"])
    policy.bind_resources(
        compute_res=ComputeResource(base_speed=1.0),
        transfer_links={"fast": fast_link, "slow": slow_link},
        work_per_transfer=1.0,
        work_per_block=1.0,
    )

    action = policy.resolve_actions(memories, ["a"])["a"]
    assert action == ("pull", "fast")


def test_cost_model_prefers_compute_under_load() -> None:
    memories = {
        "src": Memory(size=10, name="src"),
        "dst": Memory(size=10, name="dst"),
    }
    _make_resident(memories["src"], "a")

    compute_res = ComputeResource(base_speed=10.0)
    congested_link = BandwidthResource(base_speed=1.0)
    for _ in range(9):
        congested_link.schedule()
        congested_link.start()

    policy = CostBasedPullLookupPolicy(local_memory="dst", pull_sources=["src"])
    policy.bind_resources(
        compute_res=compute_res,
        transfer_links={"src": congested_link},
        work_per_transfer=1.0,
        work_per_block=1.0,
    )

    assert policy.resolve_actions(memories, ["a"])["a"] == "compute"


def test_cost_model_prefill_recompute_expensive() -> None:
    memories = {
        "src": Memory(size=10, name="src"),
        "dst": Memory(size=10, name="dst"),
    }
    _make_resident(memories["src"], "a")

    req = Request("r1", 0.0, ["a"], RequestPD.PREFILL, RequestStatus.RUNNING)
    req.prefix_block_count = 1

    compute_res = ComputeResource(base_speed=1.0)
    link = BandwidthResource(base_speed=10.0, latency=0.0)
    policy = CostBasedPullLookupPolicy(local_memory="dst", pull_sources=["src"])
    policy.bind_resources(
        compute_res=compute_res,
        transfer_links={"src": link},
        work_per_transfer=1.0,
        work_per_block=1.0,
        work_per_prefill_token=100.0,
        work_per_decode_req=1.0,
        block_size=4,
    )

    assert policy.resolve_actions(memories, ["a"], req=req, block_size=4)["a"] == (
        "pull",
        "src",
    )


def test_cost_model_decode_recompute_cheap() -> None:
    memories = {
        "src": Memory(size=10, name="src"),
        "dst": Memory(size=10, name="dst"),
    }
    _make_resident(memories["src"], "a")

    req = Request(
        "r1",
        0.0,
        ["p"],
        RequestPD.DECODE,
        RequestStatus.RUNNING,
        max_output_blocks=1,
    )
    req.prefix_block_count = 1
    req.num_computed_blocks = 1

    compute_res = ComputeResource(base_speed=100.0)
    slow_link = BandwidthResource(base_speed=1.0, latency=0.5)
    policy = CostBasedPullLookupPolicy(local_memory="dst", pull_sources=["src"])
    policy.bind_resources(
        compute_res=compute_res,
        transfer_links={"src": slow_link},
        work_per_transfer=1.0,
        work_per_block=1.0,
        work_per_prefill_token=100.0,
        work_per_decode_req=0.01,
        block_size=1,
    )

    assert policy.resolve_actions(memories, ["a"], req=req)["a"] == "compute"


def test_cost_model_pending_pulls_in_allocation() -> None:
    memories = {
        "src": Memory(size=10, name="src"),
        "dst": Memory(size=10, name="dst"),
    }
    _make_resident(memories["src"], "a")
    _make_resident(memories["src"], "b")

    req = Request("r1", 0.0, ["a", "b"], RequestPD.PREFILL, RequestStatus.RUNNING)
    req.prefix_block_count = 2

    compute_res = ComputeResource(base_speed=100.0)
    link = BandwidthResource(base_speed=1.0, latency=0.0)
    policy = CostBasedPullLookupPolicy(local_memory="dst", pull_sources=["src"])
    policy.bind_resources(
        compute_res=compute_res,
        transfer_links={"src": link},
        work_per_transfer=1.0,
        work_per_block=1.0,
        work_per_prefill_token=100.0,
        block_size=1,
    )

    actions = policy.resolve_actions(memories, ["a", "b"], req=req, block_size=1)
    assert actions["a"] == ("pull", "src")
    assert actions["b"] == "compute"


def test_cost_model_link_scheduled_load() -> None:
    memories = {
        "src": Memory(size=10, name="src"),
        "dst": Memory(size=10, name="dst"),
    }
    _make_resident(memories["src"], "a")

    compute_res = ComputeResource(base_speed=10.0)
    link = BandwidthResource(base_speed=2.0, latency=0.0)
    for _ in range(4):
        link.schedule()

    policy = CostBasedPullLookupPolicy(local_memory="dst", pull_sources=["src"])
    policy.bind_resources(
        compute_res=compute_res,
        transfer_links={"src": link},
        work_per_transfer=1.0,
        work_per_block=1.0,
    )

    assert policy.resolve_actions(memories, ["a"])["a"] == "compute"


def test_cost_model_pull_only_ignores_compute() -> None:
    memories = {
        "src": Memory(size=10, name="src"),
        "dst": Memory(size=10, name="dst"),
    }
    _make_resident(memories["src"], "a")

    compute_res = ComputeResource(base_speed=100.0)
    slow_link = BandwidthResource(base_speed=0.1)

    policy = CostBasedPullLookupPolicy(local_memory="dst", pull_sources=["src"])
    policy.bind_resources(
        compute_res=compute_res,
        transfer_links={"src": slow_link},
        work_per_transfer=1.0,
        work_per_block=1.0,
    )

    assert policy.resolve_actions(memories, ["a"], allow_compute=False) == {
        "a": ("pull", "src")
    }


def test_local_satisfied_inflight() -> None:
    memories = {"hbm": Memory(size=10, name="hbm")}
    local = memories["hbm"]
    block = local.append_reserved("a", "r1")
    block.task = object()  # type: ignore[assignment]
    assert local_satisfied(local, "a")
    policy = ComputeOnlyLookupPolicy("hbm")
    assert policy.resolve_actions(memories, ["a"]) == {}


def test_request_metrics_phases() -> None:
    pool = TaskPool()
    memories = {"hbm": Memory(size=10, name="hbm")}
    req = Request("r1", 1.0, ["a", "b"], RequestPD.PREFILL, RequestStatus.PENDING)
    eng = Engine(
        "e0",
        [req],
        pool,
        memories,
        "hbm",
        ComputeOnlyLookupPolicy(local_memory="hbm"),
        ComputeResource(base_speed=1.0),
        work_per_block=1.0,
    )
    sim = Simulator([eng], pool)

    sim.run()
    m = req.metrics
    assert m.released_at == 1.0
    assert m.finished_at is not None
    assert m.finished_at >= m.released_at
    assert m.latency == m.finished_at - 1.0
    assert m.computes == 2
    assert m.forward_steps >= 1
    assert m.run_time > 0
    assert m.engine_id == "e0"


def test_request_metrics_pd_decode() -> None:
    from pd import PDConfig

    pool = TaskPool()
    memories = {
        "npu-0:hbm": Memory(size=10, name="npu-0:hbm"),
        "npu-1:hbm": Memory(size=10, name="npu-1:hbm"),
    }
    prefill = Request(
        "r1",
        0.0,
        ["a", "b", "c"],
        RequestPD.PREFILL,
        RequestStatus.PENDING,
        max_output_blocks=1,
    )
    npu0 = Engine(
        "npu-0",
        [prefill],
        pool,
        memories,
        "npu-0:hbm",
        ComputeOnlyLookupPolicy(local_memory="npu-0:hbm"),
        ComputeResource(base_speed=4.0),
        work_per_block=1.0,
    )
    npu1 = Engine(
        "npu-1",
        [],
        pool,
        memories,
        "npu-1:hbm",
        CostBasedPullLookupPolicy(local_memory="npu-1:hbm", pull_sources=["npu-0:hbm"]),
        ComputeResource(base_speed=4.0),
        BandwidthResource(base_speed=4.0),
        work_per_transfer=1.0,
        work_per_block=1.0,
        remote_kv_wait=True,
    )
    sim = Simulator([npu0, npu1], pool, pd=PDConfig(spawn_map={"npu-0": "npu-1"}))
    sim.run()

    decode = next(r for r in npu1.completed if r.req_id == "r1")
    pm = prefill.metrics
    dm = decode.metrics

    assert pm.computes == 3
    assert pm.pulls == 0
    assert pm.finished_at is not None

    assert dm.remote_kv_admits == 1
    assert dm.pulls == 3
    assert dm.remote_kv_time >= 0
    assert dm.computes == 1
    assert dm.finished_at is not None
    assert dm.engine_id == "npu-1"


def test_sweep_smoke() -> None:
    from sweep import SweepConfig, run_sweep

    rows = run_sweep(
        SweepConfig(
            presets=("baseline", "ordered_pull"),
            num_requests=8,
            seeds=1,
            base_seed=42,
        )
    )
    assert len(rows) == 2
    assert all(r.status == "ok" for r in rows)
    assert rows[0].decode_p99_latency >= 0


def run_unit_tests() -> None:
    tests = [
        # placement + spill
        test_hbm_and_dram_placement_creates_copy,
        test_dram_retains_block_after_hbm_eviction,
        test_placement_e2e_pull_from_dram,
        test_dram_lru_eviction_when_tier_full,
        test_spill_on_evict_without_prior_mirror,
        test_lmcache_chunk_hash_groups_aligned_blocks,
        test_chunk_key_for_hbm_block_aligned_group,
        test_chunked_dram_mirror_and_pull,
        test_chunked_dram_pull_transfer_cost_e2e,
        test_spill_e2e_after_hbm_pressure,
        # retention
        test_unbounded_retention_allows_duplicate_residents,
        test_single_copy_per_tier_trims_oldest_duplicate,
        test_consume_on_pull_removes_unheld_source,
        test_consume_on_pull_keeps_held_source,
        test_consume_on_pull_e2e,
        # eviction
        test_lru_eviction_picks_oldest_touch,
        test_lru_eviction_skips_held_and_excluded,
        test_lru_eviction_under_allocate_pressure,
        # lookup / cost model
        test_lookup_compute,
        test_pull_only_rejects_compute_fallback,
        test_cost_model_picks_faster_pull_source,
        test_cost_model_prefers_compute_under_load,
        test_cost_model_prefill_recompute_expensive,
        test_cost_model_decode_recompute_cheap,
        test_cost_model_pending_pulls_in_allocation,
        test_cost_model_link_scheduled_load,
        test_cost_model_pull_only_ignores_compute,
        # scheduler / tasks / memory
        test_task_prereq_ordering,
        test_prefix_block_count_on_arrival,
        test_finish_frees_kv,
        test_local_satisfied_inflight,
        # metrics
        test_request_metrics_phases,
        test_request_metrics_pd_decode,
        test_sweep_smoke,
    ]
    for test in tests:
        test()
        print(f"{test.__name__} ok")
