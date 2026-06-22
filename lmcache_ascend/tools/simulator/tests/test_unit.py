"""Small unit tests for policies, tasks, memory, and scheduler."""

from __future__ import annotations

from simulator.plan import EntryPlan, WorkEntry
from simulator.eviction import FIFOEviction, LRUEviction, RandomEviction
from simulator.schedule import local_satisfied
from simulator.engine_config import placement_edges
from simulator.placement import HBMOnly, TieredPlacement
from simulator.tier import graph_from_memories
from simulator.retention import (
    ConsumeOnPull,
    GlobalCopyCap,
    SingleCopyPerTier,
    UnboundedRetention,
    consume_on_pull_retention,
)
from simulator.kv_content import (
    ContentKey,
    content_id,
    storage_key,
    tier_has_block,
    transfer_block_count,
)
from simulator.memory import BlockState, KVBlock, Memory, collect_content_copies
from simulator.tasks import BatchLoadTask, ForwardTask, StoreTask, TaskPool, TaskStatus
from simulator.tests.test_helpers import (
    SimpleTask,
    make_resident,
    policies_compute,
    policies_pull,
    schedule_compute,
    schedule_pull,
)

from simulator.engine import Engine
from simulator.request import Request, RequestPD, RequestStatus
from simulator.resource import BandwidthResource, ComputeResource
from simulator.scheduler import Scheduler
from simulator.simulator import Simulator


def _dram_placement(memories: dict[str, Memory]) -> TieredPlacement:
    graph = graph_from_memories(memories)
    return TieredPlacement(graph, placement_edges(("dram",)))


def test_hbm_and_dram_placement_creates_copy() -> None:
    memories = {
        "hbm": Memory(size=2, name="hbm"),
        "dram": Memory(size=10, name="dram"),
    }
    req = Request("r1", 0.0, ["a"], RequestPD.PREFILL, RequestStatus.RUNNING)
    block = memories["hbm"].append_reserved("a", "r1")
    block.state = BlockState.RESIDENT

    _dram_placement(memories).place_copy(
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
    _dram_placement(memories).place_copy(
        memories, local_memory="hbm", block=block, req=req, now=1.0
    )

    memories["hbm"].remove_block(block)
    assert memories["hbm"].best_resident("a") is None

    policy = schedule_pull(local_memory="hbm", pull_sources=["dram"])
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
        policies_pull("hbm", ["dram"], mirror_tiers=("dram",)),
        ComputeResource(base_speed=8.0),
        BandwidthResource(base_speed=8.0),
        work_per_block=1.0,
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
    policy = _dram_placement(memories)

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

    _dram_placement(memories).spill_on_evict(
        memories, local_memory="hbm", block=block, now=5.0, req=req
    )
    memories["hbm"].remove_block(block)

    assert memories["hbm"].best_resident("a") is None
    assert memories["dram"].best_resident("a") is not None


def test_content_id_groups_aligned_blocks() -> None:
    assert content_id(["a"]) == "a"
    assert content_id(["a", "b", "c", "d"]) == "chunk:a|b|c|d"


def test_storage_key_aligned_group() -> None:
    req = Request("r1", 0.0, ["a", "b", "c", "d", "e"], RequestPD.PREFILL, RequestStatus.RUNNING)
    req.prefix_block_count = 5
    dram = Memory(size=10, name="dram", chunk_blocks=4)
    assert storage_key(req, "c", dram.chunk_blocks) == "chunk:a|b|c|d"
    assert storage_key(req, "e", dram.chunk_blocks) == "e"


def test_chunked_dram_mirror_and_pull() -> None:
    memories = {
        "hbm": Memory(size=10, name="hbm"),
        "dram": Memory(size=10, name="dram", chunk_blocks=4),
    }
    req = Request("r1", 0.0, ["a", "b", "c", "d"], RequestPD.PREFILL, RequestStatus.RUNNING)
    req.prefix_block_count = 4
    block = KVBlock("b", BlockState.RESIDENT)
    memories["hbm"].append(block)

    _dram_placement(memories).place_copy(
        memories, local_memory="hbm", block=block, req=req, now=1.0
    )
    chunk_key = "chunk:a|b|c|d"
    assert memories["dram"].best_resident(chunk_key) is not None
    assert memories["dram"].best_resident("b") is None

    policy = schedule_pull(local_memory="hbm", pull_sources=["dram"])
    assert policy.resolve_actions(memories, ["c"], req=req)["c"] == ("pull", "dram")
    assert tier_has_block(memories["dram"], req, "c")
    assert transfer_block_count(memories["dram"], req, "c") == 4


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
        policies_pull("hbm", ["dram"]),
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
        policies_pull("hbm", ["dram"], mirror_tiers=("dram",)),
        ComputeResource(base_speed=8.0),
        BandwidthResource(base_speed=8.0),
        work_per_block=1.0,
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
        policies_pull("hbm", ["dram"], retention=consume_on_pull_retention),
        ComputeResource(base_speed=8.0),
        BandwidthResource(base_speed=8.0),
        work_per_block=1.0,
        hold_kv_on_complete=True,
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


def test_fifo_eviction_picks_oldest_insert() -> None:
    mem = Memory(size=4, name="hbm")
    first = KVBlock("a", BlockState.RESIDENT)
    second = KVBlock("b", BlockState.RESIDENT)
    third = KVBlock("c", BlockState.RESIDENT)
    for block in (first, second, third):
        mem.append(block)

    policy = FIFOEviction()
    victims = policy.pick_victims(mem, 2, exclude=set())
    assert [v.hash for v in victims] == ["a", "b"]


def test_random_eviction_is_seeded() -> None:
    mem = Memory(size=4, name="hbm")
    for name in ("a", "b", "c", "d"):
        mem.append(KVBlock(name, BlockState.RESIDENT))

    a = RandomEviction(seed=7).pick_victims(mem, 2, exclude=set())
    b = RandomEviction(seed=7).pick_victims(mem, 2, exclude=set())
    c = RandomEviction(seed=8).pick_victims(mem, 2, exclude=set())
    assert [v.hash for v in a] == [v.hash for v in b]
    assert len({v.hash for v in a}) == 2
    assert {v.hash for v in c} <= {"a", "b", "c", "d"}


def test_fifo_eviction_under_allocate_pressure() -> None:
    memories = {"hbm": Memory(size=2, name="hbm")}
    policy = schedule_compute(
        local_memory="hbm",
        eviction_policy=FIFOEviction(),
    )
    sched = Scheduler(policy, memories, "hbm")

    for name in ("old", "mid"):
        block = KVBlock(name, BlockState.RESIDENT)
        memories["hbm"].append(block)

    req = Request("r1", 0.0, ["new"], RequestPD.PREFILL, RequestStatus.RUNNING)
    req.prefix_block_count = 1
    sched.running.append(req)

    result = sched._allocate_blocks(req, ["new"], set(), [])
    assert result is not None
    assert len(result.evicts) == 1
    assert result.evicts[0].hash == "old"


def test_lru_eviction_under_allocate_pressure() -> None:
    memories = {"hbm": Memory(size=2, name="hbm")}
    policy = schedule_compute(local_memory="hbm")
    assert isinstance(policy.eviction_policy, LRUEviction)
    sched = Scheduler(policy, memories, "hbm")

    for name, touch_t in (("old", 1.0), ("mid", 5.0)):
        block = KVBlock(name, BlockState.RESIDENT)
        memories["hbm"].append(block)
        memories["hbm"].touch(block, touch_t)

    req = Request("r1", 0.0, ["new"], RequestPD.PREFILL, RequestStatus.RUNNING)
    req.prefix_block_count = 1
    sched.running.append(req)

    result = sched._allocate_blocks(req, ["new"], set(), [])
    assert result is not None
    assert len(result.evicts) == 1
    assert result.evicts[0].hash == "old"


def test_lookup_compute() -> None:
    memories = {"hbm": Memory(size=10, name="hbm")}
    policy = schedule_compute(local_memory="hbm")
    result = policy.lookup(memories, ["a"])
    assert result is not None
    assert result.blocks == {"a": "compute"}
    assert result.evicts == []


def test_min_cost_pull_prefers_dram_when_compute_expensive() -> None:
    from simulator.estimate import CostContext
    from simulator.read_path import ReadPathSpec
    from simulator.resource import BandwidthResource, ComputeResource
    from simulator.schedule import ScheduleConfig, SchedulePolicy

    memories = {
        "hbm": Memory(size=10, name="hbm"),
        "dram": Memory(size=10, name="dram"),
    }
    make_resident(memories["dram"], "a")
    policy = SchedulePolicy(
        ScheduleConfig(
            local_memory="hbm",
            pull_sources=("dram",),
            read_path=ReadPathSpec(kind="min_cost"),
        )
    )
    req = Request("r1", 0.0, ["a"], RequestPD.DECODE, RequestStatus.RUNNING)
    req.prefix_block_count = 1
    cost_ctx = CostContext(
        memories=memories,
        transfer_links={"dram": BandwidthResource(base_speed=100.0)},
        compute_res=ComputeResource(base_speed=1.0),
        work_per_transfer=1.0,
        work_per_prefill_token=1.0,
        work_per_decode_req=1000.0,
    )
    action = policy.resolve_block(
        memories, "a", allow_compute=True, req=req, cost_ctx=cost_ctx
    )
    assert action == ("pull", "dram")


def test_threshold_pull_when_cheaper() -> None:
    from simulator.estimate import CostContext
    from simulator.read_path import ReadPathSpec
    from simulator.resource import BandwidthResource, ComputeResource
    from simulator.schedule import ScheduleConfig, SchedulePolicy

    memories = {
        "hbm": Memory(size=10, name="hbm"),
        "dram": Memory(size=10, name="dram"),
    }
    make_resident(memories["dram"], "a")
    policy = SchedulePolicy(
        ScheduleConfig(
            local_memory="hbm",
            pull_sources=("dram",),
            read_path=ReadPathSpec(kind="threshold", threshold_ratio=0.5),
        )
    )
    req = Request("r1", 0.0, ["a"], RequestPD.DECODE, RequestStatus.RUNNING)
    req.prefix_block_count = 1
    cost_ctx = CostContext(
        memories=memories,
        transfer_links={"dram": BandwidthResource(base_speed=100.0)},
        compute_res=ComputeResource(base_speed=10.0),
        work_per_transfer=1.0,
        work_per_prefill_token=1.0,
        work_per_decode_req=10.0,
    )
    action = policy.resolve_block(
        memories, "a", allow_compute=True, req=req, cost_ctx=cost_ctx
    )
    assert action == ("pull", "dram")


def test_pull_only_rejects_compute_fallback() -> None:
    memories = {
        "src": Memory(size=10, name="src"),
        "dst": Memory(size=10, name="dst"),
    }
    make_resident(memories["src"], "a")
    make_resident(memories["src"], "b")
    policy = schedule_pull(local_memory="dst", pull_sources=["src"])

    assert policy.resolve_actions(memories, ["a", "b", "c"])["c"] == "compute"
    assert policy.resolve_actions(memories, ["a", "b", "c"], allow_compute=False) is None


def test_task_prereq_ordering() -> None:
    pool = TaskPool()
    res = ComputeResource(base_speed=1.0)

    first = SimpleTask(1.0, res)
    second = SimpleTask(1.0, res)
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
    sched = Scheduler(schedule_compute("hbm"), memories, "hbm")
    req = Request("r1", 1.0, ["a", "b", "c"], RequestPD.PREFILL, RequestStatus.PENDING)
    sched.add_request(req)
    sched.release_arrivals(1.0)
    assert req.prefix_block_count == 3
    assert req.status == RequestStatus.WAITING


def test_finish_frees_kv() -> None:
    memories = {"hbm": Memory(size=10, name="hbm")}
    sched = Scheduler(schedule_compute("hbm"), memories, "hbm")
    req = Request("r1", 0.0, ["a"], RequestPD.PREFILL, RequestStatus.RUNNING)
    req.prefix_block_count = 1
    make_resident(memories["hbm"], "a", "r1")
    sched.running.append(req)

    sched.finish_request(req)
    assert memories["hbm"].used_size() == 0
    assert req in sched.completed


def test_local_satisfied_inflight() -> None:
    memories = {"hbm": Memory(size=10, name="hbm")}
    local = memories["hbm"]
    block = local.append_reserved("a", "r1")
    block.task = object()  # type: ignore[assignment]
    assert local_satisfied(local, "a")
    policy = schedule_compute("hbm")
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
        policies_compute("hbm"),
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
    from simulator.pd import PDConfig

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
        policies_compute("npu-0:hbm"),
        ComputeResource(base_speed=4.0),
        work_per_block=1.0,
    )
    npu1 = Engine(
        "npu-1",
        [],
        pool,
        memories,
        policies_pull("npu-1:hbm", ["npu-0:hbm"]),
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
    assert pm.prefix_computes == 3
    assert pm.pulls == 0
    assert pm.finished_at is not None

    assert dm.remote_kv_admits == 1
    assert dm.pulls == 3
    assert dm.prefix_pulls == 3
    assert dm.prefix_computes == 0
    assert dm.remote_kv_time >= 0
    assert dm.computes == 1
    assert dm.finished_at is not None
    assert dm.engine_id == "npu-1"


def test_prefix_entry_metrics() -> None:
    from simulator.estimate import is_prefix_block, record_entry_metrics
    from simulator.plan import EntryPlan, WorkEntry

    decode = Request(
        "d1",
        0.0,
        ["p0", "p1", "blk:d1:0"],
        RequestPD.DECODE,
        RequestStatus.RUNNING,
        prefix_block_count=2,
        max_output_blocks=1,
    )
    assert is_prefix_block(decode, "p0")
    assert not is_prefix_block(decode, "blk:d1:0")

    entry = WorkEntry(
        decode,
        ["p0", "blk:d1:0"],
        EntryPlan(
            blocks={
                "p0": ("pull", "npu-0:dram"),
                "blk:d1:0": "compute",
            }
        ),
        num_scheduled_tokens=1,
    )
    record_entry_metrics(entry)
    m = decode.metrics
    assert m.pulls == 1
    assert m.prefix_pulls == 1
    assert m.prefix_dram_pulls == 1
    assert m.computes == 1
    assert m.prefix_computes == 0


def test_global_copy_cap_trims_across_tiers() -> None:
    memories = {
        "hbm": Memory(size=10, name="hbm"),
        "other": Memory(size=10, name="other"),
    }
    req = Request("r1", 0.0, ["a"], RequestPD.PREFILL, RequestStatus.RUNNING)
    memories["hbm"].append(KVBlock("a", BlockState.RESIDENT))
    memories["other"].append(KVBlock("a", BlockState.RESIDENT))
    memories["other"].append(KVBlock("a", BlockState.RESIDENT))

    cap = GlobalCopyCap(2, ["hbm", "other"], per_tier_cap=None)
    cap.on_block_resident(
        memories,
        tier_key="other",
        block=memories["other"].list()[-1],
        now=1.0,
        req=req,
    )

    total = sum(len(mem.resident_copies("a")) for mem in memories.values())
    assert total == 2


def test_tiered_placement_async_store_e2e() -> None:
    pool = TaskPool()
    memories = {
        "hbm": Memory(size=10, name="hbm"),
        "ssd": Memory(size=10, name="ssd", chunk_blocks=4),
    }
    write_link = BandwidthResource(base_speed=4.0, latency=0.0)
    eng = Engine(
        "e0",
        [],
        pool,
        memories,
        policies_compute("hbm", async_write_tiers=frozenset({"ssd"})),
        ComputeResource(base_speed=8.0),
        work_per_block=1.0,
        write_links={"ssd": write_link},
        work_per_store=4.0,
    )
    eng.schedule_request(
        Request("r1", 0.0, ["a", "b", "c", "d"], RequestPD.PREFILL, RequestStatus.PENDING)
    )
    Simulator([eng], pool).run()

    chunk_key = "chunk:a|b|c|d"
    assert memories["ssd"].best_resident(chunk_key) is not None
    assert write_link.queued_load() == 0


def test_inflight_remote_source_waits_not_preempts() -> None:
    memories = {
        "hbm": Memory(size=10, name="hbm"),
        "src": Memory(size=10, name="src", chunk_blocks=4),
    }
    chunk_key = "chunk:a|b|c|d"
    loading = KVBlock(chunk_key, BlockState.LOADING)
    loading.task = object()  # type: ignore[assignment]
    memories["src"].append(loading)

    req = Request("r1", 0.0, ["a", "b", "c", "d"], RequestPD.PREFILL, RequestStatus.RUNNING)
    req.prefix_block_count = 4
    policy = schedule_pull(local_memory="hbm", pull_sources=["src"])
    actions = policy.resolve_actions(memories, ["c"], req=req)
    assert actions == {"c": "wait"}


def test_batch_load_task_amortizes_work() -> None:
    pool = TaskPool()
    memories = {
        "hbm": Memory(size=10, name="hbm"),
        "dram": Memory(size=10, name="dram", chunk_blocks=4),
    }
    chunk_key = "chunk:a|b|c|d"
    memories["dram"].append(KVBlock(chunk_key, BlockState.RESIDENT))

    link = BandwidthResource(base_speed=1.0, latency=1.0)
    eng = Engine(
        "e0",
        [],
        pool,
        memories,
        policies_pull("hbm", ["dram"]),
        ComputeResource(base_speed=100.0),
        transfer_links={"dram": link},
        work_per_block=1.0,
        work_per_transfer=1.0,
    )
    eng.schedule_request(
        Request("r1", 0.0, ["a", "b", "c", "d"], RequestPD.PREFILL, RequestStatus.PENDING)
    )
    eng.release_arrivals(0.0)
    plan = eng.try_schedule_and_execute(0.0)
    assert plan is not None
    tasks = [t for t in pool.tasks if t.batch_id == plan.batch_id]
    assert any(isinstance(t, BatchLoadTask) for t in tasks)

    t0 = Simulator([eng], pool).run()
    assert t0 == 5.0


def test_pull_dedupe_across_requests_in_batch() -> None:
    pool = TaskPool()
    memories = {
        "hbm": Memory(size=20, name="hbm"),
        "dram": Memory(size=10, name="dram", chunk_blocks=4),
    }
    chunk_key = "chunk:a|b|c|d"
    memories["dram"].append(KVBlock(chunk_key, BlockState.RESIDENT))

    link = BandwidthResource(base_speed=1.0, latency=1.0)
    eng = Engine(
        "e0",
        [],
        pool,
        memories,
        policies_pull("hbm", ["dram"]),
        ComputeResource(base_speed=100.0),
        transfer_links={"dram": link},
        work_per_block=1.0,
        work_per_transfer=1.0,
        max_num_seqs=4,
        max_num_batched_tokens=32,
    )
    eng.schedule_request(
        Request("r1", 0.0, ["a", "b", "c", "d"], RequestPD.PREFILL, RequestStatus.PENDING)
    )
    eng.schedule_request(
        Request("r2", 0.0, ["a", "b", "c", "d"], RequestPD.PREFILL, RequestStatus.PENDING)
    )
    eng.release_arrivals(0.0)
    plan = eng.try_schedule_and_execute(0.0)
    assert plan is not None
    assert len(plan.entries) == 2

    tasks = [t for t in pool.tasks if t.batch_id == plan.batch_id]
    batch_loads = [t for t in tasks if isinstance(t, BatchLoadTask)]
    assert len(batch_loads) == 1
    assert len(batch_loads[0].blocks) == 8

    finish = Simulator([eng], pool).run()
    assert finish == 5.0
    assert eng.completed[0].metrics.pulls == 4
    assert eng.completed[1].metrics.pulls == 4


def test_sync_evict_frees_before_pull() -> None:
    pool = TaskPool()
    memories = {"hbm": Memory(size=1, name="hbm")}
    memories["hbm"].append(KVBlock("old", BlockState.RESIDENT))
    eng = Engine(
        "e0",
        [],
        pool,
        memories,
        policies_compute("hbm"),
        ComputeResource(base_speed=8.0),
        work_per_block=1.0,
        sync_evict=True,
    )
    eng.schedule_request(
        Request("r1", 0.0, ["new"], RequestPD.PREFILL, RequestStatus.PENDING)
    )
    eng.release_arrivals(0.0)
    plan = eng.try_schedule_and_execute(0.0)
    assert plan is not None
    assert memories["hbm"].best_resident("old") is None
    assert memories["hbm"].find_reserved_for("new", "r1") is not None


def test_content_key_same_across_tiers() -> None:
    req = Request("r1", 0.0, ["a", "b", "c", "d"], RequestPD.PREFILL, RequestStatus.RUNNING)
    req.prefix_block_count = 4
    hbm = Memory(size=10, name="hbm", chunk_blocks=1)
    dram = Memory(size=10, name="dram", chunk_blocks=4)

    content = ContentKey.from_block("c")
    assert str(content) == "c"
    assert storage_key(req, "c", hbm.chunk_blocks) == "c"
    assert storage_key(req, "c", dram.chunk_blocks) == "chunk:a|b|c|d"

    dram.append(KVBlock("chunk:a|b|c|d", BlockState.RESIDENT))
    copies = collect_content_copies({"hbm": hbm, "dram": dram}, ["hbm", "dram"], content, req=req)
    assert len(copies) == 1
    assert copies[0][0] == "dram"


def test_global_copy_cap_ssd_chunk_tier() -> None:
    memories = {
        "hbm": Memory(size=10, name="hbm", chunk_blocks=1),
        "dram": Memory(size=10, name="dram", chunk_blocks=4),
        "other": Memory(size=10, name="other", chunk_blocks=1),
    }
    req = Request("r1", 0.0, ["a", "b", "c", "d"], RequestPD.PREFILL, RequestStatus.RUNNING)
    req.prefix_block_count = 4
    memories["hbm"].append(KVBlock("a", BlockState.RESIDENT))
    memories["dram"].append(KVBlock("chunk:a|b|c|d", BlockState.RESIDENT))
    memories["other"].append(KVBlock("a", BlockState.RESIDENT))
    memories["other"].append(KVBlock("a", BlockState.RESIDENT))

    cap = GlobalCopyCap(2, ["hbm", "dram", "other"], per_tier_cap=None)
    cap.on_block_resident(
        memories,
        tier_key="other",
        block=memories["other"].list()[-1],
        now=1.0,
        req=req,
    )

    content = ContentKey.from_block("a")
    copies = collect_content_copies(memories, ["hbm", "dram", "other"], content, req=req)
    assert len(copies) == 2


def test_execute_plan_tags_pool_tasks() -> None:
    pool = TaskPool()
    memories = {
        "hbm": Memory(size=10, name="hbm"),
        "ssd": Memory(size=10, name="ssd", chunk_blocks=4),
    }
    write_link = BandwidthResource(base_speed=4.0, latency=0.0)
    eng = Engine(
        "e0",
        [],
        pool,
        memories,
        policies_compute("hbm", async_write_tiers=frozenset({"ssd"})),
        ComputeResource(base_speed=8.0),
        work_per_block=1.0,
        write_links={"ssd": write_link},
        work_per_store=4.0,
    )
    eng.schedule_request(
        Request("r1", 0.0, ["a", "b", "c", "d"], RequestPD.PREFILL, RequestStatus.PENDING)
    )
    eng.release_arrivals(0.0)
    plan = eng.try_schedule_and_execute(0.0)
    assert plan is not None

    tagged = [t for t in pool.tasks if t.batch_id == plan.batch_id]
    assert tagged
    assert all(t.batch_id == plan.batch_id for t in tagged)


def test_batch_complete_waits_for_tagged_tasks() -> None:
    pool = TaskPool()
    sim = Simulator([], pool)
    compute = ComputeResource(base_speed=10.0, latency=0.0)
    write = BandwidthResource(base_speed=4.0, latency=0.0)
    hbm = Memory(size=10, name="hbm")
    ssd = Memory(size=10, name="ssd")
    reserved = KVBlock("a", BlockState.RESERVED)
    hbm.append(reserved)
    tier_block = KVBlock("a", BlockState.RESERVED)
    ssd.append(tier_block)

    forward = ForwardTask(2.0, compute, [reserved], {})
    store = StoreTask(4.0, write, ssd, tier_block)
    pool.add(forward, [], batch_id=5)
    pool.add(store, [forward], batch_id=5)

    assert not sim._batch_complete(5)
    pool.start_ready(0.0)
    for _ in range(10):
        if sim._batch_complete(5):
            break
        running = pool.running()
        if not running:
            pool.start_ready(10.0)
            continue
        pool.advance_running_to(min(t.estimated_end() for t in running))
        pool.finish_done()
        pool.start_ready(10.0)

    assert sim._batch_complete(5)


def test_sweep_smoke() -> None:
    import csv
    import tempfile
    from pathlib import Path

    from simulator.sweep import SweepConfig, run_sweep

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = str(Path(tmp) / "compare.csv")
        rows = run_sweep(
            SweepConfig(
                presets=("baseline", "ordered_pull"),
                num_requests=8,
                seeds=1,
                base_seed=42,
                csv_path=csv_path,
            )
        )
        assert len(rows) == 2
        assert all(r.status == "ok" for r in rows)
        assert rows[0].decode_p99_latency >= 0

        with open(csv_path, newline="", encoding="utf-8") as handle:
            table = list(csv.reader(handle))
        assert table[0][0] == "metric"
        assert table[0][1:] == ["baseline", "ordered_pull"]
        metrics = {row[0]: row[1:] for row in table[1:]}
        assert metrics["status"] == ["ok", "ok"]
        assert "decode_p99_latency" in metrics


def test_blocks_from_gib() -> None:
    from simulator.capacity import blocks_from_gib, gib_for_blocks

    gib = 32.0
    blocks = blocks_from_gib(
        gib,
        tokens_per_block=512,
        kv_bytes_per_token=256.0,
    )
    assert blocks >= 1
    assert abs(gib_for_blocks(blocks, tokens_per_block=512, kv_bytes_per_token=256.0) - gib) < 0.01


def test_eviction_preset_sweep_smoke() -> None:
    from simulator.presets import EVICTION_PRESET_NAMES
    from simulator.sweep import SimConfig, SweepConfig, run_sweep

    rows = run_sweep(
        SweepConfig(
            presets=EVICTION_PRESET_NAMES,
            num_requests=12,
            seeds=1,
            base_seed=42,
            sim=SimConfig.with_block_slots(hbm=24, dram=32),
        )
    )
    assert len(rows) == len(EVICTION_PRESET_NAMES)
    assert all(r.status == "ok" for r in rows)
    by_name = {r.preset: r for r in rows}
    assert by_name["evict_lru"].preemptions >= 0
    assert by_name["evict_lru"].lifecycle_hbm_frees >= 0
    assert by_name["evict_lru"].tier_evictions >= 0


def test_load_mooncake_trace() -> None:
    import json

    from simulator.mooncake_trace import DEFAULT_TRACE_PATH, load_mooncake_trace
    from simulator.workload import WorkloadConfig

    cfg = WorkloadConfig(
        num_requests=8,
        trace_path=str(DEFAULT_TRACE_PATH),
        trace_offset=50,
    )
    requests, shared_pool = load_mooncake_trace(cfg)
    assert len(requests) == 8
    assert requests[0].req_id == "trace:50"
    assert requests[0].prefix_block_count == len(requests[0].block_hashes)
    assert all(h.startswith("mooncake:") for h in requests[0].block_hashes)
    assert requests[0].max_output_blocks >= 1

    with open(DEFAULT_TRACE_PATH, encoding="utf-8") as handle:
        for _ in range(50):
            handle.readline()
        record = json.loads(handle.readline())
    assert requests[0].max_output_blocks == max(
        1,
        (int(record["output_length"]) + cfg.tokens_per_block - 1)
        // cfg.tokens_per_block,
    )
    assert len(shared_pool) >= len(requests[0].block_hashes)

    arrivals = [r.arrival_time for r in requests]
    assert arrivals == sorted(arrivals)


def test_partition_by_hbm() -> None:
    from simulator.mooncake_trace import (
        DEFAULT_TRACE_PATH,
        is_hbm_admittable,
        partition_by_hbm,
        request_block_footprint,
    )
    from simulator.workload import WorkloadConfig, build_workload

    reqs, _ = build_workload(
        WorkloadConfig(num_requests=500, trace_path=str(DEFAULT_TRACE_PATH))
    )
    hbm = 128
    admittable, rejected = partition_by_hbm(reqs, hbm)
    assert len(admittable) + len(rejected) == len(reqs)
    assert all(is_hbm_admittable(r, hbm) for r in admittable)
    assert all(not is_hbm_admittable(r, hbm) for r in rejected)
    assert len(rejected) >= 1
    assert max(request_block_footprint(r) for r in rejected) > hbm


def test_drop_oversized_sweep_smoke() -> None:
    from simulator.sweep import SimConfig, SweepConfig, run_sweep
    from simulator.mooncake_trace import DEFAULT_TRACE_PATH

    rows = run_sweep(
        SweepConfig(
            presets=("baseline",),
            num_requests=500,
            seeds=1,
            base_seed=42,
            trace_path=str(DEFAULT_TRACE_PATH),
            drop_oversized=True,
            sim=SimConfig.with_block_slots(
                hbm=128,
                kv_bytes_per_token=256.0,
                max_num_seqs=32,
                max_num_batched_tokens=128,
            ),
        )
    )
    assert len(rows) == 1
    assert rows[0].status == "ok", rows[0].error
    assert rows[0].num_requests == 500
    assert rows[0].rejected_requests >= 1
    assert rows[0].admittable_requests == 500 - rows[0].rejected_requests


def test_trace_sweep_smoke() -> None:
    from simulator.sweep import SimConfig, SweepConfig, run_sweep
    from simulator.mooncake_trace import DEFAULT_TRACE_PATH

    rows = run_sweep(
        SweepConfig(
            presets=("baseline",),
            num_requests=12,
            seeds=1,
            base_seed=42,
            trace_path=str(DEFAULT_TRACE_PATH),
            drop_oversized=True,
            sim=SimConfig.with_block_slots(
                hbm=128,
                kv_bytes_per_token=256.0,
                max_num_seqs=32,
                max_num_batched_tokens=128,
            ),
        )
    )
    assert len(rows) == 1
    assert rows[0].status == "ok", rows[0].error
    assert rows[0].num_requests == 12


def test_event_trace_analyzer_roundtrip() -> None:
    import os
    import tempfile
    from pathlib import Path

    from simulator.analyze import analyze, load_events
    from simulator.presets import PRESETS
    from simulator.sweep import SimConfig, SweepConfig, run_sweep_case
    from simulator.workload import WorkloadConfig

    with tempfile.TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "trace.jsonl"
        os.environ["SIM_TRACE"] = "1"
        os.environ["SIM_TRACE_PATH"] = str(trace_path)
        try:
            row = run_sweep_case(
                PRESETS["min_cost_pull"],
                WorkloadConfig(num_requests=8, seed=42),
                SimConfig.with_block_slots(hbm=16, dram=32),
                sweep_cfg=SweepConfig(read_path="min_cost"),
            )
            assert row.status == "ok", row.error
            assert trace_path.is_file()
            events = load_events(trace_path)
            summary = analyze(events)
            assert summary["event_count"] > 0
            assert summary["decisions"]
        finally:
            os.environ.pop("SIM_TRACE", None)
            os.environ.pop("SIM_TRACE_PATH", None)


def test_policy_matrix_sweep_smoke() -> None:
    from simulator.sweep import SimConfig, SweepConfig, run_sweep

    rows = run_sweep(
        SweepConfig(
            presets=("min_cost_pull", "evict_lfu", "evict_lru"),
            num_requests=8,
            seeds=1,
            base_seed=7,
            sim=SimConfig.with_block_slots(hbm=24, dram=32),
        )
    )
    assert len(rows) == 3
    assert all(r.status == "ok" for r in rows)
    assert rows[0].read_path == "min_cost"

