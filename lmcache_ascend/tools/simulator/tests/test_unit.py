"""Small unit tests for policies, tasks, memory, and scheduler."""

from __future__ import annotations

from memory import BlockState, Memory
from policies import (
    ComputeOnlyLookupPolicy,
    CostBasedPullLookupPolicy,
    OrderedPullLookupPolicy,
    local_satisfied,
)
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
    from engine import Engine

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
    from engine import Engine
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


def run_unit_tests() -> None:
    tests = [
        test_lookup_compute,
        test_pull_only_rejects_compute_fallback,
        test_cost_model_picks_faster_pull_source,
        test_cost_model_prefers_compute_under_load,
        test_cost_model_prefill_recompute_expensive,
        test_cost_model_decode_recompute_cheap,
        test_cost_model_pending_pulls_in_allocation,
        test_cost_model_link_scheduled_load,
        test_cost_model_pull_only_ignores_compute,
        test_task_prereq_ordering,
        test_prefix_block_count_on_arrival,
        test_finish_frees_kv,
        test_local_satisfied_inflight,
        test_request_metrics_phases,
        test_request_metrics_pd_decode,
    ]
    for test in tests:
        test()
        print(f"{test.__name__} ok")
