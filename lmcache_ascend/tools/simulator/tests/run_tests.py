"""Integration and unit tests for the KV cache simulator."""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[2]
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from simulator.engine import Engine
from simulator.pd import PDConfig
from simulator.request import Request, RequestPD, RequestStatus
from simulator.simulator import Simulator
from simulator.tasks import TaskPool
from simulator.tests.test_discovery import (
    ensure_path,
    run_critical_tests,
    run_unit_tests,
)
from simulator.tests.test_sweep_regression import run_sweep_regression_tests


def run_pd_demo() -> None:
    from simulator.memory import Memory
    from simulator.policies import (
        ComputeOnlyLookupPolicy,
        CostBasedPullLookupPolicy,
    )
    from simulator.resource import BandwidthResource, ComputeResource

    pool = TaskPool()
    memories = {
        "npu-0:hbm": Memory(size=100, name="npu-0:hbm"),
        "npu-1:hbm": Memory(size=100, name="npu-1:hbm"),
    }
    prefill_requests = [
        Request(
            "r1",
            0.0,
            ["a", "b", "c"],
            RequestPD.PREFILL,
            RequestStatus.PENDING,
            max_output_blocks=5,
        ),
    ]
    npu0 = Engine(
        engine_id="npu-0",
        requests=prefill_requests,
        pool=pool,
        memories=memories,
        local_memory="npu-0:hbm",
        policy=ComputeOnlyLookupPolicy(local_memory="npu-0:hbm"),
        compute_res=ComputeResource(base_speed=1.0),
        work_per_block=1.0,
    )
    npu1 = Engine(
        engine_id="npu-1",
        requests=[],
        pool=pool,
        memories=memories,
        local_memory="npu-1:hbm",
        policy=CostBasedPullLookupPolicy(
            local_memory="npu-1:hbm", pull_sources=["npu-0:hbm"]
        ),
        compute_res=ComputeResource(base_speed=1.0),
        bandwidth_res=BandwidthResource(base_speed=1.0),
        work_per_block=1.0,
        work_per_transfer=1.0,
    )
    sim = Simulator(
        [npu0, npu1],
        pool,
        pd=PDConfig(spawn_map={"npu-0": "npu-1"}),
    )
    finish = sim.run()
    print(f"finish_time={finish}")
    print(f"npu-0:hbm blocks={[b.hash for b in memories['npu-0:hbm'].list()]}")
    print(f"npu-1:hbm blocks={[b.hash for b in memories['npu-1:hbm'].list()]}")
    print(f"r1 prefill status={prefill_requests[0].status}")
    decode_req = next((r for r in npu1.completed if r.req_id == "r1"), None)
    if decode_req is not None:
        print(
            f"r1 decode status={decode_req.status} "
            f"computed={decode_req.num_computed_blocks}/"
            f"{decode_req.total_blocks()}"
        )
    assert memories["npu-0:hbm"].used_size() == 0
    assert memories["npu-1:hbm"].used_size() == 0


def run_deadlock_test() -> None:
    from simulator.memory import Memory
    from simulator.policies import ComputeOnlyLookupPolicy
    from simulator.resource import ComputeResource

    pool = TaskPool()
    hbm = Memory(size=6, name="npu-0:hbm")
    memories = {"npu-0:hbm": hbm}
    requests = [
        Request(
            "r1",
            0.0,
            ["a", "b", "c"],
            RequestPD.DECODE,
            RequestStatus.PENDING,
            max_output_blocks=2,
        ),
        Request(
            "r2",
            0.0,
            ["x", "y", "z"],
            RequestPD.DECODE,
            RequestStatus.PENDING,
            max_output_blocks=1,
        ),
    ]
    eng = Engine(
        engine_id="npu-0",
        requests=requests,
        pool=pool,
        memories=memories,
        local_memory="npu-0:hbm",
        policy=ComputeOnlyLookupPolicy(local_memory="npu-0:hbm"),
        compute_res=ComputeResource(base_speed=1.0),
        work_per_block=1.0,
    )
    sim = Simulator([eng], pool)
    finish = sim.run()
    by_id = {r.req_id: r for r in eng.completed}
    assert finish < float("inf"), "simulation did not finish"
    assert "r1" in by_id and "r2" in by_id, "both requests must complete"
    assert by_id["r1"].status == RequestStatus.COMPLETE
    assert by_id["r2"].status == RequestStatus.COMPLETE
    total_preemptions = by_id["r1"].num_preemptions + by_id["r2"].num_preemptions
    assert total_preemptions >= 1, "at least one request should be preempted"
    assert hbm.used_size() == 0, "KV freed after completion"
    print(f"deadlock_test finish_time={finish}")
    print(f"r1 preemptions={by_id['r1'].num_preemptions} r2 preemptions={by_id['r2'].num_preemptions}")
    print(f"r1 computed={by_id['r1'].num_computed_blocks}/{by_id['r1'].total_blocks()}")
    print(f"r2 computed={by_id['r2'].num_computed_blocks}/{by_id['r2'].total_blocks()}")


def run_limits_test() -> None:
    from simulator.memory import Memory
    from simulator.policies import ComputeOnlyLookupPolicy
    from simulator.scheduler import Scheduler

    memories = {"hbm": Memory(size=100, name="hbm")}
    policy = ComputeOnlyLookupPolicy(local_memory="hbm")
    sched = Scheduler(
        policy, memories, "hbm", max_num_seqs=1, max_num_batched_tokens=10, block_size=1
    )
    r1 = Request("r1", 0.0, ["a"], RequestPD.DECODE, RequestStatus.WAITING, max_output_blocks=1)
    r2 = Request("r2", 0.0, ["b"], RequestPD.DECODE, RequestStatus.WAITING, max_output_blocks=1)
    r1.prefix_block_count = 1
    r2.prefix_block_count = 1
    sched.waiting.extend([r1, r2])
    batch = sched.schedule()
    assert len(batch.entries) == 1, "max_num_seqs=1 admits one waiting request"
    assert len(sched.waiting) == 1
    assert len(sched.running) == 1

    sched2 = Scheduler(
        policy, memories, "hbm", max_num_seqs=10, max_num_batched_tokens=2, block_size=1
    )
    for rid in ("r1", "r2", "r3"):
        req = Request(
            rid, 0.0, ["p"], RequestPD.DECODE, RequestStatus.RUNNING, max_output_blocks=2
        )
        req.prefix_block_count = 1
        req.num_computed_blocks = 1
        sched2.running.append(req)
    batch2 = sched2.schedule()
    assert len(batch2.entries) == 2, "token_budget=2 schedules two decode reqs"
    assert batch2.total_num_scheduled_tokens == 2
    print("limits_test ok")


def run_chunked_prefill_test() -> None:
    from simulator.memory import Memory
    from simulator.policies import ComputeOnlyLookupPolicy
    from simulator.resource import ComputeResource

    memories = {"hbm": Memory(size=100, name="hbm")}
    policy = ComputeOnlyLookupPolicy(local_memory="hbm")
    pool = TaskPool()
    eng = Engine(
        engine_id="e0",
        requests=[],
        pool=pool,
        memories=memories,
        local_memory="hbm",
        policy=policy,
        compute_res=ComputeResource(base_speed=1.0),
        block_size=1,
        max_num_batched_tokens=2,
        enable_chunked_prefill=True,
        work_per_block=1.0,
    )
    prefix = [f"p{i}" for i in range(5)]
    req = Request("r1", 0.0, list(prefix), RequestPD.PREFILL, RequestStatus.PENDING)
    eng.schedule_request(req)

    sim = Simulator([eng], pool)
    steps = 0
    while sim.step():
        steps += 1

    assert len(eng.completed) == 1
    assert eng.completed[0].num_computed_blocks == len(prefix)
    assert steps >= 3, "5 tokens with budget 2 needs multiple prefill steps"
    assert memories["hbm"].used_size() == 0
    print(f"chunked_prefill_test ok steps={steps}")


def run_pd_read_test() -> None:
    from simulator.memory import Memory
    from simulator.policies import (
        ComputeOnlyLookupPolicy,
        CostBasedPullLookupPolicy,
    )
    from simulator.resource import BandwidthResource, ComputeResource

    pool = TaskPool()
    memories = {
        "npu-0:hbm": Memory(size=100, name="npu-0:hbm"),
        "npu-1:hbm": Memory(size=100, name="npu-1:hbm"),
    }
    prefill = Request(
        "r1",
        0.0,
        ["a", "b", "c"],
        RequestPD.PREFILL,
        RequestStatus.PENDING,
        max_output_blocks=2,
    )
    npu0 = Engine(
        engine_id="npu-0",
        requests=[prefill],
        pool=pool,
        memories=memories,
        local_memory="npu-0:hbm",
        policy=ComputeOnlyLookupPolicy(local_memory="npu-0:hbm"),
        compute_res=ComputeResource(base_speed=1.0),
        work_per_block=1.0,
    )
    npu1 = Engine(
        engine_id="npu-1",
        requests=[],
        pool=pool,
        memories=memories,
        local_memory="npu-1:hbm",
        policy=CostBasedPullLookupPolicy(
            local_memory="npu-1:hbm", pull_sources=["npu-0:hbm"]
        ),
        compute_res=ComputeResource(base_speed=1.0),
        bandwidth_res=BandwidthResource(base_speed=1.0),
        work_per_block=1.0,
        work_per_transfer=1.0,
    )
    sim = Simulator([npu0, npu1], pool, pd=PDConfig(spawn_map={"npu-0": "npu-1"}))
    finish = sim.run()

    decode = next((r for r in npu1.completed if r.req_id == "r1"), None)
    assert decode is not None
    assert decode.num_computed_blocks == decode.total_blocks()
    assert not prefill.kv_held_for_transfer, "P KV released after D completes"
    assert memories["npu-0:hbm"].used_size() == 0, "P prefix freed after transfer"
    assert memories["npu-1:hbm"].used_size() == 0, "D KV freed after decode"
    assert npu1.remote_kv_wait is True
    assert npu0.hold_kv_on_complete is True
    print(f"pd_read_test ok finish_time={finish}")


def run_remote_kv_admit_test() -> None:
    from simulator.memory import BlockState, Memory
    from simulator.policies import OrderedPullLookupPolicy
    from simulator.scheduler import Scheduler

    memories = {
        "npu-0:hbm": Memory(size=10, name="npu-0:hbm"),
        "npu-1:hbm": Memory(size=10, name="npu-1:hbm"),
    }
    for block_hash in ("a", "b", "c"):
        memories["npu-0:hbm"].append_reserved(block_hash, "producer")
        block = memories["npu-0:hbm"].find_reserved_for(block_hash, "producer")
        block.state = BlockState.RESIDENT

    policy = OrderedPullLookupPolicy(local_memory="npu-1:hbm", pull_sources=["npu-0:hbm"])
    sched = Scheduler(
        policy,
        memories,
        "npu-1:hbm",
        remote_kv_wait=True,
    )
    req = Request(
        "r1",
        0.0,
        ["a", "b", "c"],
        RequestPD.DECODE,
        RequestStatus.WAITING,
        max_output_blocks=1,
    )
    req.prefix_block_count = 3
    sched.waiting.append(req)

    batch = sched.schedule()
    assert len(batch.entries) == 1
    assert batch.entries[0].remote_kv
    assert req.status == RequestStatus.WAITING_REMOTE_KV
    print("remote_kv_admit_test ok")


def run_pd_backpressure_test() -> None:
    from simulator.memory import BlockState, Memory
    from simulator.policies import OrderedPullLookupPolicy
    from simulator.scheduler import Scheduler

    memories = {
        "npu-0:hbm": Memory(size=10, name="npu-0:hbm"),
        "npu-1:hbm": Memory(size=2, name="npu-1:hbm"),
    }
    for block_hash in ("a", "b", "c"):
        memories["npu-0:hbm"].append_reserved(block_hash, "producer")
        block = memories["npu-0:hbm"].find_reserved_for(block_hash, "producer")
        block.state = BlockState.RESIDENT

    policy = OrderedPullLookupPolicy(local_memory="npu-1:hbm", pull_sources=["npu-0:hbm"])
    sched = Scheduler(
        policy,
        memories,
        "npu-1:hbm",
        max_num_seqs=10,
        remote_kv_wait=True,
    )
    req = Request(
        "r1",
        0.0,
        ["a", "b", "c"],
        RequestPD.DECODE,
        RequestStatus.WAITING,
        max_output_blocks=1,
    )
    req.prefix_block_count = 3
    sched.waiting.append(req)

    batch = sched.schedule()
    assert len(batch.entries) == 0, "D HBM cannot fit 3-block remote-KV pre-alloc"
    assert req.status == RequestStatus.WAITING
    print("pd_backpressure_test ok")


def run_remote_kv_max_seqs_test() -> None:
    from simulator.memory import BlockState, Memory
    from simulator.policies import OrderedPullLookupPolicy
    from simulator.scheduler import Scheduler

    memories = {
        "npu-0:hbm": Memory(size=10, name="npu-0:hbm"),
        "npu-1:hbm": Memory(size=10, name="npu-1:hbm"),
    }
    for block_hash in ("a", "b"):
        memories["npu-0:hbm"].append_reserved(block_hash, "producer")
        block = memories["npu-0:hbm"].find_reserved_for(block_hash, "producer")
        block.state = BlockState.RESIDENT

    policy = OrderedPullLookupPolicy(local_memory="npu-1:hbm", pull_sources=["npu-0:hbm"])
    sched = Scheduler(
        policy,
        memories,
        "npu-1:hbm",
        max_num_seqs=1,
        remote_kv_wait=True,
    )
    r1 = Request("r1", 0.0, ["a", "b"], RequestPD.DECODE, RequestStatus.WAITING, max_output_blocks=1)
    r1.prefix_block_count = 2
    r1.status = RequestStatus.WAITING_REMOTE_KV
    r2 = Request("r2", 0.0, ["a", "b"], RequestPD.DECODE, RequestStatus.WAITING, max_output_blocks=1)
    r2.prefix_block_count = 2
    sched.waiting.extend([r1, r2])

    batch = sched.schedule()
    assert len(batch.entries) == 0, "second decode blocked while r1 holds remote-KV slot"
    print("remote_kv_max_seqs_test ok")


def run_remote_kv_queue_rotation_test() -> None:
    """WAITING_REMOTE_KV at head does not block unrelated waiting admits behind it."""
    from simulator.memory import Memory
    from simulator.policies import ComputeOnlyLookupPolicy
    from simulator.scheduler import Scheduler

    memories = {"hbm": Memory(size=100, name="hbm")}
    policy = ComputeOnlyLookupPolicy(local_memory="hbm")
    sched = Scheduler(policy, memories, "hbm", max_num_batched_tokens=10, remote_kv_wait=True)

    blocked = Request("r1", 0.0, ["a"], RequestPD.DECODE, RequestStatus.WAITING, max_output_blocks=1)
    blocked.prefix_block_count = 1
    blocked.status = RequestStatus.WAITING_REMOTE_KV

    ready = Request("r2", 0.0, ["b"], RequestPD.DECODE, RequestStatus.WAITING, max_output_blocks=1)
    ready.prefix_block_count = 1
    sched.waiting.extend([blocked, ready])

    batch = sched.schedule()
    assert len(batch.entries) == 1
    assert batch.entries[0].req.req_id == "r2"
    assert ready in sched.running
    assert blocked in sched.waiting
    print("remote_kv_queue_rotation_test ok")


def run_waiting_preempt_test() -> None:
    """WAITING admit uses preempt path when HBM is full."""
    from simulator.memory import BlockState, Memory
    from simulator.policies import ComputeOnlyLookupPolicy
    from simulator.scheduler import Scheduler

    memories = {"hbm": Memory(size=4, name="hbm")}
    policy = ComputeOnlyLookupPolicy(local_memory="hbm")
    sched = Scheduler(policy, memories, "hbm", max_num_batched_tokens=10)

    running = Request("r1", 0.0, ["a", "b", "c"], RequestPD.DECODE, RequestStatus.RUNNING, max_output_blocks=1)
    running.prefix_block_count = 3
    running.num_computed_blocks = 4
    sched.running.append(running)
    for block_hash in ("a", "b", "c"):
        memories["hbm"].append_reserved(block_hash, "r1")
        block = memories["hbm"].find_reserved_for(block_hash, "r1")
        block.state = BlockState.RESIDENT

    waiting = Request("r2", 0.0, ["x", "y", "z"], RequestPD.DECODE, RequestStatus.WAITING, max_output_blocks=1)
    waiting.prefix_block_count = 3
    sched.waiting.append(waiting)

    batch = sched.schedule()
    assert len(batch.preempted) >= 1, "waiting admit should preempt running request"
    assert waiting in sched.running
    print("waiting_preempt_test ok")


def run_pd_config_validation_test() -> None:
    from simulator.memory import Memory
    from simulator.policies import ComputeOnlyLookupPolicy
    from simulator.resource import ComputeResource

    pool = TaskPool()
    memories = {"p": Memory(size=10, name="p"), "d": Memory(size=10, name="d")}
    npu0 = Engine(
        "p",
        [],
        pool,
        memories,
        "p",
        ComputeOnlyLookupPolicy(local_memory="p"),
        ComputeResource(base_speed=1.0),
    )
    npu1 = Engine(
        "d",
        [],
        pool,
        memories,
        "d",
        ComputeOnlyLookupPolicy(local_memory="d"),
        ComputeResource(base_speed=1.0),
    )
    try:
        Simulator([npu0, npu1], pool, pd=PDConfig(spawn_map={"p": "d"}))
        raise AssertionError("expected ValueError for decode without pull_sources")
    except ValueError:
        pass
    print("pd_config_validation_test ok")


_ALL = {
    "pd": [
        run_pd_read_test,
        run_remote_kv_admit_test,
        run_pd_backpressure_test,
        run_remote_kv_max_seqs_test,
        run_remote_kv_queue_rotation_test,
        run_pd_config_validation_test,
    ],
    "waiting": [run_waiting_preempt_test, run_remote_kv_queue_rotation_test],
    "deadlock": [run_deadlock_test],
    "limits": [run_limits_test],
    "chunked": [run_chunked_prefill_test],
}


def run_stress_test() -> None:
    from simulator.tests.test_stress import run_stress_test as _run

    _run()


def run_stress_heavy_test() -> None:
    from simulator.tests.test_stress import run_stress_heavy_test as _run

    _run()


def run_stress_benchmark_cli() -> None:
    from simulator.tests.test_stress import run_stress_benchmark

    run_stress_benchmark()


def run_stress_seed_sweep_cli() -> None:
    from simulator.tests.test_stress import run_stress_seed_sweep

    run_stress_seed_sweep()


_ALL["stress"] = [run_stress_test]
_ALL["stress-heavy"] = [run_stress_heavy_test]
_ALL["stress-benchmark"] = [run_stress_benchmark_cli]
_ALL["stress-seeds"] = [run_stress_seed_sweep_cli]
_ALL["critical"] = [run_critical_tests]
_ALL["unit"] = [run_unit_tests]
_ALL["sweep-regression"] = [run_sweep_regression_tests]


def main(argv: list[str] | None = None) -> None:
    ensure_path()
    argv = argv if argv is not None else sys.argv[1:]
    if argv:
        key = argv[0]
        if key == "sweep":
            from simulator.sweep import main as sweep_main

            sweep_main(argv[1:])
            return
        if key not in _ALL:
            raise SystemExit(f"unknown test group: {key!r} (try: {', '.join(sorted(_ALL))})")
        for test in _ALL[key]:
            test()
        return

    run_unit_tests()
    print()
    run_critical_tests()
    print()
    run_pd_demo()
    print()
    run_limits_test()
    print()
    run_chunked_prefill_test()
    print()
    run_waiting_preempt_test()
    print()
    for test in _ALL["pd"]:
        test()
        print()


if __name__ == "__main__":
    main()
