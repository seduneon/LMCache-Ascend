"""Tests for simulator invariants and PD / memory critical paths."""

from __future__ import annotations

from simulator.engine import Engine
from simulator.memory import BlockState, Memory
from simulator.plan import EntryPlan, WorkEntry
from simulator.request import Request, RequestPD, RequestStatus
from simulator.resource import BandwidthResource, ComputeResource
from simulator.scheduler import Scheduler
from simulator.sim_log import SimLogConfig
from simulator.simulator import Simulator
from simulator.tasks import BatchLoadTask, TaskPool, TaskStatus
from simulator.tests.test_helpers import (
    SimpleTask,
    execute_plan,
    make_plan,
    make_resident,
    policies_compute,
    policies_pull,
    schedule_compute,
    schedule_pull,
)


def _pd_engines(
    *,
    prefill_requests: list[Request],
    decode_hbm: int = 100,
    prefill_hbm: int = 100,
    max_num_seqs: int = 10,
):
    from simulator.engine import Engine
    from simulator.pd import PDConfig

    pool = TaskPool()
    memories = {
        "npu-0:hbm": Memory(size=prefill_hbm, name="npu-0:hbm"),
        "npu-1:hbm": Memory(size=decode_hbm, name="npu-1:hbm"),
    }
    compute = ComputeResource(base_speed=4.0)
    link = BandwidthResource(base_speed=4.0, latency=0.01)
    npu0 = Engine(
        engine_id="npu-0",
        requests=prefill_requests,
        pool=pool,
        memories=memories,
        policies=policies_compute("npu-0:hbm"),
        compute_res=compute,
        work_per_block=1.0,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=32,
    )
    npu1 = Engine(
        engine_id="npu-1",
        requests=[],
        pool=pool,
        memories=memories,
        policies=policies_pull("npu-1:hbm", ["npu-0:hbm"]),
        compute_res=compute,
        bandwidth_res=link,
        work_per_transfer=1.0,
        work_per_block=1.0,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=32,
        remote_kv_wait=True,
    )
    sim = Simulator(
        [npu0, npu1],
        pool,
        pd=PDConfig(spawn_map={"npu-0": "npu-1"}),
    )
    return sim, npu0, npu1, memories, pool


def test_micro_step_waits_for_arrival() -> None:
    """With no runnable work, one step() advances now to the next arrival."""
    from simulator.engine import Engine

    pool = TaskPool()
    memories = {"hbm": Memory(size=10, name="hbm")}
    eng = Engine(
        "e0",
        [Request("r1", 2.5, ["a"], RequestPD.PREFILL, RequestStatus.PENDING)],
        pool,
        memories,
        policies_compute("hbm"),
        ComputeResource(base_speed=1.0),
        work_per_block=1.0,
    )
    sim = Simulator([eng], pool)

    assert sim.step()
    assert sim.now == 2.5
    assert "e0" not in sim._in_flight
    sim.run()
    assert sim.event_steps >= 1


def test_event_steps_matches_completed_steps() -> None:
    from simulator.engine import Engine

    pool = TaskPool()
    memories = {"hbm": Memory(size=10, name="hbm")}
    eng = Engine(
        "e0",
        [Request("r1", 0.0, ["a"], RequestPD.PREFILL, RequestStatus.PENDING)],
        pool,
        memories,
        policies_compute("hbm"),
        ComputeResource(base_speed=1.0),
        work_per_block=1.0,
    )
    sim = Simulator([eng], pool)
    finish = sim.run()
    assert finish > 0
    assert sim.event_steps > 0


def test_in_flight_blocks_reschedule() -> None:
    """An engine with an in-flight batch must not execute another batch."""
    from simulator.engine import Engine

    pool = TaskPool()
    memories = {"hbm": Memory(size=10, name="hbm")}
    eng = Engine(
        "e0",
        [
            Request("r1", 0.0, ["a", "b"], RequestPD.PREFILL, RequestStatus.PENDING),
            Request("r2", 0.0, ["c", "d"], RequestPD.PREFILL, RequestStatus.PENDING),
        ],
        pool,
        memories,
        policies_compute("hbm"),
        ComputeResource(base_speed=1.0),
        work_per_block=1.0,
        max_num_batched_tokens=2,
    )
    sim = Simulator([eng], pool)
    sim.run()
    assert len(eng.completed) == 2


def test_time_monotonic_across_steps() -> None:
    sim, npu0, _, _, _ = _pd_engines(
        prefill_requests=[
            Request("r1", 0.0, ["a", "b"], RequestPD.PREFILL, RequestStatus.PENDING, max_output_blocks=1),
        ]
    )
    prev = sim.now
    while sim.step():
        assert sim.now >= prev
        prev = sim.now


def test_pd_kv_held_after_remote_kv_promote() -> None:
    """Prefill KV stays held after D pulls prefix but before D finishes."""
    sim, npu0, npu1, _, _ = _pd_engines(
        prefill_requests=[
            Request(
                "r1",
                0.0,
                ["a", "b", "c"],
                RequestPD.PREFILL,
                RequestStatus.PENDING,
                max_output_blocks=2,
            )
        ]
    )
    saw_held_while_decode_running = False

    while sim.step():
        prefill_done = next((r for r in npu0.completed if r.req_id == "r1"), None)
        decode = next(
            (r for r in npu1.running + list(npu1.waiting) if r.req_id == "r1"),
            None,
        )
        if (
            prefill_done is not None
            and prefill_done.kv_held_for_transfer
            and decode is not None
            and decode in npu1.running
            and decode.num_computed_blocks > 0
            and decode.num_computed_blocks < decode.total_blocks()
        ):
            saw_held_while_decode_running = True
            break

    assert saw_held_while_decode_running, "expected P KV held while D is mid-decode"
    prefill_done = next(r for r in npu0.completed if r.req_id == "r1")
    assert prefill_done.kv_held_for_transfer


def test_pd_kv_survives_manual_preemption() -> None:
    """Narrow: scheduler preempt hook after remote-KV promote; sim must recover."""
    sim, npu0, npu1, memories, _ = _pd_engines(
        prefill_requests=[
            Request(
                "r1",
                0.0,
                ["a", "b", "c"],
                RequestPD.PREFILL,
                RequestStatus.PENDING,
                max_output_blocks=1,
            ),
        ],
    )

    promoted = False
    while sim.step():
        prefill = next((r for r in npu0.completed if r.req_id == "r1"), None)
        decode = next((r for r in npu1.running if r.req_id == "r1"), None)
        if (
            prefill is not None
            and prefill.kv_held_for_transfer
            and decode is not None
            and decode.num_computed_blocks == decode.prefix_block_count
        ):
            promoted = True
            break

    assert promoted, "expected decode promoted after remote-KV pull"

    prefill = next(r for r in npu0.completed if r.req_id == "r1")
    decode = next(r for r in npu1.running if r.req_id == "r1")
    assert prefill.kv_held_for_transfer
    for block_hash in ("a", "b", "c"):
        assert memories["npu-0:hbm"].best_resident(block_hash) is not None

    npu1.scheduler._preempt_request(decode)
    assert prefill.kv_held_for_transfer
    for block_hash in ("a", "b", "c"):
        assert memories["npu-0:hbm"].best_resident(block_hash) is not None

    waiting = next(r for r in npu1.waiting if r.req_id == "r1")
    assert waiting.num_computed_blocks == 0
    assert npu1.scheduler._needs_remote_kv(waiting)

    while sim.step():
        pass

    decode_done = next(r for r in npu1.completed if r.req_id == "r1")
    assert decode_done.num_computed_blocks == decode_done.total_blocks()
    assert not prefill.kv_held_for_transfer
    assert memories["npu-0:hbm"].used_size() == 0
    assert memories["npu-1:hbm"].used_size() == 0


def test_pd_organic_decode_preemption_e2e() -> None:
    """Tight decode HBM forces real scheduler preemption; full PD path must finish."""
    blocks = ["a", "b", "c", "d"]
    sim, npu0, npu1, memories, _ = _pd_engines(
        prefill_requests=[
            Request(
                f"r{i}",
                i * 0.05,
                blocks,
                RequestPD.PREFILL,
                RequestStatus.PENDING,
                max_output_blocks=2,
            )
            for i in range(3)
        ],
        decode_hbm=6,
        max_num_seqs=3,
    )

    sim.run(wall_timeout_s=30.0)

    assert len(npu0.completed) == 3
    assert len(npu1.completed) == 3
    decode_preemptions = sum(r.metrics.preemptions for r in npu1.completed)
    assert decode_preemptions >= 1, (
        "expected organic decode-side preemption under tight HBM"
    )
    preempted = next(r for r in npu1.completed if r.metrics.preemptions >= 1)
    assert preempted.metrics.pulls + preempted.metrics.local_hits >= 1
    for req_id in ("r0", "r1", "r2"):
        prefill = next(r for r in npu0.completed if r.req_id == req_id)
        decode = next(r for r in npu1.completed if r.req_id == req_id)
        assert not prefill.kv_held_for_transfer
        assert decode.metrics.finished_at is not None
        assert decode.metrics.computes >= 2
    assert memories["npu-0:hbm"].used_size() == 0
    assert memories["npu-1:hbm"].used_size() == 0


def test_stress_n16_seed42_regression() -> None:
    """Regression for the n=16 PD deadlock (early P KV release + decode preempt)."""
    from simulator.tests.test_stress import StressConfig, run_stress_test

    run_stress_test(
        StressConfig(
            num_requests=16,
            seed=42,
            show_progress=False,
            log=SimLogConfig(enabled=False),
        )
    )


def test_pd_kv_released_only_on_decode_complete() -> None:
    sim, npu0, npu1, memories, _ = _pd_engines(
        prefill_requests=[
            Request(
                "r1",
                0.0,
                ["a", "b"],
                RequestPD.PREFILL,
                RequestStatus.PENDING,
                max_output_blocks=1,
            )
        ]
    )
    sim.run()
    prefill = next(r for r in npu0.completed if r.req_id == "r1")
    decode = next(r for r in npu1.completed if r.req_id == "r1")
    assert decode.status == RequestStatus.COMPLETE
    assert not prefill.kv_held_for_transfer
    assert memories["npu-0:hbm"].used_size() == 0


def test_parallel_pull_tasks_start_together() -> None:
    """Pull tasks in one batch depend on evicts only, not on each other."""
    pool = TaskPool()
    memories = {
        "src": Memory(size=10, name="src"),
        "dst": Memory(size=10, name="dst"),
    }
    for block_hash in ("a", "b"):
        make_resident(memories["src"], block_hash)

    eng = Engine(
        "d",
        [],
        pool,
        memories,
        policies_pull("dst", ["src"]),
        ComputeResource(base_speed=1.0),
        BandwidthResource(base_speed=1.0),
        work_per_transfer=1.0,
    )

    r1 = Request("r1", 0.0, ["a"], RequestPD.DECODE, RequestStatus.RUNNING)
    r1.prefix_block_count = 1
    r2 = Request("r2", 0.0, ["b"], RequestPD.DECODE, RequestStatus.RUNNING)
    r2.prefix_block_count = 1
    work = make_plan(
        [
            WorkEntry(r1, ["a"], EntryPlan(blocks={"a": ("pull", "src")})),
            WorkEntry(r2, ["b"], EntryPlan(blocks={"b": ("pull", "src")})),
        ],
        engine_id="d",
    )

    tasks = execute_plan(eng, work)
    pulls = [t for t in tasks if isinstance(t, BatchLoadTask)]
    assert len(pulls) == 2
    for pull in pulls:
        assert pull.prereqs == []

    pool.start_ready(0.0)
    assert all(t.status == TaskStatus.RUNNING for t in pulls)


def test_task_latency_before_work() -> None:
    res = ComputeResource(base_speed=2.0, latency=0.25)
    task = SimpleTask(1.0, res)
    task.reserve_resource()
    task.start(1.0)
    assert task.now == 1.25
    assert task.estimated_end() == 1.25 + 0.5


def test_held_blocks_not_evictable() -> None:
    mem = Memory(size=4, name="hbm")
    block = mem.append_reserved("a", "r1")
    block.state = BlockState.RESIDENT
    assert not mem.can_evict_block(block)

    block.holders.clear()
    assert mem.can_evict_block(block)


def test_shared_block_survives_partial_free() -> None:
    mem = Memory(size=4, name="hbm")
    block = mem.append_reserved("shared", "r1")
    block.state = BlockState.RESIDENT
    block.holders.add("r2")

    mem.free_request("r1")
    assert mem.used_size() == 1
    assert block in mem.list()

    mem.free_request("r2")
    assert mem.used_size() == 0


def test_scheduler_limits() -> None:
    memories = {"hbm": Memory(size=100, name="hbm")}
    policy = schedule_compute(local_memory="hbm")
    sched = Scheduler(
        policy, memories, "hbm", max_num_seqs=1, max_num_batched_tokens=10, block_size=1
    )
    r1 = Request("r1", 0.0, ["a"], RequestPD.DECODE, RequestStatus.WAITING, max_output_blocks=1)
    r2 = Request("r2", 0.0, ["b"], RequestPD.DECODE, RequestStatus.WAITING, max_output_blocks=1)
    r1.prefix_block_count = 1
    r2.prefix_block_count = 1
    sched.waiting.extend([r1, r2])
    batch = sched.schedule()
    assert len(batch.entries) == 1
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
    assert len(batch2.entries) == 2
    assert batch2.total_num_scheduled_tokens == 2


def test_chunked_prefill() -> None:
    memories = {"hbm": Memory(size=100, name="hbm")}
    pool = TaskPool()
    eng = Engine(
        engine_id="e0",
        requests=[],
        pool=pool,
        memories=memories,
        policies=policies_compute("hbm"),
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
    assert steps >= 3
    assert memories["hbm"].used_size() == 0


def test_pd_read_mode_flags() -> None:
    from simulator.pd import PDConfig

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
        policies=policies_compute("npu-0:hbm"),
        compute_res=ComputeResource(base_speed=1.0),
        work_per_block=1.0,
    )
    npu1 = Engine(
        engine_id="npu-1",
        requests=[],
        pool=pool,
        memories=memories,
        policies=policies_pull("npu-1:hbm", ["npu-0:hbm"]),
        compute_res=ComputeResource(base_speed=1.0),
        bandwidth_res=BandwidthResource(base_speed=1.0),
        work_per_block=1.0,
        work_per_transfer=1.0,
    )
    sim = Simulator([npu0, npu1], pool, pd=PDConfig(spawn_map={"npu-0": "npu-1"}))
    sim.run()

    decode = next((r for r in npu1.completed if r.req_id == "r1"), None)
    assert decode is not None
    assert decode.num_computed_blocks == decode.total_blocks()
    assert not prefill.kv_held_for_transfer
    assert memories["npu-0:hbm"].used_size() == 0
    assert memories["npu-1:hbm"].used_size() == 0
    assert npu1.remote_kv_wait is True
    assert npu0.hold_kv_on_complete is True


def test_remote_kv_admit() -> None:
    memories = {
        "npu-0:hbm": Memory(size=10, name="npu-0:hbm"),
        "npu-1:hbm": Memory(size=10, name="npu-1:hbm"),
    }
    for block_hash in ("a", "b", "c"):
        make_resident(memories["npu-0:hbm"], block_hash, "producer")

    policy = schedule_pull(local_memory="npu-1:hbm", pull_sources=["npu-0:hbm"])
    sched = Scheduler(policy, memories, "npu-1:hbm", remote_kv_wait=True)
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


def test_pd_backpressure() -> None:
    memories = {
        "npu-0:hbm": Memory(size=10, name="npu-0:hbm"),
        "npu-1:hbm": Memory(size=2, name="npu-1:hbm"),
    }
    for block_hash in ("a", "b", "c"):
        make_resident(memories["npu-0:hbm"], block_hash, "producer")

    policy = schedule_pull(local_memory="npu-1:hbm", pull_sources=["npu-0:hbm"])
    sched = Scheduler(
        policy, memories, "npu-1:hbm", max_num_seqs=10, remote_kv_wait=True
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
    assert len(batch.entries) == 0
    assert req.status == RequestStatus.WAITING


def test_remote_kv_max_seqs() -> None:
    memories = {
        "npu-0:hbm": Memory(size=10, name="npu-0:hbm"),
        "npu-1:hbm": Memory(size=10, name="npu-1:hbm"),
    }
    for block_hash in ("a", "b"):
        make_resident(memories["npu-0:hbm"], block_hash, "producer")

    policy = schedule_pull(local_memory="npu-1:hbm", pull_sources=["npu-0:hbm"])
    sched = Scheduler(policy, memories, "npu-1:hbm", max_num_seqs=1, remote_kv_wait=True)
    r1 = Request("r1", 0.0, ["a", "b"], RequestPD.DECODE, RequestStatus.WAITING, max_output_blocks=1)
    r1.prefix_block_count = 2
    r1.status = RequestStatus.WAITING_REMOTE_KV
    r2 = Request("r2", 0.0, ["a", "b"], RequestPD.DECODE, RequestStatus.WAITING, max_output_blocks=1)
    r2.prefix_block_count = 2
    sched.waiting.extend([r1, r2])

    batch = sched.schedule()
    assert len(batch.entries) == 0


def test_remote_kv_queue_rotation() -> None:
    memories = {"hbm": Memory(size=100, name="hbm")}
    policy = schedule_compute(local_memory="hbm")
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


def test_waiting_preempt() -> None:
    memories = {"hbm": Memory(size=4, name="hbm")}
    policy = schedule_compute(local_memory="hbm")
    sched = Scheduler(policy, memories, "hbm", max_num_batched_tokens=10)

    running = Request("r1", 0.0, ["a", "b", "c"], RequestPD.DECODE, RequestStatus.RUNNING, max_output_blocks=1)
    running.prefix_block_count = 3
    running.num_computed_blocks = 4
    sched.running.append(running)
    for block_hash in ("a", "b", "c"):
        make_resident(memories["hbm"], block_hash, "r1")

    waiting = Request("r2", 0.0, ["x", "y", "z"], RequestPD.DECODE, RequestStatus.WAITING, max_output_blocks=1)
    waiting.prefix_block_count = 3
    sched.waiting.append(waiting)

    batch = sched.schedule()
    assert len(batch.preempted) >= 1
    assert waiting in sched.running


def test_pd_config_validation() -> None:
    from simulator.pd import PDConfig

    pool = TaskPool()
    memories = {"p": Memory(size=10, name="p"), "d": Memory(size=10, name="d")}
    npu0 = Engine("p", [], pool, memories, policies_compute("p"), ComputeResource(base_speed=1.0))
    npu1 = Engine("d", [], pool, memories, policies_compute("d"), ComputeResource(base_speed=1.0))
    try:
        Simulator([npu0, npu1], pool, pd=PDConfig(spawn_map={"p": "d"}))
        raise AssertionError("expected ValueError for decode without pull_sources")
    except ValueError:
        pass

