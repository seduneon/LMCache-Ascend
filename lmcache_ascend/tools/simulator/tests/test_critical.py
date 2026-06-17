"""Tests for simulator invariants and PD / memory critical paths."""

from __future__ import annotations

from memory import BlockState, Memory
from policies import ComputeOnlyLookupPolicy, CostBasedPullLookupPolicy
from request import Request, RequestPD, RequestStatus
from resource import BandwidthResource, ComputeResource
from scheduler import Batch, BatchEntry
from sim_log import SimLogConfig
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


def _pd_engines(
    *,
    prefill_requests: list[Request],
    decode_hbm: int = 100,
    prefill_hbm: int = 100,
    max_num_seqs: int = 10,
):
    from engine import Engine
    from pd import PDConfig

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
        local_memory="npu-0:hbm",
        policy=ComputeOnlyLookupPolicy(local_memory="npu-0:hbm"),
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
        local_memory="npu-1:hbm",
        policy=CostBasedPullLookupPolicy(
            local_memory="npu-1:hbm", pull_sources=["npu-0:hbm"]
        ),
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
    from engine import Engine

    pool = TaskPool()
    memories = {"hbm": Memory(size=10, name="hbm")}
    eng = Engine(
        "e0",
        [Request("r1", 2.5, ["a"], RequestPD.PREFILL, RequestStatus.PENDING)],
        pool,
        memories,
        "hbm",
        ComputeOnlyLookupPolicy(local_memory="hbm"),
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
    from engine import Engine

    pool = TaskPool()
    memories = {"hbm": Memory(size=10, name="hbm")}
    eng = Engine(
        "e0",
        [Request("r1", 0.0, ["a"], RequestPD.PREFILL, RequestStatus.PENDING)],
        pool,
        memories,
        "hbm",
        ComputeOnlyLookupPolicy(local_memory="hbm"),
        ComputeResource(base_speed=1.0),
        work_per_block=1.0,
    )
    sim = Simulator([eng], pool)
    finish = sim.run()
    assert finish > 0
    assert sim.event_steps > 0


def test_in_flight_blocks_reschedule() -> None:
    """An engine with an in-flight batch must not execute another batch."""
    from engine import Engine

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
        "hbm",
        ComputeOnlyLookupPolicy(local_memory="hbm"),
        ComputeResource(base_speed=1.0),
        work_per_block=1.0,
        max_num_batched_tokens=2,
    )
    sim = Simulator([eng], pool)

    original = eng.execute_batch

    def guarded_execute(batch: Batch, now: float = 0.0):
        assert "e0" not in sim._in_flight, "execute_batch while batch still in flight"
        return original(batch, now)

    eng.execute_batch = guarded_execute  # type: ignore[method-assign]

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
    from tests.test_stress import StressConfig, run_stress_test

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
    from engine import Engine
    from policies import LookupResult, OrderedPullLookupPolicy
    from tasks import BatchLoadTask

    pool = TaskPool()
    memories = {
        "src": Memory(size=10, name="src"),
        "dst": Memory(size=10, name="dst"),
    }
    for block_hash in ("a", "b"):
        _make_resident(memories["src"], block_hash)

    eng = Engine(
        "d",
        [],
        pool,
        memories,
        "dst",
        OrderedPullLookupPolicy(local_memory="dst", pull_sources=["src"]),
        ComputeResource(base_speed=1.0),
        BandwidthResource(base_speed=1.0),
        work_per_transfer=1.0,
    )

    batch = Batch(
        entries=[
            BatchEntry(
                Request("r1", 0.0, ["a"], RequestPD.DECODE, RequestStatus.RUNNING),
                ["a"],
                LookupResult(blocks={"a": ("pull", "src")}),
            ),
            BatchEntry(
                Request("r2", 0.0, ["b"], RequestPD.DECODE, RequestStatus.RUNNING),
                ["b"],
                LookupResult(blocks={"b": ("pull", "src")}),
            ),
        ]
    )
    batch.entries[0].req.prefix_block_count = 1
    batch.entries[1].req.prefix_block_count = 1

    tasks = eng.execute_batch(batch)
    pulls = [t for t in tasks if isinstance(t, BatchLoadTask)]
    assert len(pulls) == 2
    for pull in pulls:
        assert pull.prereqs == []

    pool.start_ready(0.0)
    assert all(t.status == TaskStatus.RUNNING for t in pulls)


def test_task_latency_before_work() -> None:
    res = ComputeResource(base_speed=2.0, latency=0.25)
    task = _SimpleTask(1.0, res)
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


def run_critical_tests() -> None:
    tests = [
        test_micro_step_waits_for_arrival,
        test_event_steps_matches_completed_steps,
        test_in_flight_blocks_reschedule,
        test_time_monotonic_across_steps,
        test_pd_kv_held_after_remote_kv_promote,
        test_pd_kv_survives_manual_preemption,
        test_pd_organic_decode_preemption_e2e,
        test_stress_n16_seed42_regression,
        test_pd_kv_released_only_on_decode_complete,
        test_parallel_pull_tasks_start_together,
        test_task_latency_before_work,
        test_held_blocks_not_evictable,
        test_shared_block_survives_partial_free,
    ]
    for test in tests:
        test()
        print(f"{test.__name__} ok")
