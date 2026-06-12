"""Small unit tests for policies, tasks, memory, and scheduler."""

from __future__ import annotations

from memory import BlockState, Memory
from policies import LookupPolicy
from request import Request, RequestPD, RequestStatus
from resource import ComputeResource
from scheduler import Scheduler
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
    policy = LookupPolicy(local_memory="hbm")
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
    policy = LookupPolicy(local_memory="dst", pull_sources=["src"])

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
    sched = Scheduler(LookupPolicy("hbm"), memories, "hbm")
    req = Request("r1", 1.0, ["a", "b", "c"], RequestPD.PREFILL, RequestStatus.PENDING)
    sched.add_request(req)
    sched.release_arrivals(1.0)
    assert req.prefix_block_count == 3
    assert req.status == RequestStatus.WAITING


def test_finish_frees_kv() -> None:
    memories = {"hbm": Memory(size=10, name="hbm")}
    sched = Scheduler(LookupPolicy("hbm"), memories, "hbm")
    req = Request("r1", 0.0, ["a"], RequestPD.PREFILL, RequestStatus.RUNNING)
    req.prefix_block_count = 1
    _make_resident(memories["hbm"], "a", "r1")
    sched.running.append(req)

    sched.finish_request(req)
    assert memories["hbm"].used_size() == 0
    assert req in sched.completed


def test_local_satisfied_inflight() -> None:
    memories = {"hbm": Memory(size=10, name="hbm")}
    policy = LookupPolicy("hbm")
    local = memories["hbm"]
    block = local.append_reserved("a", "r1")
    block.task = object()  # type: ignore[assignment]
    assert policy.local_satisfied(local, "a")
    assert policy.resolve_actions(memories, ["a"]) == {}


def run_unit_tests() -> None:
    tests = [
        test_lookup_compute,
        test_pull_only_rejects_compute_fallback,
        test_task_prereq_ordering,
        test_prefix_block_count_on_arrival,
        test_finish_frees_kv,
        test_local_satisfied_inflight,
    ]
    for test in tests:
        test()
        print(f"{test.__name__} ok")
