from engine import Engine
from request import Request, RequestPD, RequestStatus
from tasks import Task, TaskPool, TaskStatus


_TERMINAL = frozenset({TaskStatus.COMPLETED})


class Simulator:
    def __init__(
        self,
        engines: list[Engine],
        pool: TaskPool,
        spawn_decode: dict[str, str] | None = None,
    ):
        self.engines = {eng.engine_id: eng for eng in engines}
        self.pool = pool
        self.spawn_decode = spawn_decode or {}
        self.now = 0.0

    def _batch_done(self, tasks: list[Task]) -> bool:
        return not tasks or all(t.status in _TERMINAL for t in tasks)

    def _drain_batch(self, tasks: list[Task]) -> None:
        self.pool.start_ready(self.now)
        while not self._batch_done(tasks):
            running = [t for t in tasks if t.status == TaskStatus.RUNNING]
            if not running:
                self.pool.start_ready(self.now)
                running = [t for t in tasks if t.status == TaskStatus.RUNNING]
            if not running:
                break

            t_next = min(t.estimated_end() for t in running)
            if t_next < self.now:
                break

            self.pool.advance_running_to(t_next)
            self.pool.finish_done()
            self.now = t_next
            self.pool.start_ready(self.now)

    def _idle(self) -> bool:
        for eng in self.engines.values():
            if eng.waiting or eng.running or eng.scheduler.pending:
                return False
        return not self.pool.running()

    def step(self) -> bool:
        for eng in self.engines.values():
            eng.release_arrivals(self.now)

        batches = {}
        step_tasks: list[Task] = []
        for eng in self.engines.values():
            batch = eng.schedule()
            batches[eng.engine_id] = batch
            if batch.entries:
                step_tasks.extend(eng.execute_batch(batch))

        if not step_tasks:
            if self._idle():
                return False
            t_arrivals = [eng.next_arrival() for eng in self.engines.values()]
            arrivals = [t for t in t_arrivals if t is not None]
            if arrivals:
                self.now = min(arrivals)
                return True
            return False

        self._drain_batch(step_tasks)

        completed_by_engine: dict[str, list[Request]] = {}
        for eng in self.engines.values():
            completed_by_engine[eng.engine_id] = eng.apply_batch(
                batches[eng.engine_id]
            )

        for eng in self.engines.values():
            for req in completed_by_engine[eng.engine_id]:
                decode_id = self.spawn_decode.get(eng.engine_id)
                if decode_id is None or req.pd != RequestPD.PREFILL:
                    continue
                decode_eng = self.engines[decode_id]
                decode_eng.schedule_request(
                    Request(
                        req.req_id,
                        self.now,
                        list(req.block_hashes[: req.prefix_block_count]),
                        RequestPD.DECODE,
                        RequestStatus.PENDING,
                        max_output_blocks=req.max_output_blocks,
                        prefix_block_count=req.prefix_block_count,
                    )
                )

        return True

    def run(self, max_steps: int = 100_000) -> float:
        for _ in range(max_steps):
            if not self.step():
                break
        return self.now


def run_pd_demo() -> None:
    from memory import Memory
    from policies import LookupPolicy
    from resource import BandwidthResource, ComputeResource

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
        policy=LookupPolicy(local_memory="npu-0:hbm"),
        compute_res=ComputeResource(base_speed=1.0),
        work_per_block=1.0,
    )
    npu1 = Engine(
        engine_id="npu-1",
        requests=[],
        pool=pool,
        memories=memories,
        local_memory="npu-1:hbm",
        policy=LookupPolicy(local_memory="npu-1:hbm", pull_sources=["npu-0:hbm"]),
        compute_res=ComputeResource(base_speed=1.0),
        bandwidth_res=BandwidthResource(base_speed=1.0),
        work_per_block=1.0,
        work_per_transfer=1.0,
    )
    sim = Simulator([npu0, npu1], pool, spawn_decode={"npu-0": "npu-1"})
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


def run_deadlock_test() -> None:
    """Two running requests fill memory; decode preemption must unblock progress."""
    from memory import Memory
    from policies import LookupPolicy
    from resource import ComputeResource

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
        policy=LookupPolicy(local_memory="npu-0:hbm"),
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
    print(f"deadlock_test finish_time={finish}")
    print(f"r1 preemptions={by_id['r1'].num_preemptions} r2 preemptions={by_id['r2'].num_preemptions}")
    print(f"r1 computed={by_id['r1'].num_computed_blocks}/{by_id['r1'].total_blocks()}")
    print(f"r2 computed={by_id['r2'].num_computed_blocks}/{by_id['r2'].total_blocks()}")


def run_limits_test() -> None:
    """max_num_seqs and max_num_batched_tokens (vLLM scheduler limits)."""
    from memory import Memory
    from policies import LookupPolicy
    from resource import ComputeResource
    from scheduler import Scheduler

    memories = {"hbm": Memory(size=100, name="hbm")}
    policy = LookupPolicy(local_memory="hbm")
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "deadlock":
        run_deadlock_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "limits":
        run_limits_test()
    else:
        run_pd_demo()
        print()
        run_deadlock_test()
        print()
        run_limits_test()
