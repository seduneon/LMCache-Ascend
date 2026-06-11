from engine import Engine
from request import Request, RequestPD, RequestStatus
from tasks import TaskPool


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

    def step(self) -> bool:
        for eng in self.engines.values():
            eng.release_arrivals(self.now)

        for eng in self.engines.values():
            while eng.admit():
                pass

        self.pool.start_ready(self.now)

        t_tasks = self.pool.next()
        t_arrivals = [eng.next_arrival() for eng in self.engines.values()]
        candidates = [t for t in [t_tasks, *t_arrivals] if t is not None]
        if not candidates:
            return False

        t_next = min(candidates)
        if t_next < self.now:
            return False

        if self.pool.running():
            self.pool.advance_running_to(t_next)
            self.pool.finish_done()
            self.pool.start_ready(t_next)

        self.now = t_next

        for eng in self.engines.values():
            while eng.admit():
                pass

        for eng in self.engines.values():
            for req in eng.check_completions():
                decode_id = self.spawn_decode.get(eng.engine_id)
                if decode_id is None or req.pd != RequestPD.PREFILL:
                    continue
                decode_eng = self.engines[decode_id]
                decode_eng.schedule_request(
                    Request(
                        req.req_id,
                        self.now,
                        list(req.block_hashes),
                        RequestPD.DECODE,
                        RequestStatus.PENDING,
                        req.tokens,
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
        Request("r1", 0.0, ["a", "b", "c"], RequestPD.PREFILL, RequestStatus.PENDING, 30),
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
    for req in npu1.active:
        print(f"{req.req_id} decode active (unexpected)")
    if not npu1.waiting and not npu1.pending:
        print("r1 decode status=complete")


if __name__ == "__main__":
    run_pd_demo()
