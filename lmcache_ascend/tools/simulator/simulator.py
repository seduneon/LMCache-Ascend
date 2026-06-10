from engine import Engine
from tasks import TaskPool


class Simulator:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.pool: TaskPool = engine.pool
        self.now = 0.0

    def step(self) -> bool:
        self.engine.release_arrivals(self.now)

        while self.engine.admit():
            pass

        self.pool.start_ready(self.now)

        t_tasks = self.pool.next()
        t_arrival = self.engine.next_arrival()
        candidates = [t for t in (t_tasks, t_arrival) if t is not None]
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

        while self.engine.admit():
            pass

        self.engine.check_completions()
        return True

    def run(self, max_steps: int = 100_000) -> float:
        for _ in range(max_steps):
            if not self.step():
                break
        return self.now


if __name__ == "__main__":
    from memory import Memory
    from policies import ComputeAllLookup
    from request import Request, RequestPD, RequestStatus
    from resource import ComputeResource

    pool = TaskPool()
    hbm = Memory(size=100, name="hbm")
    requests = [
        Request("r1", 0.0, ["a", "b", "c"], RequestPD.PREFILL, RequestStatus.PENDING, 30),
        Request("r2", 5.0, ["a", "d"], RequestPD.PREFILL, RequestStatus.PENDING, 20),
    ]
    engine = Engine(
        requests=requests,
        pool=pool,
        memories={"hbm": hbm},
        policy=ComputeAllLookup(),
        compute_res=ComputeResource(base_speed=1.0),
        work_per_block=1.0,
    )
    finish = Simulator(engine).run()
    print(f"finish_time={finish}")
    for req in requests:
        print(f"{req.req_id} status={req.status}")
    print(f"hbm blocks={[b.hash for b in hbm.list()]}")
