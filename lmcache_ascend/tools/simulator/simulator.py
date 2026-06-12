from engine import Engine
from pd import PDConfig
from request import Request, RequestPD, RequestStatus
from tasks import Task, TaskPool, TaskStatus


_TERMINAL = frozenset({TaskStatus.COMPLETED})


class Simulator:
    def __init__(
        self,
        engines: list[Engine],
        pool: TaskPool,
        pd: PDConfig | None = None,
    ):
        self.engines = {eng.engine_id: eng for eng in engines}
        self.pool = pool
        self.pd = pd
        self.spawn_map: dict[str, str] = pd.spawn_map if pd else {}
        if pd is not None:
            pd.validate_and_apply(self.engines)
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
        remote_kv_by_engine: dict[str, list[Request]] = {}
        for eng in self.engines.values():
            finished, remote_kv_done = eng.apply_batch(batches[eng.engine_id])
            completed_by_engine[eng.engine_id] = finished
            remote_kv_by_engine[eng.engine_id] = remote_kv_done

        for eng_id, remote_done in remote_kv_by_engine.items():
            for req in remote_done:
                if req.prefill_engine_id is None:
                    continue
                prefill_eng = self.engines[req.prefill_engine_id]
                prefill = next(
                    (r for r in prefill_eng.completed if r.req_id == req.req_id),
                    None,
                )
                if prefill is not None and prefill.kv_held_for_transfer:
                    prefill_eng.release_held_kv(req.req_id)
                    prefill.kv_held_for_transfer = False

        for eng in self.engines.values():
            for req in completed_by_engine[eng.engine_id]:
                decode_id = self.spawn_map.get(eng.engine_id)
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
                        prefill_engine_id=eng.engine_id,
                    )
                )

        return True

    def run(self, max_steps: int = 100_000) -> float:
        for _ in range(max_steps):
            if not self.step():
                break
        return self.now


if __name__ == "__main__":
    from tests.run_tests import main

    main()
