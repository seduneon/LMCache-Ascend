from __future__ import annotations

import time
from dataclasses import dataclass

from batch_context import BatchContext
from engine import Engine
from pd import PDConfig
from request import Request, RequestPD, RequestStatus
from scheduler import Batch
from sim_log import SimLogger
from sim_progress import SimProgress
from tasks import Task, TaskPool, TaskStatus


@dataclass
class InFlightBatch:
    batch: Batch
    batch_id: int


class Simulator:
    """Discrete-event sim: each step() advances ``now`` by exactly one event."""

    def __init__(
        self,
        engines: list[Engine],
        pool: TaskPool,
        pd: PDConfig | None = None,
        log: SimLogger | None = None,
        progress: SimProgress | None = None,
    ):
        self.engines = {eng.engine_id: eng for eng in engines}
        self.pool = pool
        self.pd = pd
        self.spawn_map: dict[str, str] = pd.spawn_map if pd else {}
        if pd is not None:
            pd.validate_and_apply(self.engines)
        self.now = 0.0
        self.log = log
        self.progress = progress
        self._in_flight: dict[str, InFlightBatch] = {}
        self._next_batch_id = 0
        self.event_steps = 0

    def _alloc_batch_id(self) -> int:
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        return batch_id

    def _batch_complete(self, batch_id: int) -> bool:
        tagged = [t for t in self.pool.tasks if t.batch_id == batch_id]
        if not tagged:
            return True
        return all(task.status == TaskStatus.COMPLETED for task in tagged)

    def _idle(self) -> bool:
        for eng in self.engines.values():
            if eng.waiting or eng.running or eng.scheduler.pending:
                return False
        return not self.pool.running() and not self.pool.ready()

    def _count_new_arrivals(self, before: dict[str, int]) -> int:
        total = 0
        for eng_id, eng in self.engines.items():
            total += before[eng_id] - len(eng.scheduler.pending)
        return total

    def _schedule_engines(self) -> None:
        for eng in self.engines.values():
            if eng.engine_id in self._in_flight:
                continue
            batch = eng.schedule(self.now)
            if self.log:
                self.log.on_schedule(eng.engine_id, self, batch)
            if not batch.entries:
                continue
            batch_id = self._alloc_batch_id()
            batch_ctx = BatchContext(batch_id=batch_id, engine_id=eng.engine_id)
            tasks = eng.execute_batch(batch, self.now, batch_ctx)
            if self.log:
                self.log.on_execute(eng.engine_id, self, tasks)
            self._in_flight[eng.engine_id] = InFlightBatch(batch=batch, batch_id=batch_id)

    def _next_event_time(self) -> float | None:
        running = self.pool.running()
        if running:
            return min(task.estimated_end() for task in running)

        if self._idle():
            return None

        arrivals = [
            t
            for eng in self.engines.values()
            if (t := eng.next_arrival()) is not None
        ]
        if arrivals:
            return min(arrivals)

        self.pool.start_ready(self.now)
        running = self.pool.running()
        if running:
            return min(task.estimated_end() for task in running)

        raise RuntimeError(
            f"simulation stuck at now={self.now:.4f}: "
            "queues non-empty but no runnable tasks and no pending arrivals"
        )

    def _advance_to(self, t_next: float) -> None:
        if self.log:
            self.log.on_time_advance(self, t_next, len(self.pool.running()))
        self.now = t_next
        self.pool.advance_running_to(self.now)
        self.pool.finish_done()
        self.pool.start_ready(self.now)
        self.pool.compact()

    def _apply_completed_batches(self) -> bool:
        """Apply batches whose tasks finished; spawn PD decodes. Returns True if any applied."""
        applied = False
        completed_by_engine: dict[str, list[Request]] = {}
        remote_kv_by_engine: dict[str, list[Request]] = {}

        for eng_id, inflight in list(self._in_flight.items()):
            if not self._batch_complete(inflight.batch_id):
                continue

            eng = self.engines[eng_id]
            finished, remote_kv_done = eng.apply_batch(inflight.batch, self.now)
            completed_by_engine[eng_id] = finished
            remote_kv_by_engine[eng_id] = remote_kv_done
            del self._in_flight[eng_id]
            applied = True

            if self.log and (finished or remote_kv_done):
                self.log.on_apply(
                    eng_id,
                    self,
                    finished=finished,
                    remote_kv_done=remote_kv_done,
                )

        for eng_id, finished in completed_by_engine.items():
            for req in finished:
                if req.pd == RequestPD.DECODE and req.prefill_engine_id is not None:
                    prefill_eng = self.engines.get(req.prefill_engine_id)
                    if prefill_eng is not None:
                        prefill = next(
                            (r for r in prefill_eng.completed if r.req_id == req.req_id),
                            None,
                        )
                        if prefill is not None and prefill.kv_held_for_transfer:
                            prefill_eng.release_held_kv(req.req_id)
                            prefill.kv_held_for_transfer = False
                            if self.log:
                                self.log.on_kv_released(
                                    self, prefill_eng.engine_id, req.req_id
                                )

                decode_id = self.spawn_map.get(eng_id)
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
                        prefill_engine_id=eng_id,
                    )
                )
                if self.log:
                    self.log.on_pd_spawn(self, eng_id, decode_id, req.req_id)

        return applied

    def step(self) -> bool:
        if self.log:
            self.log.on_step_begin(self)

        pending_before = {
            eng_id: len(eng.scheduler.pending) for eng_id, eng in self.engines.items()
        }

        for eng in self.engines.values():
            eng.release_arrivals(self.now)

        if self.log:
            released = self._count_new_arrivals(pending_before)
            self.log.on_arrivals_released(self, released)

        self._schedule_engines()
        self.pool.start_ready(self.now)
        self.pool.finish_done()

        t_next = self._next_event_time()
        if t_next is None:
            return False

        if not self.pool.running():
            if self.log:
                self.log.on_wait_arrival(self, t_next)
            self._advance_to(t_next)
            self._apply_completed_batches()
            return True

        if t_next <= self.now:
            raise RuntimeError(
                f"event time did not advance at now={self.now:.6f} "
                f"(next={t_next:.6f}, running={len(self.pool.running())})"
            )

        self._advance_to(t_next)
        self._apply_completed_batches()
        return True

    def run(
        self,
        max_steps: int = 100_000,
        *,
        wall_timeout_s: float | None = None,
    ) -> float:
        if self.log:
            self.log.on_run_start(self)
        if self.progress:
            self.progress.begin()

        steps = 0
        hit_max_steps = True
        self.event_steps = 0
        wall_start = time.perf_counter() if wall_timeout_s is not None else None
        for _ in range(max_steps):
            if wall_timeout_s is not None and wall_start is not None:
                elapsed = time.perf_counter() - wall_start
                if elapsed > wall_timeout_s:
                    raise RuntimeError(
                        f"simulation exceeded wall_timeout_s={wall_timeout_s:.1f} "
                        f"after {self.event_steps} event-steps at now={self.now:.4f} "
                        f"(possible livelock; elapsed={elapsed:.1f}s)"
                    )
            steps += 1
            if not self.step():
                hit_max_steps = False
                break
            self.event_steps += 1
            if self.progress:
                self.progress.on_step(self)

        if self.progress:
            self.progress.finish(self, hit_max_steps=hit_max_steps)
        if self.log:
            self.log.on_run_end(self, steps=steps, hit_max_steps=hit_max_steps)

        return self.now


if __name__ == "__main__":
    from tests.run_tests import main

    main()
