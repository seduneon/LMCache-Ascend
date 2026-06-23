from __future__ import annotations

import time

from .engine import Engine
from .pd import DecodeSpawn, KvRelease, PDConfig, SimEvent
from .plan import BatchPlan
from simulator.core.request import Request, RequestPD, RequestStatus
from simulator.observability.event_trace import EventTraceWriter, trace_config_from_env
from simulator.observability.sim_log import SimLogger
from simulator.observability.sim_progress import SimProgress
from .tasks import BatchLoadTask, EvictTask, ForwardTask, StoreTask, Task, TaskPool, TaskStatus


class Simulator:
    """Discrete-event sim: each step() advances ``now`` by exactly one event."""

    def __init__(
        self,
        engines: list[Engine],
        pool: TaskPool,
        pd: PDConfig | None = None,
        log: SimLogger | None = None,
        progress: SimProgress | None = None,
        event_trace: EventTraceWriter | None = None,
    ):
        self.engines = {eng.engine_id: eng for eng in engines}
        self.pool = pool
        self.pd = pd
        self.routing = pd.resolved_routing() if pd is not None else None
        if pd is not None:
            pd.validate_and_apply(self.engines)
        self.now = 0.0
        self.log = log
        self.progress = progress
        self.event_trace = event_trace
        if self.event_trace is not None:
            self.event_trace.open()
        for eng in engines:
            eng.event_trace = event_trace
        self._in_flight: dict[str, BatchPlan] = {}
        self.event_steps = 0

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
            plan = eng.try_schedule_and_execute(self.now)
            if plan is None:
                continue
            if self.log:
                self.log.on_schedule(eng.engine_id, self, plan)
                tagged = [t for t in self.pool.tasks if t.batch_id == plan.batch_id]
                self.log.on_execute(eng.engine_id, self, tagged)
            self._in_flight[eng.engine_id] = plan

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

    def _trace_task_kind(self, task: Task) -> str:
        if isinstance(task, BatchLoadTask):
            return "pull"
        if isinstance(task, StoreTask):
            return "store"
        if isinstance(task, ForwardTask):
            return "forward"
        if isinstance(task, EvictTask):
            return "evict"
        return type(task).__name__

    def _emit_task_starts(self) -> None:
        if self.event_trace is None:
            return
        for task in self.pool.running():
            meta = task.trace_meta
            if meta is None or meta.get("started"):
                continue
            meta["started"] = True
            meta["start_t"] = task.now
            self.event_trace.on_task_start(
                now=task.now,
                engine_id=meta.get("engine_id", ""),
                batch_id=meta.get("batch_id"),
                task_kind=self._trace_task_kind(task),
                tier=getattr(task, "memory", None) and task.memory.name,
                work=task.work_left,
                resource_busy=task.resource.queued_load(),
            )

    def _emit_task_ends(self) -> None:
        if self.event_trace is None:
            return
        for task in self.pool.tasks:
            if task.status != TaskStatus.COMPLETED:
                continue
            meta = task.trace_meta
            if meta is None or meta.get("ended"):
                continue
            meta["ended"] = True
            start_t = meta.get("start_t", self.now)
            self.event_trace.on_task_end(
                now=self.now,
                engine_id=meta.get("engine_id", ""),
                batch_id=meta.get("batch_id"),
                task_kind=self._trace_task_kind(task),
                tier=getattr(task, "memory", None) and task.memory.name,
                start_t=start_t,
            )

    def _advance_to(self, t_next: float) -> None:
        if self.log:
            self.log.on_time_advance(self, t_next, len(self.pool.running()))
        if self.event_trace is not None:
            tiers = {
                eng.local_memory: {
                    "used": eng.memories[eng.local_memory].used_size(),
                    "size": eng.memories[eng.local_memory].size,
                }
                for eng in self.engines.values()
            }
            for eng in self.engines.values():
                for name, mem in eng.memories.items():
                    if name not in tiers:
                        tiers[name] = {"used": mem.used_size(), "size": mem.size}
            self.event_trace.on_tier_sample(now=t_next, tiers=tiers)
        self.now = t_next
        self.pool.advance_running_to(self.now)
        self.pool.finish_done()
        self._emit_task_ends()
        self.pool.start_ready(self.now)
        self._emit_task_starts()
        self.pool.compact()

    def dispatch(self, event: SimEvent) -> None:
        if isinstance(event, KvRelease):
            prefill_eng = self.engines.get(event.prefill_engine_id)
            if prefill_eng is None:
                return
            prefill = next(
                (r for r in prefill_eng.completed if r.req_id == event.req_id),
                None,
            )
            if prefill is not None and prefill.kv_held_for_transfer:
                prefill_eng.release_held_kv(event.req_id)
                prefill.kv_held_for_transfer = False
                if self.log:
                    self.log.on_kv_released(self, event.prefill_engine_id, event.req_id)
            return

        decode_eng = self.engines.get(event.decode_engine_id)
        if decode_eng is None:
            return
        decode_eng.schedule_request(
            Request(
                event.req_id,
                event.arrival_time,
                list(event.prefix_blocks),
                RequestPD.DECODE,
                RequestStatus.PENDING,
                max_output_blocks=event.max_output_blocks,
                prefix_block_count=event.prefix_block_count,
                prefill_engine_id=event.prefill_engine_id,
            )
        )
        if self.log:
            self.log.on_pd_spawn(
                self,
                event.prefill_engine_id,
                event.decode_engine_id,
                event.req_id,
            )

    def _events_from_commit(self, eng_id: str, finished) -> list[SimEvent]:
        events: list[SimEvent] = []
        for req in finished:
            if req.pd == RequestPD.DECODE and req.prefill_engine_id is not None:
                events.append(
                    KvRelease(req_id=req.req_id, prefill_engine_id=req.prefill_engine_id)
                )
            if self.routing is None or req.pd != RequestPD.PREFILL:
                continue
            decode_id = self.routing.route(req, eng_id)
            events.append(
                DecodeSpawn(
                    req_id=req.req_id,
                    arrival_time=self.now,
                    prefix_blocks=tuple(req.block_hashes[: req.prefix_block_count]),
                    max_output_blocks=req.max_output_blocks,
                    prefix_block_count=req.prefix_block_count,
                    prefill_engine_id=eng_id,
                    decode_engine_id=decode_id,
                )
            )
        return events

    def _apply_completed_batches(self) -> bool:
        applied = False
        for eng_id, plan in list(self._in_flight.items()):
            if not self._batch_complete(plan.batch_id):
                continue
            eng = self.engines[eng_id]
            finished, remote_kv_done = eng.apply_plan(plan, self.now)
            del self._in_flight[eng_id]
            applied = True
            if self.log and (finished or remote_kv_done):
                self.log.on_apply(
                    eng_id,
                    self,
                    finished=finished,
                    remote_kv_done=remote_kv_done,
                )
            for event in self._events_from_commit(eng_id, finished):
                self.dispatch(event)
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
            self.log.on_arrivals_released(self, self._count_new_arrivals(pending_before))

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
        if self.event_trace is not None:
            self.event_trace.close()

        return self.now


if __name__ == "__main__":
    from simulator.tests.run_tests import main

    main()
