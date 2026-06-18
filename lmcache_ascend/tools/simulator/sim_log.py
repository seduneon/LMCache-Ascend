"""Optional progress logging for Simulator runs (``SIM_LOG=1``)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from request import RequestStatus
from tasks import BatchLoadTask, EvictTask, ForwardTask, StoreTask, Task

if TYPE_CHECKING:
    from engine import Engine
    from scheduler import Batch
    from simulator import Simulator


def log_config_from_env() -> SimLogConfig:
    enabled = os.environ.get("SIM_LOG", "").lower() in ("1", "true", "yes")
    detail = os.environ.get("SIM_LOG_DETAIL", "").lower() in ("1", "true", "yes")
    interval = int(os.environ.get("SIM_LOG_INTERVAL", "100"))
    return SimLogConfig(enabled=enabled, detail=detail, step_interval=max(1, interval))


@dataclass
class SimLogConfig:
    enabled: bool = False
    detail: bool = False
    step_interval: int = 100


@dataclass
class _EngineCounts:
    pending: int = 0
    waiting: int = 0
    remote_kv: int = 0
    running: int = 0
    completed: int = 0


class SimLogger:
    def __init__(self, config: SimLogConfig | None = None, *, stream=None):
        self.config = config or SimLogConfig()
        self.stream = stream or sys.stderr
        self.step = 0

    def _write(self, msg: str) -> None:
        if not self.config.enabled:
            return
        print(msg, file=self.stream, flush=True)

    def _engine_counts(self, eng: Engine) -> _EngineCounts:
        sched = eng.scheduler
        remote_kv = sum(
            1 for req in sched.waiting if req.status == RequestStatus.WAITING_REMOTE_KV
        )
        return _EngineCounts(
            pending=len(sched.pending),
            waiting=len(sched.waiting) - remote_kv,
            remote_kv=remote_kv,
            running=len(sched.running),
            completed=len(sched.completed),
        )

    def _format_engines(self, engines: dict[str, Engine]) -> str:
        parts = []
        for eng_id in sorted(engines):
            c = self._engine_counts(engines[eng_id])
            parts.append(
                f"{eng_id}[pend={c.pending} wait={c.waiting} rkv={c.remote_kv} "
                f"run={c.running} done={c.completed}]"
            )
        return " ".join(parts)

    @staticmethod
    def _task_kind(task: Task) -> str:
        if isinstance(task, ForwardTask):
            return f"fwd×{len(task.blocks)}"
        if isinstance(task, BatchLoadTask):
            return f"pull×{len(task.blocks)}"
        if isinstance(task, StoreTask):
            return "store"
        if isinstance(task, EvictTask):
            return "evict"
        return type(task).__name__

    def _format_batch(self, batch: Batch) -> str:
        if not batch.entries and not batch.preempted:
            return "empty"
        remote = sum(1 for e in batch.entries if e.remote_kv)
        compute = sum(
            1
            for e in batch.entries
            if any(a == "compute" for a in e.result.blocks.values())
        )
        pull = sum(
            1
            for e in batch.entries
            if any(isinstance(a, tuple) and a[0] == "pull" for a in e.result.blocks.values())
        )
        return (
            f"entries={len(batch.entries)} tokens={batch.total_num_scheduled_tokens} "
            f"remote_kv={remote} compute={compute} pull={pull} "
            f"preempted={len(batch.preempted)}"
        )

    def on_run_start(self, sim: Simulator) -> None:
        if not self.config.enabled:
            return
        self._write(
            f"[sim] start engines={len(sim.engines)} pd_spawn={sim.spawn_map or '{}'}"
        )

    def on_run_end(self, sim: Simulator, *, steps: int, hit_max_steps: bool) -> None:
        if not self.config.enabled:
            return
        status = "max_steps" if hit_max_steps else "done"
        self._write(
            f"[sim] {status} steps={steps} now={sim.now:.4f} "
            f"{self._format_engines(sim.engines)}"
        )

    def on_step_begin(self, sim: Simulator) -> None:
        if not self.config.enabled:
            return
        self.step += 1
        if self.config.detail or self.step == 1 or self.step % self.config.step_interval == 0:
            self._write(
                f"[sim] step={self.step} now={sim.now:.4f} "
                f"{self._format_engines(sim.engines)}"
            )

    def on_arrivals_released(self, sim: Simulator, count: int) -> None:
        if not self.config.enabled or not self.config.detail or count == 0:
            return
        self._write(f"[sim] t={sim.now:.4f} released {count} arrival(s)")

    def on_schedule(self, engine_id: str, sim: Simulator, batch: Batch) -> None:
        if not self.config.enabled:
            return
        if not self.config.detail and not batch.entries and not batch.preempted:
            return
        self._write(
            f"[sim] t={sim.now:.4f} schedule {engine_id} {self._format_batch(batch)}"
        )
        if batch.preempted and (self.config.detail or self.step % self.config.step_interval == 0):
            ids = ",".join(r.req_id for r in batch.preempted[:8])
            suffix = "..." if len(batch.preempted) > 8 else ""
            self._write(f"[sim] t={sim.now:.4f} preempted on {engine_id}: {ids}{suffix}")

    def on_execute(self, engine_id: str, sim: Simulator, tasks: list[Task]) -> None:
        if not self.config.enabled or not self.config.detail or not tasks:
            return
        kinds: dict[str, int] = {}
        for task in tasks:
            kind = self._task_kind(task)
            kinds[kind] = kinds.get(kind, 0) + 1
        summary = " ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        self._write(f"[sim] t={sim.now:.4f} execute {engine_id} tasks: {summary}")

    def on_time_advance(self, sim: Simulator, t_next: float, running: int) -> None:
        if not self.config.enabled or not self.config.detail:
            return
        self._write(
            f"[sim] event {sim.now:.4f} -> {t_next:.4f} running_tasks={running}"
        )

    def on_apply(
        self,
        engine_id: str,
        sim: Simulator,
        *,
        finished: list,
        remote_kv_done: list,
    ) -> None:
        if not self.config.enabled:
            return
        if not finished and not remote_kv_done:
            return
        if not self.config.detail and not (
            self.step % self.config.step_interval == 0 or remote_kv_done
        ):
            return
        parts = []
        if finished:
            parts.append(f"finished={len(finished)}")
        if remote_kv_done:
            parts.append(f"remote_kv={len(remote_kv_done)}")
        self._write(f"[sim] t={sim.now:.4f} apply {engine_id} {' '.join(parts)}")

    def on_pd_spawn(self, sim: Simulator, prefill_id: str, decode_id: str, req_id: str) -> None:
        if not self.config.enabled or not self.config.detail:
            return
        self._write(
            f"[sim] t={sim.now:.4f} pd spawn {req_id} {prefill_id}->{decode_id}"
        )

    def on_kv_released(self, sim: Simulator, prefill_id: str, req_id: str) -> None:
        if not self.config.enabled or not self.config.detail:
            return
        self._write(f"[sim] t={sim.now:.4f} kv released {req_id} on {prefill_id}")

    def on_wait_arrival(self, sim: Simulator, t_next: float) -> None:
        if not self.config.enabled:
            return
        self._write(
            f"[sim] t={sim.now:.4f} idle batch, jump to arrival @ {t_next:.4f} "
            f"{self._format_engines(sim.engines)}"
        )

    def milestone(self, sim: Simulator, msg: str) -> None:
        if not self.config.enabled:
            return
        self._write(f"[sim] t={sim.now:.4f} {msg} {self._format_engines(sim.engines)}")
