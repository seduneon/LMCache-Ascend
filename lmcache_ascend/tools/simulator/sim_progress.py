"""Progress bar and stall detection for long Simulator runs."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from request import RequestPD, RequestStatus

if TYPE_CHECKING:
    from simulator import Simulator


@dataclass
class SimProgressConfig:
    total_requests: int
    decode_engine_id: str = "npu-1"
    update_interval: int = 1
    stall_step_limit: int = 2_000
    bar_width: int = 36


class SimProgress:
    """Live decode-completion progress with stall detection."""

    def __init__(self, config: SimProgressConfig, *, stream=None):
        self.config = config
        self.stream = stream or sys.stderr
        self.step = 0
        self._last_done = -1
        self._stall_steps = 0
        self._last_now: float | None = None
        self._started = False

    def _decode_done(self, sim: Simulator) -> int:
        eng = sim.engines.get(self.config.decode_engine_id)
        if eng is None:
            return 0
        return sum(
            1
            for req in eng.completed
            if req.pd == RequestPD.DECODE and req.status == RequestStatus.COMPLETE
        )

    def _snapshot(self, sim: Simulator) -> str:
        parts = []
        for eng_id in sorted(sim.engines):
            sched = sim.engines[eng_id].scheduler
            rkv = sum(
                1 for r in sched.waiting if r.status == RequestStatus.WAITING_REMOTE_KV
            )
            parts.append(
                f"{eng_id}:p{len(sched.pending)}w{len(sched.waiting)-rkv}"
                f"r{len(sched.running)}d{len(sched.completed)}"
            )
        return " ".join(parts)

    def _render(self, sim: Simulator, done: int) -> None:
        total = self.config.total_requests
        width = self.config.bar_width
        if total <= 0:
            pct = 1.0
            filled = width
        else:
            pct = done / total
            filled = min(width, int(width * pct))

        bar = "=" * filled + (">" if filled < width else "") + " " * (width - filled - (1 if filled < width else 0))
        line = (
            f"\r[sim] [{bar}] {done}/{total} decode "
            f"| step {self.step} | now {sim.now:.2f} | {self._snapshot(sim)}"
        )
        print(line, end="", file=self.stream, flush=True)

    def on_step(self, sim: Simulator) -> None:
        self.step += 1
        done = self._decode_done(sim)
        total = self.config.total_requests

        if (
            self.step == 1
            or self.step % self.config.update_interval == 0
            or done != self._last_done
            or done >= total
        ):
            self._render(sim, done)

        if done > self._last_done or sim.now != self._last_now:
            self._stall_steps = 0
            self._last_done = done
            self._last_now = sim.now
        elif done < total:
            self._stall_steps += 1
            if self._stall_steps >= self.config.stall_step_limit:
                raise RuntimeError(
                    "simulation stalled: no decode progress for "
                    f"{self._stall_steps} sim-steps at now={sim.now:.4f} "
                    f"({done}/{total} decode done). "
                    f"{self._snapshot(sim)}"
                )

    def finish(self, sim: Simulator, *, hit_max_steps: bool) -> None:
        done = self._decode_done(sim)
        status = "TIMEOUT" if hit_max_steps else "done"
        print(
            f"\n[sim] {status} decode {done}/{self.config.total_requests} "
            f"steps={self.step} now={sim.now:.4f}",
            file=self.stream,
            flush=True,
        )

    def begin(self) -> None:
        if self._started:
            return
        self._started = True
        print(
            f"[sim] progress tracking {self.config.total_requests} decode completions "
            f"(stall_limit={self.config.stall_step_limit} steps)",
            file=self.stream,
            flush=True,
        )
