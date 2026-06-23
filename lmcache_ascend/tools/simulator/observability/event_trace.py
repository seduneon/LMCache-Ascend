"""Structured JSONL event trace for simulator analysis."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

from simulator.policy.read_path import ReadPathDecision


def trace_config_from_env() -> EventTraceConfig:
    enabled = os.environ.get("SIM_TRACE", "").lower() in ("1", "true", "yes")
    path = os.environ.get("SIM_TRACE_PATH", "sim_trace.jsonl")
    sample = int(os.environ.get("SIM_TRACE_SAMPLE", "1"))
    tier_interval = int(os.environ.get("SIM_TRACE_TIER_INTERVAL", "1"))
    return EventTraceConfig(
        enabled=enabled,
        path=path,
        sample_every=max(1, sample),
        tier_sample_every=max(1, tier_interval),
    )


@dataclass
class EventTraceConfig:
    enabled: bool = False
    path: str = "sim_trace.jsonl"
    sample_every: int = 1
    tier_sample_every: int = 1


class EventTraceWriter:
    """Append-only JSONL trace sink."""

    def __init__(self, config: EventTraceConfig | None = None):
        self.config = config or EventTraceConfig()
        self._handle: TextIO | None = None
        self._decision_seq = 0
        self._tier_sample_seq = 0
        self._pending_estimates: dict[str, float] = {}

    def open(self) -> None:
        if not self.config.enabled or self._handle is not None:
            return
        path = Path(self.config.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def emit(self, event: dict[str, Any]) -> None:
        if not self.config.enabled or self._handle is None:
            return
        self._handle.write(json.dumps(event, separators=(",", ":")) + "\n")

    def on_decision(
        self,
        *,
        now: float,
        engine_id: str,
        req_id: str,
        decision: ReadPathDecision | None,
    ) -> None:
        if not self.config.enabled or decision is None:
            return
        self._decision_seq += 1
        if self._decision_seq % self.config.sample_every != 0:
            return
        est = decision.estimate
        pull_est = est.pull if est is not None else {}
        wait_est = est.wait if est is not None else {}
        event = {
            "kind": "decision",
            "t": now,
            "engine_id": engine_id,
            "req_id": req_id,
            "block_hash": decision.block_hash,
            "chosen": decision.chosen,
            "reason": decision.reason,
            "pull_est": pull_est,
            "compute_est": est.compute if est is not None else None,
            "wait_est": wait_est,
        }
        if decision.chosen.startswith("pull:"):
            key = f"{engine_id}:{req_id}:{decision.block_hash}"
            src = decision.chosen.split(":", 1)[1]
            self._pending_estimates[key] = pull_est.get(src, 0.0)
        self.emit(event)

    def on_task_start(
        self,
        *,
        now: float,
        engine_id: str,
        batch_id: int | None,
        task_kind: str,
        tier: str | None,
        work: float,
        resource_busy: int,
        req_id: str | None = None,
        block_hash: str | None = None,
    ) -> None:
        self.emit(
            {
                "kind": "task_start",
                "t": now,
                "engine_id": engine_id,
                "batch_id": batch_id,
                "task_kind": task_kind,
                "tier": tier,
                "work": work,
                "resource_busy": resource_busy,
                "req_id": req_id,
                "block_hash": block_hash,
            }
        )

    def on_task_end(
        self,
        *,
        now: float,
        engine_id: str,
        batch_id: int | None,
        task_kind: str,
        tier: str | None,
        start_t: float,
        req_id: str | None = None,
        block_hash: str | None = None,
    ) -> None:
        actual = now - start_t
        estimate_error: float | None = None
        if task_kind == "pull" and req_id and block_hash:
            key = f"{engine_id}:{req_id}:{block_hash}"
            predicted = self._pending_estimates.pop(key, None)
            if predicted is not None:
                estimate_error = actual - predicted
        self.emit(
            {
                "kind": "task_end",
                "t": now,
                "engine_id": engine_id,
                "batch_id": batch_id,
                "task_kind": task_kind,
                "tier": tier,
                "duration": actual,
                "estimate_error": estimate_error,
                "req_id": req_id,
                "block_hash": block_hash,
            }
        )

    def on_evict(
        self,
        *,
        now: float,
        engine_id: str,
        tier: str,
        block_hash: str,
        policy: str = "local",
    ) -> None:
        self.emit(
            {
                "kind": "evict",
                "t": now,
                "engine_id": engine_id,
                "tier": tier,
                "block_hash": block_hash,
                "policy": policy,
            }
        )

    def on_tier_sample(self, *, now: float, tiers: dict[str, dict[str, int]]) -> None:
        if not self.config.enabled:
            return
        self._tier_sample_seq += 1
        if self._tier_sample_seq % self.config.tier_sample_every != 0:
            return
        self.emit({"kind": "tier_sample", "t": now, "tiers": tiers})

    def on_request_phase(
        self,
        *,
        now: float,
        engine_id: str,
        req_id: str,
        phase: str,
    ) -> None:
        self.emit(
            {
                "kind": "request_phase",
                "t": now,
                "engine_id": engine_id,
                "req_id": req_id,
                "phase": phase,
            }
        )


def decision_to_dict(decision: ReadPathDecision) -> dict[str, Any]:
    payload = {
        "block_hash": decision.block_hash,
        "chosen": decision.chosen,
        "reason": decision.reason,
    }
    if decision.estimate is not None:
        payload["estimate"] = asdict(decision.estimate)
    return payload
