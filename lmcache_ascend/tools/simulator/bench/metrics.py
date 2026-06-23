"""Sweep metrics: row schema, aggregation, and CSV helpers."""

from __future__ import annotations

import statistics
import csv
from dataclasses import dataclass

from simulator.runtime.engine import Engine
from simulator.core.memory import Memory
from simulator.core.request import RequestPD, RequestStatus
from .presets import PresetSpec
from .workload import WorkloadConfig
from .sweep_config import SimConfig

def _local_tier_end_state_ok(
    preset: PresetSpec,
    mem: Memory,
    *,
    is_prefill: bool,
) -> tuple[bool, str]:
    if mem.used_size() > mem.size:
        return False, f"over_capacity used={mem.used_size()} size={mem.size}"
    blocks = mem.list()
    if any(block.holders for block in blocks):
        held = sum(1 for block in blocks if block.holders)
        return False, f"held_blocks={held}"
    if mem.used_size() == 0:
        return True, ""
    retain = (
        preset.prefill_retain_prefix_cache
        if is_prefill
        else preset.decode_retain_prefix_cache
    )
    if retain and all(mem.can_evict_block(block) for block in blocks):
        return True, ""
    return False, f"used={mem.used_size()}"


@dataclass
class SweepRow:
    preset: str
    seed: int
    num_requests: int
    status: str
    admittable_requests: int = 0
    rejected_requests: int = 0
    error: str = ""
    steps: int = 0
    finish_time: float = 0.0
    wall_seconds: float = 0.0
    preemptions: int = 0
    decode_p50_latency: float = 0.0
    decode_p99_latency: float = 0.0
    decode_pulls: int = 0
    decode_computes: int = 0
    decode_local_hits: int = 0
    decode_prefix_pulls: int = 0
    decode_prefix_computes: int = 0
    decode_prefix_local_hits: int = 0
    decode_prefix_dram_pulls: int = 0
    decode_evictions: int = 0
    prefill_computes: int = 0
    prefill_evictions: int = 0
    prefill_local_hits: int = 0
    prefill_hit_ratio: float = 0.0
    decode_hit_ratio: float = 0.0
    prefix_pull_ratio: float = 0.0
    dram_hit_rate: float = 0.0
    lifecycle_hbm_frees: int = 0
    tier_evictions: int = 0
    pull_ratio: float = 0.0
    dram_slots_used: int = 0
    ssd_slots_used: int = 0
    peak_duplicate_count: int = 0
    tier_used_at_end: str = ""
    read_path: str = ""


SWEEP_CSV_FIELDS = [f.name for f in SweepRow.__dataclass_fields__.values()]
SWEEP_COMPARE_FIELDS = [name for name in SWEEP_CSV_FIELDS if name not in ("preset", "seed")]
SWEEP_COMPARE_NUMERIC = frozenset(
    name
    for name in SWEEP_COMPARE_FIELDS
    if name
    not in (
        "status",
        "error",
        "num_requests",
        "tier_used_at_end",
    )
)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)

def _aggregate_metrics(
    preset: str,
    seed: int,
    workload: WorkloadConfig,
    engines: dict[str, Engine],
    topo,
    *,
    steps: int,
    finish_time: float,
    wall_seconds: float,
    memories: dict,
    tier_roles: dict[str, str],
    admittable_requests: int,
    rejected_requests: int,
    read_path: str = "",
) -> SweepRow:
    prefill_engs = [engines[eid] for eid in topo.prefill_ids]
    decode_engs = [engines[eid] for eid in topo.decode_ids]
    decode_completed = [
        r
        for eng in decode_engs
        for r in eng.completed
        if r.pd == RequestPD.DECODE
    ]
    prefill_completed = [r for eng in prefill_engs for r in eng.completed]

    decode_latencies = [
        r.metrics.latency for r in decode_completed if r.metrics.latency is not None
    ]

    decode_pulls = sum(r.metrics.pulls for r in decode_completed)
    decode_computes = sum(r.metrics.computes for r in decode_completed)
    decode_local = sum(r.metrics.local_hits for r in decode_completed)
    decode_prefix_pulls = sum(r.metrics.prefix_pulls for r in decode_completed)
    decode_prefix_computes = sum(r.metrics.prefix_computes for r in decode_completed)
    decode_prefix_local = sum(r.metrics.prefix_local_hits for r in decode_completed)
    decode_prefix_dram_pulls = sum(
        r.metrics.prefix_dram_pulls for r in decode_completed
    )
    decode_evictions = sum(r.metrics.evictions for r in decode_completed)

    decode_actions = decode_pulls + decode_computes
    pull_ratio = decode_pulls / decode_actions if decode_actions else 0.0

    prefix_resolve = decode_prefix_pulls + decode_prefix_computes
    prefix_pull_ratio = (
        decode_prefix_pulls / prefix_resolve if prefix_resolve else 0.0
    )
    dram_hit_rate = (
        decode_prefix_dram_pulls / decode_prefix_pulls
        if decode_prefix_pulls
        else 0.0
    )

    preemptions = sum(
        r.metrics.preemptions
        for eng in prefill_engs + decode_engs
        for r in eng.completed
    )
    prefill_computes = sum(r.metrics.computes for r in prefill_completed)
    prefill_evictions = sum(r.metrics.evictions for r in prefill_completed)
    prefill_local = sum(r.metrics.local_hits for r in prefill_completed)
    prefill_prefix_local = sum(
        r.metrics.prefix_local_hits for r in prefill_completed
    )
    prefill_actions = (
        sum(r.metrics.prefix_computes for r in prefill_completed) + prefill_prefix_local
    )
    prefill_hit_ratio = (
        prefill_prefix_local / prefill_actions if prefill_actions else 0.0
    )
    decode_prefix_actions = (
        decode_prefix_pulls + decode_prefix_computes + decode_prefix_local
    )
    decode_hit_ratio = (
        decode_prefix_local / decode_prefix_actions
        if decode_prefix_actions
        else 0.0
    )

    lifecycle_hbm_frees = sum(
        mem.lifecycle_frees
        for name, mem in memories.items()
        if tier_roles.get(name) == "local"
    )
    tier_evictions = sum(
        mem.tier_evictions
        for name, mem in memories.items()
        if tier_roles.get(name) == "downstream"
    )

    tier_used_at_end = ";".join(
        f"{name}:{mem.used_size()}/{mem.size}" for name, mem in sorted(memories.items())
    )
    dram = memories.get("npu-0:dram")
    dram_slots_used = dram.used_size() if dram is not None else 0
    ssd = memories.get("npu-0:ssd")
    ssd_slots_used = ssd.used_size() if ssd is not None else 0
    peak_dup = max(eng.cache.peak_duplicate_count for eng in engines.values())

    return SweepRow(
        preset=preset,
        seed=seed,
        num_requests=workload.num_requests,
        admittable_requests=admittable_requests,
        rejected_requests=rejected_requests,
        status="ok",
        steps=steps,
        finish_time=finish_time,
        wall_seconds=wall_seconds,
        preemptions=preemptions,
        decode_p50_latency=_percentile(decode_latencies, 50),
        decode_p99_latency=_percentile(decode_latencies, 99),
        decode_pulls=decode_pulls,
        decode_computes=decode_computes,
        decode_local_hits=decode_local,
        decode_prefix_pulls=decode_prefix_pulls,
        decode_prefix_computes=decode_prefix_computes,
        decode_prefix_local_hits=decode_prefix_local,
        decode_prefix_dram_pulls=decode_prefix_dram_pulls,
        decode_evictions=decode_evictions,
        prefill_computes=prefill_computes,
        prefill_evictions=prefill_evictions,
        prefill_local_hits=prefill_local,
        prefill_hit_ratio=prefill_hit_ratio,
        decode_hit_ratio=decode_hit_ratio,
        prefix_pull_ratio=prefix_pull_ratio,
        dram_hit_rate=dram_hit_rate,
        lifecycle_hbm_frees=lifecycle_hbm_frees,
        tier_evictions=tier_evictions,
        pull_ratio=pull_ratio,
        dram_slots_used=dram_slots_used,
        ssd_slots_used=ssd_slots_used,
        peak_duplicate_count=peak_dup,
        tier_used_at_end=tier_used_at_end,
        read_path=read_path,
    )

def _row_values(row: SweepRow) -> dict[str, object]:
    return {field: getattr(row, field) for field in SWEEP_CSV_FIELDS}


def _aggregate_preset_rows(rows: list[SweepRow]) -> dict[str, object]:
    """Collapse seed runs for one preset into a single compare column."""
    if not rows:
        return {field: "" for field in SWEEP_COMPARE_FIELDS}

    failures = [r for r in rows if r.status != "ok"]
    ok_rows = [r for r in rows if r.status == "ok"]
    if not ok_rows:
        return _row_values(failures[0])

    if len(ok_rows) == 1:
        values = _row_values(ok_rows[0])
        values["seed_runs"] = 1
        return values

    values = _row_values(ok_rows[0])
    values["status"] = "ok"
    values["error"] = ""
    values["seed_runs"] = len(ok_rows)
    for name in SWEEP_COMPARE_NUMERIC:
        values[name] = statistics.mean(getattr(r, name) for r in ok_rows)
    if failures:
        values["status"] = "partial_fail"
        values["error"] = f"{len(failures)}/{len(rows)} runs failed"
    return values


def _compare_metric_names() -> list[str]:
    names = ["seed_runs", *SWEEP_COMPARE_FIELDS]
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _format_compare_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_raw_csv(path: str, rows: list[SweepRow]) -> None:
    """One row per (preset, seed) run — full detail."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_values(row))


def _write_compare_csv(
    path: str,
    rows: list[SweepRow],
    *,
    preset_order: tuple[str, ...],
) -> None:
    """Wide compare table: rows are metrics, columns are presets."""
    by_preset: dict[str, list[SweepRow]] = {}
    for row in rows:
        by_preset.setdefault(row.preset, []).append(row)

    columns = {
        preset: _aggregate_preset_rows(by_preset.get(preset, ()))
        for preset in preset_order
    }

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", *preset_order])
        for metric in _compare_metric_names():
            writer.writerow(
                [metric]
                + [
                    _format_compare_cell(columns[preset].get(metric, ""))
                    for preset in preset_order
                ]
            )
