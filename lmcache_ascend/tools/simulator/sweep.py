"""Policy sweep harness: fixed workloads, aggregated metrics, CSV output."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass, field, replace

from .engine import Engine
from .memory import Memory
from .pd import PDConfig
from .presets import (
    DEFAULT_PRESET_NAMES,
    PRESETS,
    EngineBuildConfig,
    PresetSpec,
    build_pd_engines,
)
from .request import RequestPD, RequestStatus
from .sim_log import SimLogConfig, SimLogger
from .sim_progress import SimProgress, SimProgressConfig
from .simulator import Simulator
from .tasks import TaskPool
from .topology import SimResources
from .trace import (
    DEFAULT_TRACE_PATH,
    MOONCAKE_TOKENS_PER_BLOCK,
    partition_by_hbm,
)
from .workload import WorkloadConfig, build_workload


@dataclass(frozen=True)
class SimConfig:
    """Resource and scheduler knobs shared across policy presets."""

    hbm_size: int = 40
    dram_size: int = 80
    ssd_size: int = 160
    dram_chunk_blocks: int = 4
    ssd_chunk_blocks: int = 4
    ssd_write_speed: float = 8.0
    max_num_seqs: int = 12
    max_num_batched_tokens: int = 24
    compute_speed: float = 64.0
    link_speed: float = 32.0
    link_latency: float = 0.01
    work_per_block: float = 1.0
    work_per_transfer: float = 1.0
    max_steps: int = 5_000_000
    max_steps_per_request: int = 500
    wall_timeout_s: float | None = None
    show_progress: bool = False

    def resources(self) -> SimResources:
        return SimResources(
            hbm_size=self.hbm_size,
            dram_size=self.dram_size,
            ssd_size=self.ssd_size,
            dram_chunk_blocks=self.dram_chunk_blocks,
            ssd_chunk_blocks=self.ssd_chunk_blocks,
        )

    def engine_build(self) -> EngineBuildConfig:
        return EngineBuildConfig(
            compute_speed=self.compute_speed,
            link_speed=self.link_speed,
            link_latency=self.link_latency,
            ssd_write_speed=self.ssd_write_speed,
            work_per_block=self.work_per_block,
            work_per_transfer=self.work_per_transfer,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            resources=self.resources(),
        )


HBM_TIER_KEYS: tuple[str, ...] = ("npu-0:hbm", "npu-1:hbm")


def _hbm_end_state_ok(
    preset: PresetSpec, tier_key: str, mem: Memory
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
        preset.prefill_retain_hbm_prefix_cache
        if tier_key == "npu-0:hbm"
        else preset.decode_retain_hbm_prefix_cache
        if tier_key == "npu-1:hbm"
        else False
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
    decode_evictions: int = 0
    prefill_computes: int = 0
    prefill_evictions: int = 0
    prefill_local_hits: int = 0
    prefill_hit_ratio: float = 0.0
    decode_hit_ratio: float = 0.0
    lifecycle_hbm_frees: int = 0
    tier_evictions: int = 0
    pull_ratio: float = 0.0
    dram_slots_used: int = 0
    ssd_slots_used: int = 0
    peak_duplicate_count: int = 0
    tier_used_at_end: str = ""


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


def build_engines(
    requests: list,
    pool: TaskPool,
    preset: PresetSpec,
    sim_cfg: SimConfig,
    *,
    rng_seed: int = 0,
) -> tuple[Engine, Engine, dict]:
    npu0, npu1, topo = build_pd_engines(
        requests,
        pool,
        preset,
        cfg=sim_cfg.engine_build(),
        resources=sim_cfg.resources(),
        rng_seed=rng_seed,
    )
    return npu0, npu1, topo.memories


def _aggregate_metrics(
    preset: str,
    seed: int,
    workload: WorkloadConfig,
    npu0: Engine,
    npu1: Engine,
    *,
    steps: int,
    finish_time: float,
    wall_seconds: float,
    memories: dict,
    admittable_requests: int,
    rejected_requests: int,
) -> SweepRow:
    decode_latencies = [
        r.metrics.latency
        for r in npu1.completed
        if r.pd == RequestPD.DECODE and r.metrics.latency is not None
    ]

    decode_pulls = sum(r.metrics.pulls for r in npu1.completed)
    decode_computes = sum(r.metrics.computes for r in npu1.completed)
    decode_local = sum(r.metrics.local_hits for r in npu1.completed)
    decode_evictions = sum(r.metrics.evictions for r in npu1.completed)

    decode_actions = decode_pulls + decode_computes
    pull_ratio = decode_pulls / decode_actions if decode_actions else 0.0

    preemptions = sum(r.metrics.preemptions for r in npu0.completed + npu1.completed)
    prefill_computes = sum(r.metrics.computes for r in npu0.completed)
    prefill_evictions = sum(r.metrics.evictions for r in npu0.completed)
    prefill_local = sum(r.metrics.local_hits for r in npu0.completed)
    prefill_actions = prefill_computes + prefill_local
    prefill_hit_ratio = prefill_local / prefill_actions if prefill_actions else 0.0
    decode_hit_ratio = decode_local / decode_actions if decode_actions else 0.0

    lifecycle_hbm_frees = sum(
        mem.lifecycle_frees for name, mem in memories.items() if name in HBM_TIER_KEYS
    )
    tier_evictions = sum(
        mem.tier_evictions for name, mem in memories.items() if name not in HBM_TIER_KEYS
    )

    tier_used_at_end = ";".join(
        f"{name}:{mem.used_size()}/{mem.size}" for name, mem in sorted(memories.items())
    )
    dram = memories.get("npu-0:dram")
    dram_slots_used = dram.used_size() if dram is not None else 0
    ssd = memories.get("npu-0:ssd")
    ssd_slots_used = ssd.used_size() if ssd is not None else 0
    peak_dup = max(npu0._peak_duplicate_count, npu1._peak_duplicate_count)

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
        decode_evictions=decode_evictions,
        prefill_computes=prefill_computes,
        prefill_evictions=prefill_evictions,
        prefill_local_hits=prefill_local,
        prefill_hit_ratio=prefill_hit_ratio,
        decode_hit_ratio=decode_hit_ratio,
        lifecycle_hbm_frees=lifecycle_hbm_frees,
        tier_evictions=tier_evictions,
        pull_ratio=pull_ratio,
        dram_slots_used=dram_slots_used,
        ssd_slots_used=ssd_slots_used,
        peak_duplicate_count=peak_dup,
        tier_used_at_end=tier_used_at_end,
    )


def run_sweep_case(
    preset: PresetSpec,
    workload: WorkloadConfig,
    sim_cfg: SimConfig,
    *,
    drop_oversized: bool = False,
) -> SweepRow:
    requests, _ = build_workload(workload)
    rejected_requests = 0
    if drop_oversized:
        requests, rejected = partition_by_hbm(requests, sim_cfg.hbm_size)
        rejected_requests = len(rejected)
        if not requests:
            return SweepRow(
                preset=preset.name,
                seed=workload.seed,
                num_requests=workload.num_requests,
                admittable_requests=0,
                rejected_requests=rejected_requests,
                status="fail",
                error=(
                    f"no admittable requests after drop-oversized "
                    f"(hbm={sim_cfg.hbm_size})"
                ),
            )
    admittable_requests = len(requests)

    pool = TaskPool()
    npu0, npu1, memories = build_engines(
        requests, pool, preset, sim_cfg, rng_seed=workload.seed
    )

    progress = (
        SimProgress(
            SimProgressConfig(
                total_requests=admittable_requests,
                stall_step_limit=2_000,
            )
        )
        if sim_cfg.show_progress
        else None
    )
    sim = Simulator(
        [npu0, npu1],
        pool,
        pd=PDConfig(
            spawn_map={"npu-0": "npu-1"},
            hold_kv_on_complete=preset.hold_kv_on_complete,
        ),
        log=SimLogger(SimLogConfig(enabled=False)),
        progress=progress,
    )

    wall_timeout = sim_cfg.wall_timeout_s
    if wall_timeout is None:
        if workload.trace_path is not None:
            span = max((r.arrival_time for r in requests), default=0.0)
            wall_timeout = max(60.0, span + admittable_requests * 3.0)
        else:
            wall_timeout = max(30.0, admittable_requests * 2.0)

    t0 = time.perf_counter()
    try:
        finish = sim.run(max_steps=sim_cfg.max_steps, wall_timeout_s=wall_timeout)
    except Exception as exc:
        return SweepRow(
            preset=preset.name,
            seed=workload.seed,
            num_requests=workload.num_requests,
            admittable_requests=admittable_requests,
            rejected_requests=rejected_requests,
            status="fail",
            error=str(exc),
            wall_seconds=time.perf_counter() - t0,
        )
    wall_seconds = time.perf_counter() - t0
    steps = sim.event_steps

    expected = {r.req_id for r in requests}
    decode_done = {
        r.req_id
        for r in npu1.completed
        if r.pd == RequestPD.DECODE and r.status == RequestStatus.COMPLETE
    }
    if decode_done != expected:
        return SweepRow(
            preset=preset.name,
            seed=workload.seed,
            num_requests=workload.num_requests,
            admittable_requests=admittable_requests,
            rejected_requests=rejected_requests,
            status="fail",
            error=f"decode incomplete missing={len(expected - decode_done)}",
            steps=steps,
            finish_time=finish,
            wall_seconds=wall_seconds,
        )

    for tier_key in HBM_TIER_KEYS:
        mem = memories.get(tier_key)
        if mem is None:
            continue
        ok, detail = _hbm_end_state_ok(preset, tier_key, mem)
        if not ok:
            return SweepRow(
                preset=preset.name,
                seed=workload.seed,
                num_requests=workload.num_requests,
                admittable_requests=admittable_requests,
                rejected_requests=rejected_requests,
                status="fail",
                error=f"{tier_key} HBM leaked {detail}",
                steps=steps,
                finish_time=finish,
                wall_seconds=wall_seconds,
            )

    step_budget = admittable_requests * sim_cfg.max_steps_per_request
    if steps > step_budget:
        return SweepRow(
            preset=preset.name,
            seed=workload.seed,
            num_requests=workload.num_requests,
            admittable_requests=admittable_requests,
            rejected_requests=rejected_requests,
            status="fail",
            error=f"steps {steps} > budget {step_budget}",
            steps=steps,
            finish_time=finish,
            wall_seconds=wall_seconds,
        )

    return _aggregate_metrics(
        preset.name,
        workload.seed,
        workload,
        npu0,
        npu1,
        steps=steps,
        finish_time=finish,
        wall_seconds=wall_seconds,
        memories=memories,
        admittable_requests=admittable_requests,
        rejected_requests=rejected_requests,
    )


@dataclass
class SweepConfig:
    presets: tuple[str, ...] = DEFAULT_PRESET_NAMES
    num_requests: int = 64
    seeds: int = 3
    base_seed: int = 1000
    sim: SimConfig = field(default_factory=SimConfig)
    csv_path: str | None = None
    raw_csv_path: str | None = None
    trace_path: str | None = None
    trace_offset: int = 0
    trace_time_scale: float = 0.001
    tokens_per_block: int = MOONCAKE_TOKENS_PER_BLOCK
    hbm: int = 40
    drop_oversized: bool = False


def _workload_config(cfg: SweepConfig, seed: int) -> WorkloadConfig:
    return WorkloadConfig(
        num_requests=cfg.num_requests,
        seed=seed,
        trace_path=cfg.trace_path,
        trace_offset=cfg.trace_offset,
        trace_time_scale=cfg.trace_time_scale,
        tokens_per_block=cfg.tokens_per_block,
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


def run_sweep(cfg: SweepConfig) -> list[SweepRow]:
    rows: list[SweepRow] = []
    unknown = [name for name in cfg.presets if name not in PRESETS]
    if unknown:
        raise ValueError(f"unknown presets: {unknown} (choose from {sorted(PRESETS)})")

    probe_workload = _workload_config(cfg, cfg.base_seed)
    hbm_size = cfg.hbm
    sim_cfg = replace(cfg.sim, hbm_size=hbm_size)

    if cfg.drop_oversized:
        probe_requests, _ = build_workload(probe_workload)
        _, rejected = partition_by_hbm(probe_requests, hbm_size)
        if rejected:
            print(
                f"drop-oversized: skipping {len(rejected)} requests with "
                f"footprint > hbm={hbm_size} "
                f"({len(probe_requests) - len(rejected)} admittable)",
                flush=True,
            )

    for preset_name in cfg.presets:
        preset = PRESETS[preset_name]
        for i in range(cfg.seeds):
            seed = cfg.base_seed + i
            workload = _workload_config(cfg, seed)
            row = run_sweep_case(
                preset,
                workload,
                sim_cfg,
                drop_oversized=cfg.drop_oversized,
            )
            rows.append(row)

    if cfg.raw_csv_path:
        _write_raw_csv(cfg.raw_csv_path, rows)
    if cfg.csv_path:
        _write_compare_csv(cfg.csv_path, rows, preset_order=cfg.presets)

    return rows


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
                + [_format_compare_cell(columns[preset].get(metric, "")) for preset in preset_order]
            )


def _print_table(rows: list[SweepRow]) -> None:
    ok = sum(1 for r in rows if r.status == "ok")
    print(
        f"sweep: {ok}/{len(rows)} passed  "
        f"presets={len({r.preset for r in rows})}  "
        f"requests={rows[0].num_requests if rows else 0}",
        flush=True,
    )
    header = (
        f"{'preset':<16} {'seed':>6} {'status':<6} "
        f"{'wall_s':>7} {'steps':>7} {'preempt':>7} "
        f"{'pull%':>6} {'p99_lat':>8} {'pulls':>6} {'dram':>5}"
    )
    print(header, flush=True)
    for row in rows:
        if row.status != "ok":
            print(
                f"{row.preset:<16} {row.seed:>6} {row.status:<6} "
                f"FAIL: {row.error}",
                flush=True,
            )
            continue
        print(
            f"{row.preset:<16} {row.seed:>6} {row.status:<6} "
            f"{row.wall_seconds:7.3f} {row.steps:7d} {row.preemptions:7d} "
            f"{100 * row.pull_ratio:5.1f}% "
            f"{row.decode_p99_latency:8.3f} "
            f"{row.decode_pulls:6d} {row.dram_slots_used:5d}",
            flush=True,
        )


def list_presets() -> None:
    print("Policy presets:", flush=True)
    for name in sorted(PRESETS):
        print(f"  {name:<16} {PRESETS[name].description}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare KV cache policies on a fixed PD workload.",
    )
    parser.add_argument(
        "--presets",
        default=",".join(DEFAULT_PRESET_NAMES),
        help=f"Comma-separated preset names (default: {','.join(DEFAULT_PRESET_NAMES)})",
    )
    parser.add_argument("--requests", type=int, default=64, help="Requests per run")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per preset")
    parser.add_argument("--base-seed", type=int, default=1000, help="First seed value")
    parser.add_argument("--hbm", type=int, default=40, help="HBM slots per engine")
    parser.add_argument(
        "--drop-oversized",
        action="store_true",
        help=(
            "Skip requests whose peak footprint exceeds --hbm "
            "(prefix + decode blocks); continue with the rest"
        ),
    )
    parser.add_argument("--dram", type=int, default=80, help="DRAM slots (dram_tier preset)")
    parser.add_argument(
        "--csv",
        metavar="PATH",
        help="Write wide compare CSV (metrics as rows, presets as columns)",
    )
    parser.add_argument(
        "--raw-csv",
        metavar="PATH",
        help="Write long CSV with one row per preset/seed run",
    )
    parser.add_argument("--list-presets", action="store_true", help="Show preset catalog")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show decode progress bar (stderr)",
    )
    parser.add_argument(
        "--trace",
        nargs="?",
        const=str(DEFAULT_TRACE_PATH),
        default=None,
        metavar="PATH",
        help=(
            "Replay Mooncake synthetic_trace.jsonl "
            f"(default: {DEFAULT_TRACE_PATH.name})"
        ),
    )
    parser.add_argument(
        "--trace-offset",
        type=int,
        default=0,
        help="Skip first N trace records before --requests slice",
    )
    parser.add_argument(
        "--trace-time-scale",
        type=float,
        default=0.001,
        help="Multiply trace timestamps (ms) to simulation seconds",
    )
    parser.add_argument(
        "--tokens-per-block",
        type=int,
        default=MOONCAKE_TOKENS_PER_BLOCK,
        help="Tokens per KV block when mapping output_length (Mooncake default: 512)",
    )
    args = parser.parse_args(argv)

    if args.list_presets:
        list_presets()
        return

    preset_names = tuple(p.strip() for p in args.presets.split(",") if p.strip())

    sim_cfg = SimConfig(
        dram_size=args.dram,
        show_progress=args.progress,
    )
    cfg = SweepConfig(
        presets=preset_names,
        num_requests=args.requests,
        seeds=args.seeds,
        base_seed=args.base_seed,
        sim=sim_cfg,
        csv_path=args.csv,
        raw_csv_path=args.raw_csv,
        trace_path=args.trace,
        trace_offset=args.trace_offset,
        trace_time_scale=args.trace_time_scale,
        tokens_per_block=args.tokens_per_block,
        hbm=args.hbm,
        drop_oversized=args.drop_oversized,
    )

    rows = run_sweep(cfg)
    _print_table(rows)

    failures = [r for r in rows if r.status != "ok"]
    if failures:
        raise SystemExit(f"sweep failed: {len(failures)}/{len(rows)} runs")

    if args.csv:
        print(f"compare csv written: {args.csv}", flush=True)
    if args.raw_csv:
        print(f"raw csv written: {args.raw_csv}", flush=True)
    print("sweep ok", flush=True)


if __name__ == "__main__":
    main()
