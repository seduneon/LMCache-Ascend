"""Policy sweep harness: fixed workloads, aggregated metrics, CSV output."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field

from .engine import Engine
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
from .workload import WorkloadConfig, generate_prefill_workload


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


@dataclass
class SweepRow:
    preset: str
    seed: int
    num_requests: int
    status: str
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
    pull_ratio: float = 0.0
    dram_slots_used: int = 0
    ssd_slots_used: int = 0
    peak_duplicate_count: int = 0
    tier_used_at_end: str = ""


SWEEP_CSV_FIELDS = [f.name for f in SweepRow.__dataclass_fields__.values()]


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
) -> tuple[Engine, Engine, dict]:
    npu0, npu1, topo = build_pd_engines(
        requests,
        pool,
        preset,
        cfg=sim_cfg.engine_build(),
        resources=sim_cfg.resources(),
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
) -> SweepRow:
    requests, _ = generate_prefill_workload(workload)
    pool = TaskPool()
    npu0, npu1, memories = build_engines(requests, pool, preset, sim_cfg)

    progress = (
        SimProgress(
            SimProgressConfig(
                total_requests=workload.num_requests,
                stall_step_limit=2_000,
            )
        )
        if sim_cfg.show_progress
        else None
    )
    sim = Simulator(
        [npu0, npu1],
        pool,
        pd=PDConfig(spawn_map={"npu-0": "npu-1"}),
        log=SimLogger(SimLogConfig(enabled=False)),
        progress=progress,
    )

    wall_timeout = sim_cfg.wall_timeout_s
    if wall_timeout is None:
        wall_timeout = max(30.0, workload.num_requests * 2.0)

    t0 = time.perf_counter()
    try:
        finish = sim.run(max_steps=sim_cfg.max_steps, wall_timeout_s=wall_timeout)
    except Exception as exc:
        return SweepRow(
            preset=preset.name,
            seed=workload.seed,
            num_requests=workload.num_requests,
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
            status="fail",
            error=f"decode incomplete missing={len(expected - decode_done)}",
            steps=steps,
            finish_time=finish,
            wall_seconds=wall_seconds,
        )

    for tier_key in HBM_TIER_KEYS:
        mem = memories.get(tier_key)
        if mem is not None and mem.used_size() != 0:
            return SweepRow(
                preset=preset.name,
                seed=workload.seed,
                num_requests=workload.num_requests,
                status="fail",
                error=f"{tier_key} HBM leaked used={mem.used_size()}",
                steps=steps,
                finish_time=finish,
                wall_seconds=wall_seconds,
            )

    step_budget = workload.num_requests * sim_cfg.max_steps_per_request
    if steps > step_budget:
        return SweepRow(
            preset=preset.name,
            seed=workload.seed,
            num_requests=workload.num_requests,
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
    )


@dataclass
class SweepConfig:
    presets: tuple[str, ...] = DEFAULT_PRESET_NAMES
    num_requests: int = 64
    seeds: int = 3
    base_seed: int = 1000
    sim: SimConfig = field(default_factory=SimConfig)
    csv_path: str | None = None


def run_sweep(cfg: SweepConfig) -> list[SweepRow]:
    rows: list[SweepRow] = []
    unknown = [name for name in cfg.presets if name not in PRESETS]
    if unknown:
        raise ValueError(f"unknown presets: {unknown} (choose from {sorted(PRESETS)})")

    for preset_name in cfg.presets:
        preset = PRESETS[preset_name]
        for i in range(cfg.seeds):
            seed = cfg.base_seed + i
            workload = WorkloadConfig(num_requests=cfg.num_requests, seed=seed)
            row = run_sweep_case(preset, workload, cfg.sim)
            rows.append(row)

    if cfg.csv_path:
        _write_csv(cfg.csv_path, rows)

    return rows


def _write_csv(path: str, rows: list[SweepRow]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in SWEEP_CSV_FIELDS})


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
    parser.add_argument("--dram", type=int, default=80, help="DRAM slots (dram_tier preset)")
    parser.add_argument("--csv", metavar="PATH", help="Write results to CSV")
    parser.add_argument("--list-presets", action="store_true", help="Show preset catalog")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show decode progress bar (stderr)",
    )
    args = parser.parse_args(argv)

    if args.list_presets:
        list_presets()
        return

    preset_names = tuple(p.strip() for p in args.presets.split(",") if p.strip())
    sim_cfg = SimConfig(
        hbm_size=args.hbm,
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
    )

    rows = run_sweep(cfg)
    _print_table(rows)

    failures = [r for r in rows if r.status != "ok"]
    if failures:
        raise SystemExit(f"sweep failed: {len(failures)}/{len(rows)} runs")

    if args.csv:
        print(f"csv written: {args.csv}", flush=True)
    print("sweep ok", flush=True)


if __name__ == "__main__":
    main()
