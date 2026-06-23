"""Sweep CLI: CSV output, tables, and argparse entrypoint."""

from __future__ import annotations

import argparse
import json
import sys

from simulator.policy.routing import ROUTING
from .mooncake_trace import DEFAULT_TRACE_PATH, MOONCAKE_TOKENS_PER_BLOCK
from .presets import DEFAULT_PRESET_NAMES, PRESETS
from .sweep_config import SimConfig, SweepConfig
from .metrics import (
    SWEEP_CSV_FIELDS,
    SweepRow,
    _aggregate_preset_rows,
    _compare_metric_names,
    _format_compare_cell,
    _row_values,
    _write_compare_csv,
    _write_raw_csv,
)
from .sweep_run import run_sweep


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
        f"{'pfx%':>6} {'dram%':>6} {'p99_lat':>8} {'pfx_pull':>8} {'dram':>5}"
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
            f"{100 * row.prefix_pull_ratio:5.1f}% "
            f"{100 * row.dram_hit_rate:5.1f}% "
            f"{row.decode_p99_latency:8.3f} "
            f"{row.decode_prefix_pulls:8d} {row.dram_slots_used:5d}",
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
    parser.add_argument(
        "--hbm-gib",
        type=float,
        default=32.0,
        dest="hbm_gib",
        help="HBM capacity per engine in GiB (converted to block slots)",
    )
    parser.add_argument(
        "--dram-gib",
        type=float,
        default=64.0,
        dest="dram_gib",
        help="DRAM tier capacity in GiB",
    )
    parser.add_argument(
        "--ssd-gib",
        type=float,
        default=256.0,
        dest="ssd_gib",
        help="SSD tier capacity in GiB",
    )
    parser.add_argument(
        "--kv-model",
        default="llama3-8b",
        choices=["toy", "llama3-8b", "llama3-70b"],
        help="Reference model for K+V bytes per token (unless --kv-bytes-per-token set)",
    )
    parser.add_argument(
        "--kv-bytes-per-token",
        type=float,
        default=None,
        help="Override K+V bytes per token for GiB→slot conversion",
    )
    parser.add_argument(
        "--drop-oversized",
        action="store_true",
        help=(
            "Skip requests whose peak footprint exceeds per-engine HBM slot count "
            "(derived from --hbm-gib); continue with the rest"
        ),
    )
    parser.add_argument(
        "--read-path",
        default=None,
        help="Override decode read-path policy (ordered_pull, min_cost, threshold, ...)",
    )
    parser.add_argument(
        "--pull-threshold",
        type=float,
        default=1.0,
        help="Threshold ratio for threshold read-path policy",
    )
    parser.add_argument(
        "--link-speed",
        type=float,
        default=None,
        help="Override per-tier read link base speed (GiB/s)",
    )
    parser.add_argument(
        "--interconnect-speed",
        type=float,
        default=None,
        help="Optional shared interconnect base speed (GiB/s)",
    )
    parser.add_argument(
        "--compute-speed",
        type=float,
        default=None,
        help="Override compute resource base speed",
    )
    parser.add_argument(
        "--experiment-spec",
        metavar="PATH",
        help="JSON file enumerating sweep axes (cartesian product)",
    )
    parser.add_argument(
        "--dram-chunk-blocks",
        type=int,
        default=4,
        help="HBM blocks per DRAM slot (LMCache chunk alignment)",
    )
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
        "--prefill",
        type=int,
        default=1,
        help="Number of prefill engines (xPyD)",
    )
    parser.add_argument(
        "--decode",
        type=int,
        default=1,
        help="Number of decode engines (xPyD)",
    )
    parser.add_argument(
        "--routing",
        default="bijection",
        choices=["bijection", "round_robin", "hash"],
        help="PD routing policy (proxy analogue)",
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
        hbm_gib=args.hbm_gib,
        dram_gib=args.dram_gib,
        ssd_gib=args.ssd_gib,
        kv_model=args.kv_model,
        kv_bytes_per_token=args.kv_bytes_per_token,
        tokens_per_block=args.tokens_per_block,
        dram_chunk_blocks=args.dram_chunk_blocks,
        show_progress=args.progress,
        link_speed=args.link_speed if args.link_speed is not None else 32.0,
        interconnect_speed=args.interconnect_speed,
        compute_speed=args.compute_speed if args.compute_speed is not None else 64.0,
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
        drop_oversized=args.drop_oversized,
        read_path=args.read_path,
        pull_threshold=args.pull_threshold,
        experiment_spec=args.experiment_spec,
        num_prefill=args.prefill,
        num_decode=args.decode,
        routing=args.routing,
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
