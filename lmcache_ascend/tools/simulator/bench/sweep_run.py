"""Sweep execution: build engines and run individual cases."""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import replace

from simulator.runtime.engine import Engine
from simulator.core.resource import BandwidthResource
from simulator.core.request import RequestPD, RequestStatus
from simulator.observability.event_trace import EventTraceWriter, trace_config_from_env
from simulator.observability.sim_log import SimLogConfig, SimLogger
from simulator.observability.sim_progress import SimProgress, SimProgressConfig
from simulator.runtime.simulator import Simulator
from simulator.runtime.pd import PDConfig
from simulator.runtime.tasks import TaskPool
from simulator.model.layout import engine_ids_for_pd
from simulator.model.capacity import format_tier_capacity
from .mooncake_trace import MOONCAKE_TOKENS_PER_BLOCK, partition_by_hbm
from .presets import PRESETS, PresetSpec, build_engines as build_sim_engines
from .workload import WorkloadConfig, build_workload
from .sweep_config import SimConfig, SweepConfig
from .metrics import (
    SweepRow,
    _aggregate_metrics,
    _local_tier_end_state_ok,
    _write_compare_csv,
    _write_raw_csv,
)

def build_engines(
    requests: list,
    pool: TaskPool,
    preset: PresetSpec,
    sim_cfg: SimConfig,
    *,
    prefill_ids: tuple[str, ...],
    decode_ids: tuple[str, ...],
    routing_name: str = "bijection",
    routing_params: dict | None = None,
    rng_seed: int = 0,
    tokens_per_block: int = MOONCAKE_TOKENS_PER_BLOCK,
):
    return build_sim_engines(
        requests,
        pool,
        preset,
        prefill_ids=prefill_ids,
        decode_ids=decode_ids,
        routing_name=routing_name,
        routing_params=routing_params,
        cfg=sim_cfg.engine_build(tokens_per_block=tokens_per_block),
        resources=sim_cfg.resources(
            tokens_per_block=tokens_per_block,
            prefill_ids=prefill_ids,
            decode_ids=decode_ids,
        ),
        rng_seed=rng_seed,
    )

def _effective_preset(preset: PresetSpec, sweep_cfg: SweepConfig) -> PresetSpec:
    from dataclasses import replace

    if not sweep_cfg.read_path:
        return preset
    return replace(
        preset,
        decode_read_path=sweep_cfg.read_path,
        decode_read_path_params={"threshold_ratio": sweep_cfg.pull_threshold},
    )


def run_sweep_case(
    preset: PresetSpec,
    workload: WorkloadConfig,
    sim_cfg: SimConfig,
    *,
    drop_oversized: bool = False,
    sweep_cfg: SweepConfig | None = None,
) -> SweepRow:
    preset = _effective_preset(preset, sweep_cfg) if sweep_cfg else preset
    read_path_label = ""
    if preset.decode_read_path is not None:
        read_path_label = preset.decode_read_path
    requests, _ = build_workload(workload)
    prefill_ids, decode_ids = engine_ids_for_pd(
        num_prefill=sweep_cfg.num_prefill if sweep_cfg else 1,
        num_decode=sweep_cfg.num_decode if sweep_cfg else 1,
    )
    resources = sim_cfg.resources(
        tokens_per_block=workload.tokens_per_block,
        prefill_ids=prefill_ids,
        decode_ids=decode_ids,
    )
    hbm_slots = resources.hbm_size(prefill_ids[0])
    rejected_requests = 0
    if drop_oversized:
        requests, rejected = partition_by_hbm(requests, hbm_slots)
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
                    f"(hbm={hbm_slots} slots, {sim_cfg.hbm_gib:g} GiB)"
                ),
            )
    admittable_requests = len(requests)

    pool = TaskPool()
    routing_name = sweep_cfg.routing if sweep_cfg else "bijection"
    engines, topo, routing = build_engines(
        requests,
        pool,
        preset,
        sim_cfg,
        prefill_ids=prefill_ids,
        decode_ids=decode_ids,
        routing_name=routing_name,
        rng_seed=workload.seed,
        tokens_per_block=workload.tokens_per_block,
    )
    memories = topo.memories
    tier_roles = topo.graph.tier_roles()
    if sim_cfg.interconnect_speed is not None:
        interconnect = BandwidthResource(
            base_speed=sim_cfg.interconnect_speed,
            latency=sim_cfg.link_latency,
        )
        for eng in engines.values():
            eng.interconnect = interconnect
            eng.cache.interconnect = interconnect

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
    trace_writer = None
    trace_cfg = trace_config_from_env()
    if trace_cfg.enabled:
        trace_writer = EventTraceWriter(trace_cfg)
    sim = Simulator(
        list(engines.values()),
        pool,
        pd=PDConfig(
            routing=routing,
            engine_role=topo.engine_role,
            hold_kv_on_complete=preset.hold_kv_on_complete,
        ),
        log=SimLogger(SimLogConfig(enabled=False)),
        progress=progress,
        event_trace=trace_writer,
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
        for eid in topo.decode_ids
        for r in engines[eid].completed
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

    for eid in topo.prefill_ids + topo.decode_ids:
        eng = engines[eid]
        is_prefill = topo.engine_role[eid] == "prefill"
        tier_key = eng.local_memory
        mem = memories.get(tier_key)
        if mem is None:
            continue
        ok, detail = _local_tier_end_state_ok(preset, mem, is_prefill=is_prefill)
        if not ok:
            return SweepRow(
                preset=preset.name,
                seed=workload.seed,
                num_requests=workload.num_requests,
                admittable_requests=admittable_requests,
                rejected_requests=rejected_requests,
                status="fail",
                error=f"{tier_key} local tier leaked {detail}",
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
        engines,
        topo,
        steps=steps,
        finish_time=finish,
        wall_seconds=wall_seconds,
        memories=memories,
        tier_roles=tier_roles,
        admittable_requests=admittable_requests,
        rejected_requests=rejected_requests,
        read_path=read_path_label,
    )
def _workload_config(cfg: SweepConfig, seed: int) -> WorkloadConfig:
    return WorkloadConfig(
        num_requests=cfg.num_requests,
        seed=seed,
        trace_path=cfg.trace_path,
        trace_offset=cfg.trace_offset,
        trace_time_scale=cfg.trace_time_scale,
        tokens_per_block=cfg.tokens_per_block,
    )

def _print_tier_capacity(sim_cfg: SimConfig, *, tokens_per_block: int) -> None:
    resources = sim_cfg.resources(tokens_per_block=tokens_per_block)
    kv_bpt = sim_cfg.resolved_kv_bytes_per_token()
    parts = [
        format_tier_capacity(
            tier,
            blocks=resources.size(tier.tier_key),
            kv_bytes_per_token=kv_bpt,
            tokens_per_block=tokens_per_block,
        )
        for tier in sim_cfg.resolved_tiers()
    ]
    print(f"tier capacity: {'; '.join(parts)}", flush=True)


def _expand_sweep_configs(cfg: SweepConfig) -> list[SweepConfig]:
    """Expand a JSON experiment spec into a cartesian product of sweep axes."""
    if not cfg.experiment_spec:
        return [cfg]
    from dataclasses import replace

    with open(cfg.experiment_spec, encoding="utf-8") as handle:
        spec = json.load(handle)
    presets = tuple(spec.get("presets", cfg.presets))
    read_paths = spec.get("read_paths", [cfg.read_path])
    pull_thresholds = spec.get("pull_thresholds", [cfg.pull_threshold])
    link_speeds = spec.get("link_speeds", [cfg.sim.link_speed])
    interconnect_speeds = spec.get(
        "interconnect_speeds", [cfg.sim.interconnect_speed]
    )
    compute_speeds = spec.get("compute_speeds", [cfg.sim.compute_speed])
    seeds = spec.get("seeds", cfg.seeds)
    num_requests = spec.get("num_requests", cfg.num_requests)

    expanded: list[SweepConfig] = []
    for preset, read_path, pull_thr, link, ic, compute in itertools.product(
        presets,
        read_paths,
        pull_thresholds,
        link_speeds,
        interconnect_speeds,
        compute_speeds,
    ):
        sim = replace(
            cfg.sim,
            link_speed=float(link),
            interconnect_speed=ic,
            compute_speed=float(compute),
        )
        expanded.append(
            replace(
                cfg,
                presets=(preset,),
                num_requests=num_requests,
                seeds=seeds,
                sim=sim,
                read_path=read_path,
                pull_threshold=float(pull_thr),
                experiment_spec=None,
            )
        )
    return expanded


def run_sweep(cfg: SweepConfig) -> list[SweepRow]:
    rows: list[SweepRow] = []
    configs = _expand_sweep_configs(cfg)
    preset_names = {name for c in configs for name in c.presets}
    unknown = [name for name in preset_names if name not in PRESETS]
    if unknown:
        raise ValueError(f"unknown presets: {unknown} (choose from {sorted(PRESETS)})")

    sim_cfg = cfg.sim
    resources = sim_cfg.resources(tokens_per_block=cfg.tokens_per_block)
    _print_tier_capacity(sim_cfg, tokens_per_block=cfg.tokens_per_block)

    if cfg.drop_oversized:
        probe_workload = _workload_config(cfg, cfg.base_seed)
        probe_requests, _ = build_workload(probe_workload)
        _, rejected = partition_by_hbm(probe_requests, resources.hbm_size("npu-0"))
        if rejected:
            print(
                f"drop-oversized: skipping {len(rejected)} requests with "
                f"footprint > hbm={resources.hbm_size('npu-0')} slots "
                f"({sim_cfg.hbm_gib:g} GiB/engine) "
                f"({len(probe_requests) - len(rejected)} admittable)",
                flush=True,
            )

    for sweep_cfg in configs:
        case_sim = sweep_cfg.sim
        for preset_name in sweep_cfg.presets:
            preset = PRESETS[preset_name]
            for i in range(sweep_cfg.seeds):
                seed = sweep_cfg.base_seed + i
                workload = _workload_config(sweep_cfg, seed)
                row = run_sweep_case(
                    preset,
                    workload,
                    case_sim,
                    drop_oversized=sweep_cfg.drop_oversized,
                    sweep_cfg=sweep_cfg,
                )
                rows.append(row)

    if cfg.raw_csv_path:
        _write_raw_csv(cfg.raw_csv_path, rows)
    if cfg.csv_path:
        _write_compare_csv(cfg.csv_path, rows, preset_order=cfg.presets)

    return rows

