"""High-load integration stress test for the KV cache simulator."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from simulator.sim_log import SimLogConfig, SimLogger
from simulator.sim_progress import SimProgress, SimProgressConfig
from simulator.simulator import Simulator
from simulator.sweep import PRESETS, SimConfig, build_engines
from simulator.tasks import TaskPool
from simulator.workload import WorkloadConfig, generate_prefill_workload

from simulator.engine import Engine
from simulator.memory import Memory
from simulator.pd import PDConfig
from simulator.request import Request, RequestPD, RequestStatus


# Default request-count sweep for benchmark / multi-seed stress runs.
STRESS_SIZES: tuple[int, ...] = (32, 64, 128, 256, 512)


@dataclass(frozen=True)
class StressConfig:
    num_requests: int = 16
    shared_pool_size: int = 48
    min_prefix_blocks: int = 6
    max_prefix_blocks: int = 14
    min_output_blocks: int = 3
    max_output_blocks: int = 8
    arrival_spacing: float = 0.02
    arrival_jitter: float = 0.01
    hbm_size: int = 40
    max_num_seqs: int = 12
    max_num_batched_tokens: int = 24
    seed: int = 42
    max_steps: int = 5_000_000
    log: SimLogConfig | None = None
    log_interval: int = 200
    show_progress: bool = True
    stall_step_limit: int = 2_000
    wall_timeout_s: float | None = None
    max_steps_per_request: int = 500
    quiet: bool = False

    def to_workload(self) -> WorkloadConfig:
        return WorkloadConfig(
            num_requests=self.num_requests,
            shared_pool_size=self.shared_pool_size,
            min_prefix_blocks=self.min_prefix_blocks,
            max_prefix_blocks=self.max_prefix_blocks,
            min_output_blocks=self.min_output_blocks,
            max_output_blocks=self.max_output_blocks,
            arrival_spacing=self.arrival_spacing,
            arrival_jitter=self.arrival_jitter,
            seed=self.seed,
        )

    def to_sim(self) -> SimConfig:
        return SimConfig(
            hbm_size=self.hbm_size,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_steps=self.max_steps,
            max_steps_per_request=self.max_steps_per_request,
            wall_timeout_s=self.wall_timeout_s,
            show_progress=self.show_progress,
        )


def _assert_idle(memories: dict[str, Memory]) -> None:
    for name, mem in memories.items():
        assert mem.used_size() == 0, f"{name} leaked KV blocks (used={mem.used_size()})"


@dataclass
class StressResult:
    num_requests: int
    steps: int
    finish_time: float
    preemptions: int
    wall_seconds: float
    seed: int = 0


def _assert_request_metrics(
    requests: list[Request],
    npu0: Engine,
    npu1: Engine,
) -> None:
    for req in requests:
        prefill = next(r for r in npu0.completed if r.req_id == req.req_id)
        decode = next(r for r in npu1.completed if r.req_id == req.req_id)
        pm, dm = prefill.metrics, decode.metrics

        assert pm.finished_at is not None, f"{req.req_id} prefill missing finished_at"
        assert dm.finished_at is not None, f"{req.req_id} decode missing finished_at"
        assert dm.latency is not None and dm.latency >= 0, f"{req.req_id} bad latency"
        assert pm.computes + pm.local_hits >= prefill.prefix_block_count, (
            f"{req.req_id} prefill computes={pm.computes} local={pm.local_hits} "
            f"prefix={prefill.prefix_block_count}"
        )
        assert dm.remote_kv_admits >= 1 or dm.local_hits >= 1 or dm.pulls >= 1, (
            f"{req.req_id} decode has no prefix resolution path "
            f"(pulls={dm.pulls} local_hits={dm.local_hits} "
            f"remote_kv={dm.remote_kv_admits})"
        )
        assert dm.computes >= req.max_output_blocks, (
            f"{req.req_id} decode computes={dm.computes} "
            f"output={req.max_output_blocks}"
        )
        assert dm.engine_id == "npu-1", f"{req.req_id} decode engine_id={dm.engine_id}"


def run_stress_test(cfg: StressConfig | None = None) -> StressResult:
    cfg = cfg or StressConfig()
    workload = cfg.to_workload()
    requests, shared_pool = generate_prefill_workload(workload)

    pool = TaskPool()
    npu0, npu1, memories = build_engines(
        requests, pool, PRESETS["baseline"], cfg.to_sim()
    )

    sim_log = SimLogger(
        cfg.log
        if cfg.log is not None
        else SimLogConfig(
            enabled=os.environ.get("SIM_LOG", "").lower() in ("1", "true", "yes"),
            detail=os.environ.get("SIM_LOG_DETAIL", "").lower() in ("1", "true", "yes"),
            step_interval=cfg.log_interval,
        )
    )
    progress = (
        SimProgress(
            SimProgressConfig(
                total_requests=cfg.num_requests,
                stall_step_limit=cfg.stall_step_limit,
            )
        )
        if cfg.show_progress
        else None
    )
    sim = Simulator(
        [npu0, npu1],
        pool,
        pd=PDConfig(spawn_map={"npu-0": "npu-1"}),
        log=sim_log,
        progress=progress,
    )

    sim_log.milestone(sim, f"stress begin requests={cfg.num_requests}")

    wall_timeout = cfg.wall_timeout_s
    if wall_timeout is None:
        wall_timeout = max(30.0, cfg.num_requests * 2.0)

    t0 = time.perf_counter()
    finish = sim.run(max_steps=cfg.max_steps, wall_timeout_s=wall_timeout)
    wall_seconds = time.perf_counter() - t0
    steps = sim.event_steps
    if steps >= cfg.max_steps:
        raise AssertionError(
            f"stress test exceeded max_steps={cfg.max_steps} (possible livelock)"
        )
    step_budget = cfg.num_requests * cfg.max_steps_per_request
    if steps > step_budget:
        raise AssertionError(
            f"stress test took {steps} steps for {cfg.num_requests} requests "
            f"(budget={step_budget}; possible livelock)"
        )

    for eng in sim.engines.values():
        sched = eng.scheduler
        assert not sched.pending, f"{eng.engine_id} has pending arrivals"
        assert not sched.waiting, f"{eng.engine_id} has waiting requests"
        assert not sched.running, f"{eng.engine_id} has running requests"
    _assert_idle(memories)

    prefill_done = {
        r.req_id for r in npu0.completed if r.pd == RequestPD.PREFILL and not r.kv_held_for_transfer
    }
    decode_done = {
        r.req_id
        for r in npu1.completed
        if r.pd == RequestPD.DECODE and r.status == RequestStatus.COMPLETE
    }
    expected = {r.req_id for r in requests}

    assert prefill_done == expected, (
        f"prefill incomplete: missing={len(expected - prefill_done)} "
        f"extra={len(prefill_done - expected)}"
    )
    assert decode_done == expected, (
        f"decode incomplete: missing={len(expected - decode_done)} "
        f"extra={len(decode_done - expected)}"
    )

    preemptions = sum(r.num_preemptions for r in npu0.completed + npu1.completed)
    held_kv = sum(1 for r in npu0.completed if r.kv_held_for_transfer)
    if held_kv != 0:
        raise AssertionError(f"{held_kv} prefill requests still hold KV for transfer")
    if cfg.num_requests >= 8:
        assert preemptions > 0, "expected memory pressure to cause at least one preemption"

    _assert_request_metrics(requests, npu0, npu1)

    if not cfg.quiet:
        print(
            "stress_test ok "
            f"requests={cfg.num_requests} "
            f"seed={cfg.seed} "
            f"shared_pool={len(shared_pool)} "
            f"steps={steps} "
            f"wall_s={wall_seconds:.3f} "
            f"finish_time={finish:.2f} "
            f"preemptions={preemptions} "
            f"hbm={cfg.hbm_size} "
            f"max_seqs={cfg.max_num_seqs} "
            f"token_budget={cfg.max_num_batched_tokens}"
        )
    return StressResult(
        num_requests=cfg.num_requests,
        steps=steps,
        finish_time=finish,
        preemptions=preemptions,
        wall_seconds=wall_seconds,
        seed=cfg.seed,
    )


def run_stress_benchmark(
    sizes: tuple[int, ...] = STRESS_SIZES,
) -> None:
    """Sweep request counts and print wall time / step scaling."""
    print("stress benchmark (PD read, chunked prefill, tight HBM)", flush=True)
    print(f"{'requests':>8}  {'wall_s':>8}  {'steps':>8}  {'sim_t':>8}  {'preempt':>7}  {'s/req':>8}", flush=True)
    prev_wall = 0.0
    for n in sizes:
        print(f"  running n={n}...", flush=True)
        result = run_stress_test(
            StressConfig(
                num_requests=n,
                log=SimLogConfig(enabled=False),
                show_progress=True,
            )
        )
        per_req = result.wall_seconds / n if n else 0.0
        ratio = result.wall_seconds / prev_wall if prev_wall > 0 else 0.0
        extra = f"  x{ratio:.1f}" if prev_wall > 0 else ""
        print(
            f"{result.num_requests:8d}  "
            f"{result.wall_seconds:8.3f}  "
            f"{result.steps:8d}  "
            f"{result.finish_time:8.2f}  "
            f"{result.preemptions:7d}  "
            f"{per_req:8.4f}{extra}"
        )
        prev_wall = result.wall_seconds
    print("stress_benchmark ok")


def run_stress_seed_sweep(
    sizes: tuple[int, ...] = STRESS_SIZES,
    *,
    seeds_per_size: int = 10,
    base_seed: int = 1000,
) -> None:
    """Run many random workloads per request count to catch seed-specific bugs."""
    print(
        "stress seed sweep (PD read, chunked prefill, tight HBM)",
        flush=True,
    )
    print(
        f"sizes={list(sizes)} seeds_per_size={seeds_per_size} "
        f"base_seed={base_seed}",
        flush=True,
    )
    print(
        f"{'requests':>8}  {'seeds':>5}  {'pass':>5}  "
        f"{'max_wall':>8}  {'max_steps':>9}  {'max_preempt':>11}  "
        f"{'max_sim_t':>9}",
        flush=True,
    )

    total_runs = 0
    total_pass = 0
    t0 = time.perf_counter()

    for n in sizes:
        passed = 0
        max_wall = 0.0
        max_steps = 0
        max_preempt = 0
        max_sim_t = 0.0

        for i in range(seeds_per_size):
            seed = base_seed + i
            total_runs += 1
            try:
                result = run_stress_test(
                    StressConfig(
                        num_requests=n,
                        seed=seed,
                        log=SimLogConfig(enabled=False),
                        show_progress=False,
                        quiet=True,
                    )
                )
            except Exception as exc:
                print(
                    f"  FAIL n={n} seed={seed}: {exc}",
                    flush=True,
                )
                continue

            passed += 1
            total_pass += 1
            max_wall = max(max_wall, result.wall_seconds)
            max_steps = max(max_steps, result.steps)
            max_preempt = max(max_preempt, result.preemptions)
            max_sim_t = max(max_sim_t, result.finish_time)

        print(
            f"{n:8d}  {seeds_per_size:5d}  {passed:5d}  "
            f"{max_wall:8.3f}  {max_steps:9d}  {max_preempt:11d}  "
            f"{max_sim_t:9.2f}",
            flush=True,
        )
        if passed != seeds_per_size:
            raise AssertionError(
                f"stress seed sweep: n={n} passed {passed}/{seeds_per_size}"
            )

    wall = time.perf_counter() - t0
    print(
        f"stress_seed_sweep ok runs={total_pass}/{total_runs} wall_s={wall:.2f}",
        flush=True,
    )


def run_stress_heavy_test() -> None:
    """Larger variant for explicit stress runs (`simulator.py stress-heavy`)."""
    run_stress_test(
        StressConfig(
            num_requests=512,
            hbm_size=32,
            max_num_seqs=16,
            max_num_batched_tokens=32,
            arrival_spacing=0.01,
            seed=7,
        )
    )
