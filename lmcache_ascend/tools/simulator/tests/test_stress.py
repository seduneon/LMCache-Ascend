"""High-load integration stress test for the KV cache simulator."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

from engine import Engine
from memory import Memory
from pd import PDConfig
from policies import ComputeOnlyLookupPolicy, CostBasedPullLookupPolicy
from request import Request, RequestPD, RequestStatus
from resource import BandwidthResource, ComputeResource
from sim_log import SimLogConfig, SimLogger
from sim_progress import SimProgress, SimProgressConfig
from simulator import Simulator
from tasks import TaskPool


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


def generate_prefill_workload(cfg: StressConfig) -> tuple[list[Request], list[str]]:
    """Build many prefill requests with overlapping shared prefixes and unique tails."""
    rng = random.Random(cfg.seed)
    shared_pool = [f"shared:{i}" for i in range(cfg.shared_pool_size)]

    requests: list[Request] = []
    for i in range(cfg.num_requests):
        prefix_len = rng.randint(cfg.min_prefix_blocks, cfg.max_prefix_blocks)
        shared_count = rng.randint(prefix_len // 3, (2 * prefix_len) // 3)
        shared_count = max(1, min(shared_count, prefix_len - 1))

        blocks: list[str] = []
        start = rng.randint(0, cfg.shared_pool_size - 1)
        for j in range(shared_count):
            blocks.append(shared_pool[(start + j) % cfg.shared_pool_size])
        for j in range(prefix_len - shared_count):
            blocks.append(f"req{i:04d}:u{j}")

        arrival = i * cfg.arrival_spacing + rng.uniform(0.0, cfg.arrival_jitter)
        output_blocks = rng.randint(cfg.min_output_blocks, cfg.max_output_blocks)
        requests.append(
            Request(
                f"r{i:04d}",
                arrival,
                blocks,
                RequestPD.PREFILL,
                RequestStatus.PENDING,
                max_output_blocks=output_blocks,
            )
        )

    return requests, shared_pool


def _assert_idle(sim: Simulator, memories: dict[str, Memory]) -> None:
    for eng in sim.engines.values():
        sched = eng.scheduler
        assert not sched.pending, f"{eng.engine_id} has pending arrivals"
        assert not sched.waiting, f"{eng.engine_id} has waiting requests"
        assert not sched.running, f"{eng.engine_id} has running requests"

    for name, mem in memories.items():
        assert mem.used_size() == 0, f"{name} leaked KV blocks (used={mem.used_size()})"


@dataclass
class StressResult:
    num_requests: int
    steps: int
    finish_time: float
    preemptions: int
    wall_seconds: float


def run_stress_test(cfg: StressConfig | None = None) -> StressResult:
    cfg = cfg or StressConfig()
    requests, shared_pool = generate_prefill_workload(cfg)

    pool = TaskPool()
    memories = {
        "npu-0:hbm": Memory(size=cfg.hbm_size, name="npu-0:hbm"),
        "npu-1:hbm": Memory(size=cfg.hbm_size, name="npu-1:hbm"),
    }
    compute = ComputeResource(base_speed=64.0)
    pull_link = BandwidthResource(base_speed=32.0, latency=0.01)

    npu0 = Engine(
        engine_id="npu-0",
        requests=requests,
        pool=pool,
        memories=memories,
        local_memory="npu-0:hbm",
        policy=ComputeOnlyLookupPolicy(local_memory="npu-0:hbm"),
        compute_res=compute,
        work_per_block=1.0,
        max_num_seqs=cfg.max_num_seqs,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        enable_chunked_prefill=True,
    )
    npu1 = Engine(
        engine_id="npu-1",
        requests=[],
        pool=pool,
        memories=memories,
        local_memory="npu-1:hbm",
        policy=CostBasedPullLookupPolicy(
            local_memory="npu-1:hbm", pull_sources=["npu-0:hbm"]
        ),
        compute_res=compute,
        bandwidth_res=pull_link,
        work_per_block=1.0,
        work_per_transfer=1.0,
        max_num_seqs=cfg.max_num_seqs,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        enable_chunked_prefill=True,
        remote_kv_wait=True,
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

    t0 = time.perf_counter()
    finish = sim.run(max_steps=cfg.max_steps)
    wall_seconds = time.perf_counter() - t0
    steps = sim.event_steps
    if steps >= cfg.max_steps:
        raise AssertionError(
            f"stress test exceeded max_steps={cfg.max_steps} (possible livelock)"
        )

    _assert_idle(sim, memories)

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

    print(
        "stress_test ok "
        f"requests={cfg.num_requests} "
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
    )


def run_stress_benchmark(
    sizes: tuple[int, ...] = (4, 8, 16, 32, 64),
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


def run_stress_heavy_test() -> None:
    """Larger variant for explicit stress runs (`simulator.py stress heavy`)."""
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
