# KV Cache Simulator

Discrete-event simulator for KV cache scheduling, eviction, compute, and PD transfer. Block-grained workflow model aligned with vLLM’s batch scheduler (not a cycle-accurate GPU sim).

```bash
cd lmcache_ascend/tools/simulator
python3.12 simulator.py            # full test suite
python3.12 simulator.py deadlock   # preemption test only
python3.12 simulator.py limits     # max_num_seqs / token_budget test only
python3.12 simulator.py chunked    # chunked prefill test
python3.12 simulator.py pd         # PD read-mode + remote-KV tests
python3.12 simulator.py waiting    # waiting preempt + queue rotation
python3.12 simulator.py unit       # unit tests only
```

## Architecture

| File | Role |
|------|------|
| `scheduler.py` | Queues, `schedule()` → `Batch` (RUNNING → WAITING, unified allocate + preempt) |
| `engine.py` | `execute_batch()` (reserve, tasks), `apply_batch()` (advance state) |
| `simulator.py` | Global clock, multi-engine step, PD spawn |
| `pd.py` | `PDConfig` — validates and applies read-mode flags to engines |
| `policies.py` | `LookupPolicy` (pull/compute), `EvictionPolicy` |
| `tasks.py` | `ForwardTask` (batched compute), `LoadTask` (pull), `EvictTask` |
| `memory.py` | Content-keyed slot budget, holders, block states |
| `tests/run_tests.py` | Integration tests and CLI |
| `tests/test_unit.py` | Policy, task, memory, scheduler unit tests |

Each `Simulator.step()`:

1. `release_arrivals`
2. Per engine: `batch = schedule()` → `execute_batch(batch)`
3. Drain all batch tasks
4. `apply_batch(batch)`; spawn decode on prefill complete

Preemption frees KV and resets the request; it does not touch the task pool — the step drains before the next `schedule()`.

## Scheduling (vLLM-shaped)

`Scheduler.schedule()` per engine, once per step:

1. **RUNNING** (FCFS) — decode output or prefill continuation when `is_prefill_chunk()`, capped by `token_budget`; `_allocate_blocks()` with preempt loop.
2. **WAITING** (if no preempt this step) — same `_allocate_blocks()` path (not a separate lookup-only path). `WAITING_REMOTE_KV` at queue head is rotated to tail so later requests can proceed.
3. Phase from request cursor: `req.is_prefill_chunk()` ↔ vLLM `num_computed_tokens < prompt_len`; `apply_batch` advances `num_computed_blocks` by blocks completed this step.

`prefix_block_count` is set when a request leaves `PENDING` (or when spawned), not only at admit time.

On normal completion, `finish_request()` calls `free_request()` so HBM is not leaked.

Engine knobs: `max_num_seqs`, `max_num_batched_tokens`, `block_size`, `enable_chunked_prefill`.

## Policy hooks

### Eviction — `EvictionPolicy`

```python
class MyEviction(EvictionPolicy):
    def pick_victims(self, hbm, count, exclude) -> list[KVBlock]: ...

LookupPolicy(local_memory="npu-0:hbm", eviction_policy=MyEviction())
```

### Pull / compute — `LookupPolicy`

When `Engine` wires `transfer_links` and `compute_res`, resolution uses a **cost model**:

```text
share_time(work, sharers) = latency + work * sharers / base_speed
t_pull(src)  = link.share_time(work_per_transfer, link.works + 1)
t_recompute  = compute.share_time(work_per_block, compute.works + 1)
action       = argmin(t_pull, t_recompute)   # tie → pull
```

`+1` is passed by the policy (work not yet queued). Running tasks use `share_time(work_left, works)` in `Task.estimated_end()`.

Among pull sources, pick the minimum `t_pull`. Each source has its own `BandwidthResource` queue so load on one link does not affect another.

Without `transfer_links`, behavior is legacy: first `RESIDENT` source in `pull_sources` order, then `compute` (or `None` when `allow_compute=False` for remote-KV admit).

```python
fast = BandwidthResource(base_speed=100.0, latency=0.01)
slow = BandwidthResource(base_speed=1.0, latency=0.5)
policy = LookupPolicy(
    local_memory="npu-1:hbm",
    pull_sources=["npu-0:ssd", "npu-0:dram"],
)
Engine(
    ...,
    policy=policy,
    transfer_links={"npu-0:ssd": slow, "npu-0:dram": fast},
    compute_res=ComputeResource(base_speed=50.0),
    work_per_transfer=1.0,
    work_per_block=1.0,
)
# Or one shared link for all sources (backward compatible):
Engine(..., bandwidth_res=BandwidthResource(base_speed=10.0))
```

## Policy experiment readiness

Roughly **~80%** ready for evict/pull/compute policy sweeps.

### Ready now

| Experiment | Hook |
|------------|------|
| Pull vs recompute (load-aware) | `transfer_links`, `compute_res`, `work_per_transfer`, `work_per_block` |
| Pull source tie-break | `pull_sources` list order when costs equal |
| Compute vs transfer cost | `work_per_prefill_token`, `work_per_decode_req`, `work_per_transfer` |
| Memory pressure + preempt | `Memory(size)`, deadlock + waiting preempt tests |
| Batch limits | `max_num_seqs`, `max_num_batched_tokens` |
| PD read mode | `PDConfig(spawn_map=...)` auto-applies engine flags |
| Custom eviction | `EvictionPolicy.pick_victims` |

### Not implemented (distorts specific experiments)

| Gap | Affects |
|-----|---------|
| PD write mode (early spawn, D gates P) | Concurrent PD overlap |
| `num_computed_blocks` bumped in `apply_batch` not at schedule | Tight memory timing |
| Per-tier placement / duplicate caps | HBM vs DRAM vs SSD placement policy |
| Pull source refcount / consumption | Multi-consumer source memory |
| Workload generator / metrics CLI | Large sweeps |

## PD 1P1D

```python
from pd import PDConfig

sim = Simulator(
    [npu0, npu1],
    pool,
    pd=PDConfig(spawn_map={"npu-0": "npu-1"}),
)
# PDConfig.validate_and_apply sets:
#   npu-0.hold_kv_on_complete = True
#   npu-1.remote_kv_wait = True  (requires pull_sources on decode policy)
```

**PD read mode** (vLLM Mooncake pull-shaped):

1. P prefill completes → KV held on P (`finish_prefill_held`).
2. Decode spawned on D → `WAITING_REMOTE_KV`: pre-alloc D slots + pull batch.
3. Pull completes → promote to `RUNNING`, `free_request` on P.
4. Decode output steps; `free_request` on D when done.

## vLLM divergences (intentional simplifications)

| Area | vLLM | Simulator |
|------|------|-----------|
| Unit of work | Tokens | Blocks (`block_size` tokens/block) |
| Physical KV | Block table + refcounts | Content-keyed slots + holders |
| Preempt | `kv_cache_manager.free` | `free_request` + reset cursor |
| PD | KV connector, async match | Read mode: late spawn, `WAITING_REMOTE_KV`, deferred P release |
| Eviction | LRU on physical blocks | Pluggable; default `FirstAvailableEviction` |

## Roadmap

1. Run metrics (evictions, pulls, computes, preemptions)
2. Move cursor bump to schedule time
3. PD write mode (early spawn); workload config file
