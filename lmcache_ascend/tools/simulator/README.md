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
python3.12 simulator.py critical   # micro-step, PD KV, memory, task DAG tests
python3.12 simulator.py stress     # PD stress test (progress logs on stderr)
python3.12 simulator.py stress-seeds  # 10 random seeds × each size (32–512)
python3.12 simulator.py stress-benchmark  # one run per size, with progress
SIM_LOG_DETAIL=1 python3.12 simulator.py stress   # verbose per-batch logs
python3.12 simulator.py stress-heavy              # larger stress run
```

## Architecture

| File | Role |
|------|------|
| `scheduler.py` | Queues, `schedule()` → `Batch` (RUNNING → WAITING, unified allocate + preempt) |
| `engine.py` | `execute_batch()` (reserve, tasks), `apply_batch()` (advance state) |
| `simulator.py` | Global clock, micro-step event loop, in-flight batches, PD spawn |
| `sim_log.py` | Optional progress logging (`SimLogger`) |
| `pd.py` | `PDConfig` — validates and applies read-mode flags to engines |
| `policies.py` | `LookupPolicy` ABC + pull/compute implementations, `EvictionPolicy` |
| `tasks.py` | `ForwardTask` (batched compute), `LoadTask` (pull), `EvictTask` |
| `memory.py` | Content-keyed slot budget, holders, block states |
| `tests/run_tests.py` | Integration tests and CLI |
| `tests/test_unit.py` | Policy, task, memory, scheduler unit tests |

Each `Simulator.step()` advances `now` by **one discrete event**:

1. `release_arrivals` at current `now`
2. Per engine (if no in-flight batch): `schedule()` → `execute_batch()` → tasks registered in flight
3. Start ready tasks; `finish_done()` for zero-work tasks; advance `now` to the next task completion **or** next arrival
4. When all tasks for a batch complete: `apply_batch()` at that event time; PD spawn uses the same `now`

There is no hidden multi-event drain inside a step — `now` always means the current event time.

Preemption frees KV and resets the request; the next schedule happens after the in-flight batch finishes.

## Scheduling (vLLM-shaped)

`Scheduler.schedule()` per engine, once per step:

1. **RUNNING** (FCFS) — decode output or prefill continuation when `is_prefill_chunk()`, capped by `token_budget`; `_allocate_blocks()` with preempt loop.
2. **WAITING** (if no preempt this step) — same `_allocate_blocks()` path (not a separate lookup-only path). `WAITING_REMOTE_KV` at queue head is rotated to tail so later requests can proceed.
3. Phase from request cursor: `req.is_prefill_chunk()` ↔ vLLM `num_computed_tokens < prompt_len`; `apply_batch` advances `num_computed_blocks` by blocks completed this step.

`prefix_block_count` is set when a request leaves `PENDING` (or when spawned), not only at admit time.

On normal completion, `finish_request()` calls `free_request()` so HBM is not leaked.

Engine knobs: `max_num_seqs`, `max_num_batched_tokens`, `block_size`, `enable_chunked_prefill`.

## Policy hooks (today)

### Eviction — `EvictionPolicy`

```python
class MyEviction(EvictionPolicy):
    def pick_victims(self, hbm, count, exclude) -> list[KVBlock]: ...

LookupPolicy(local_memory="npu-0:hbm", eviction_policy=MyEviction())
```

### Pull / compute — `LookupPolicy`

| Class | Behavior |
|-------|----------|
| `ComputeOnlyLookupPolicy` | Local hit or recompute |
| `OrderedPullLookupPolicy` | First `pull_sources` with a resident copy, else recompute |
| `CostBasedPullLookupPolicy` | Min-cost pull source vs recompute using per-link bandwidth queues |

`CostBasedPullLookupPolicy` cost model (at **schedule** time):

```text
time_for(work, load) = latency + work / speed(load)
load(link)   = link.queued_load() + pending_pulls_in_this_allocation + 1
load(compute)= compute.queued_load() + (1 if forward not yet reserved else 0) + 1

recompute_work = work_per_prefill_token * block_size   (prefill chunk)
              | work_per_decode_req                    (decode step)
              | work_per_block                         (fallback)

t_pull      = link.time_for(work_per_transfer, load(link))
t_recompute = compute.time_for(recompute_work, load(compute))  # 0 if forward already reserved
action      = argmin(t_pull, t_recompute)   # tie → pull
```

`Resource.queued_load()` = `running` + `scheduled`. Tasks call `schedule()` via `TaskPool.add` / `Task.reserve_resource()`, then `start()` / `finish()` on the resource at task lifecycle transitions.

```python
fast = BandwidthResource(base_speed=100.0, latency=0.01)
slow = BandwidthResource(base_speed=1.0, latency=0.5)
policy = CostBasedPullLookupPolicy(
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
```

## Experiment readiness (summary)

| Policy area | Readiness | Notes |
|-------------|-----------|-------|
| Pull vs recompute (read path) | **~85%** | Token-aware recompute + queued load on links/compute |
| Placement / eviction / duplicates | **~30–40%** | HBM eviction + custom victims only; no write path or tier retention |
| vLLM scheduler shape | **~85%** | Batching, preempt, chunked prefill, PD read mode |
| Sweep infrastructure | **~40%** | Per-request metrics exist; no trace replay or aggregation CLI |

Use the gap sections below when designing experiments — running a sweep that assumes a missing feature will silently give wrong conclusions.

---

## Gaps: placement, eviction, duplicates

These are the main blockers for LMCache-style **where to put KV** and **how many copies to keep** experiments.

### Write-side placement (not implemented)

| Gap | Today | Needed for vLLM/LMCache alignment |
|-----|-------|-----------------------------------|
| Where computed KV goes | All new blocks land in `local_memory` (HBM) only | Policy chooses HBM / DRAM / SSD on compute complete |
| Remote tiers | `pull_sources` are **read catalogs** — never written | Spill, promote, demote between tiers |
| Per-tier capacity | Only local HBM has a slot budget | Independent `Memory(size=…)` pressure + eviction per tier |
| Spill on evict | Victim is removed (`EvictTask`) | Option to spill HBM → slower tier instead of drop |

**Suggested hook:** `PlacementPolicy.on_block_resident(block_hash, req) → list[tier_keys]` and `on_evict_from(tier, block) → drop | spill_to`.

### Duplicate retention (partial structure only)

`Memory.blocks[hash]` is a **list** of physical copies, but nothing policy-driven uses that yet.

| Gap | Today | Needed |
|-----|-------|--------|
| Max copies per hash | Unbounded list append on reserve | Per-tier and global caps (0/1/N) |
| Cross-tier duplicates | Not modeled | e.g. keep HBM + DRAM + SSD simultaneously |
| Pull consumption | Source copy is never removed | Policy: retain vs consume vs clone |
| Deduplication | Holders refcount shared blocks | Explicit “canonical copy” vs per-request copies |

**Suggested hook:** `RetentionPolicy.max_copies(tier, block_hash)` and `should_retain_after_pull(src, dst)`.

### Eviction fidelity

| Gap | vLLM | Simulator |
|-----|------|-----------|
| Victim selection | LRU on physical blocks | Default `FirstAvailableEviction` (arbitrary order) |
| Touch on access | Updates LRU on hit / use | No last-access tracking |
| Prefix-aware scoring | Prefer evicting unshared / low-reuse | Not modeled |
| Watermarks | Reserved blocks for decode vs prefill | Not modeled |
| Eviction timing | Often synchronous at allocation | Async `EvictTask` on compute resource (distorts pressure timing) |
| Eviction scope | Local HBM only in practice | Same — remote tiers never evict |

**You can test today:** plug in custom `EvictionPolicy.pick_victims` (LRU, LFU, prefix-aware, etc.) for **HBM-only** victim choice.

### Unified tier optimizer (not implemented)

Real systems joint-optimize placement + eviction + pull. Here:

- `LookupPolicy` decides pull / compute / local hit per block.
- `EvictionPolicy` reacts only to local HBM slot deficit.

Missing: expected reuse, tier pressure, and bandwidth jointly influencing **where to keep** and **what to evict**.

---

## Gaps: pull vs recompute

These affect whether cost-based decisions match production behavior.

### What works today

- Per-block `resolve_block()` → local hit, `("pull", src)`, or `"compute"`.
- Token-aware recompute cost (prefill tokens vs decode steps) in `CostBasedPullLookupPolicy`.
- Per-link `BandwidthResource` with `queued_load()` = running + scheduled pool tasks.
- Pending pulls / forward reservation tracked within one `resolve_actions` call.
- `OrderedPullLookupPolicy` tier preference; `CostBasedPullLookupPolicy` min-cost vs recompute.
- Pull tasks depend on evict tasks; forward depends on evict + pull (DAG in `TaskPool`).

### Cost model gaps (remaining)

| Gap | Impact |
|-----|--------|
| **Per-block independence** | Each block gets its own pull/compute choice. No range pull, no amortized latency across a prefix chunk. |
| **Tie-break** | Equal cost → pull. Production may prefer compute to avoid tier churn. |

### Queue and interconnect gaps

| Gap | Impact |
|-----|--------|
| **Multi-hop / PD link** | `pull_sources` can point at another engine’s memory, but no separate interconnect model (NVLink vs PCIe vs RDMA, metadata lookup delay). |
| **Per-tier request queues** | No fairness, priority, or depth limits on pull queues per medium. |
| **In-flight source block** | If source has `inflight_incoming(hash)`, `resolve_block` returns `None` → allocation fails → may preempt. vLLM often waits or shares in-flight loads. |
| **Async prefetch** | No background prefetch, no overlap planning beyond the global task DAG. |

### Pull vs recompute — suggested fixes (priority)

1. Range/batch `LoadTask` with latency amortized once per range.
2. In-flight source sharing instead of hard `None`.
3. Optional prefetch queue.

---

## Gaps: scheduler and simulation model

Non-policy simplifications that still change measured outcomes.

| Gap | vLLM / production | Simulator | Affects |
|-----|-------------------|-----------|---------|
| Work unit | Tokens | Blocks (`block_size` tokens/block) | Token-budget vs block-grain mismatch |
| Physical KV | Block table + refcounts | Content-keyed hash + `holders` set | Hash collisions, non-prefix blocks not modeled |
| Cursor advance | At schedule / output | `num_computed_blocks` bumped in `apply_batch` | Tight-memory timing skew |
| Preempt | `kv_cache_manager.free` | `free_request` + reset cursor to 0 | Broadly similar; no partial rollback |
| PD write mode | Early D spawn, D gates P | **Not implemented** | Concurrent P/D overlap experiments |
| PD read mode | KV connector, async match | Late spawn, `WAITING_REMOTE_KV`, P holds KV until D done | Read-path only |
| Extreme HBM pressure | Backpressure / scheduling | Very tight HBM can still head-of-line stall (separate from PD KV bug, which is fixed) | Stress configs only |
| One in-flight batch per engine | vLLM pipeline overlap simplified | Batch must finish before next `schedule()` | Overlap undercount |

---

## Known bugs fixed (regression locked in tests)

| Issue | Symptom | Fix / test |
|-------|---------|------------|
| PD KV released on pull complete | Decode preempted after pull → second pull deadlocks | P KV held until D finishes (`simulator._apply_completed_batches`); `critical` PD tests |
| FP dust in task completion | `event time did not advance` at large n (e.g. stress n=512) | `_is_dust_work()` in `tasks.py`; `finish_done()` after `start_ready` in `step()` |
| Micro-step time semantics | Hidden multi-event steps | One event per `step()`; `critical` micro-step tests |

---

## Honest test coverage limits

| What tests prove | What they do **not** prove |
|------------------|----------------------------|
| Micro-step loop, monotonic time | Full vLLM scheduler parity |
| PD KV hold/release, organic decode preemption (tight HBM) | All eviction / placement policies |
| Stress n=16 seed=42 regression (old deadlock) | Optimality of pull vs recompute |
| 50/50 pass on stress-seeds 32–512 | Extreme HBM configs or write-mode PD |
| Per-request metrics invariants | Production trace replay accuracy |

`Simulator.run(wall_timeout_s=…)` fails fast instead of spinning to `max_steps`.

---

## PD 1P1D (read mode)

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
3. Pull completes → promote to `RUNNING` (P KV stays held until D finishes).
4. Decode completes → release held KV on P; `free_request` on D.

**Not implemented:** PD write mode (early spawn, decode gates prefill).

---

## Request metrics

Each `Request` has `metrics: RequestMetrics` (simulation clock):

| Field | Meaning |
|-------|---------|
| `queue_time`, `run_time`, `remote_kv_time` | Accumulated phase time |
| `latency` | `finished_at - released_at` |
| `computes`, `pulls`, `local_hits`, `evictions` | Per-request batch action counts |
| `preemptions`, `remote_kv_admits`, `forward_steps` | Scheduler / batch counters |

Aggregation CLI / sweep reporting: **not implemented** (fields exist per request).

---

## Test groups

| Group | What it proves |
|-------|----------------|
| `unit` | Policies, task DAG, basic metrics |
| `critical` | Micro-step loop, PD KV hold/release, organic decode preemption, n=16 regression |
| `stress` / `stress-seeds` | Full PD workload + metrics + wall/step budgets (32–512 requests) |
| `deadlock`, `pd`, `waiting` | Scheduler edge cases |

---

## Roadmap (prioritized for policy work)

### P0 — placement + duplicates

1. `PlacementPolicy` on compute complete (write to 0..N tiers).
2. Per-tier `Memory` with independent eviction and capacity.
3. `RetentionPolicy`: max copies per hash per tier / globally.
4. Spill-on-evict (HBM victim → DRAM/SSD or drop).
5. Touch order + `LRUEviction` default.

### P1 — trustworthy pull vs recompute (remaining)

1. Range/batch pull tasks with amortized latency.
2. In-flight source sharing.
3. Optional prefetch queue.

### P2 — experiment infrastructure

1. Workload / trace config (prefix length, sharing, arrivals).
2. Metrics aggregation CLI (pull ratio, evictions/tier, duplicate count, P99 latency).
3. Policy sweep harness (fixed seeds, compare policies).

### P3 — scheduler fidelity

1. Move cursor bump to schedule time.
2. PD write mode (early spawn).
3. Workload config file for stress / benchmark runs.
