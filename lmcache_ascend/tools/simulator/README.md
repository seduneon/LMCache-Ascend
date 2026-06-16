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
python3.12 simulator.py sweep                     # policy comparison (default presets × 3 seeds)
python3.12 simulator.py sweep --list-presets        # catalog
python3.12 simulator.py sweep --csv /tmp/sweep.csv  # write CSV
python3.12 simulator.py sweep --presets baseline,dram_tier --requests 128 --seeds 5
```

## Architecture

### Event loop

| File | Role |
|------|------|
| `simulator.py` | Global clock, micro-step event loop, in-flight batches, PD spawn |
| `engine.py` | `execute_batch()` (reserve, tasks), `apply_batch()` (advance state) |
| `scheduler.py` | Queues, `schedule()` → `Batch` (RUNNING → WAITING, unified allocate + preempt) |
| `tasks.py` | `ForwardTask` (batched compute), `LoadTask` (pull), `EvictTask` |
| `memory.py` | Content-keyed slot budget, holders, block states |
| `resource.py` | `ComputeResource` / `BandwidthResource` with `schedule/start/finish` |

### Policies and config

| File | Role |
|------|------|
| `workload.py` | Synthetic PD workload generator (shared by stress + sweep) |
| `sweep.py` | Policy presets, engine factory, metrics aggregation, CSV export |
| `cost_model.py` | Forward / recompute work units (SSOT for cost math) |
| `policies.py` | `EvictionPolicy`, `PlacementPolicy`, `RetentionPolicy`, `LookupPolicy` |
| `request.py` | Request state, PD phase, per-request metrics |
| `pd.py` | `PDConfig` — validates and applies read-mode flags to engines |

### Observability and tests

| File | Role |
|------|------|
| `sim_log.py` | Optional progress logging (`SimLogger`, `SIM_LOG=1`) |
| `sim_progress.py` | Live decode-completion bar for stress runs |
| `tests/run_tests.py` | Integration tests and CLI |
| `tests/test_unit.py` | Policy, task, memory, scheduler unit tests |
| `tests/test_critical.py` | Micro-step, PD KV, memory, task DAG tests |
| `tests/test_stress.py` | PD stress and seed sweeps |

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

| Class | Behavior |
|-------|----------|
| `LRUEviction` | Default. Evict unheld resident blocks with oldest `last_touch` |
| `FirstAvailableEviction` | Arbitrary order (legacy / deterministic tests) |

```python
class MyEviction(EvictionPolicy):
    def pick_victims(self, hbm, count, exclude) -> list[KVBlock]: ...

LookupPolicy(local_memory="npu-0:hbm", eviction_policy=MyEviction())
```

Blocks record `last_touch` (simulation time) on local hit (`engine._reserve`), pull complete (`LoadTask`), and forward complete (`ForwardTask`).

### Placement — `PlacementPolicy`

| Class | Behavior |
|-------|----------|
| `HBMOnly` | Default. KV stays on local HBM only |
| `HBMAndDRAM` | Mirror to DRAM on resident; **LRU-evict DRAM** when full; **spill to DRAM** on HBM evict |

```python
from policies import HBMAndDRAM, OrderedPullLookupPolicy

Engine(
    ...,
    local_memory="npu-0:hbm",
    placement_policy=HBMAndDRAM(dram_memory="npu-0:dram"),
    policy=OrderedPullLookupPolicy(
        local_memory="npu-0:hbm",
        pull_sources=["npu-0:dram"],
    ),
    transfer_links={"npu-0:dram": BandwidthResource(base_speed=32.0)},
)
```

Placement runs on forward/pull complete; `spill_on_evict` runs before HBM `EvictTask` removes the victim. Both use sync copies (zero transfer cost in v1). DRAM uses `LRUEviction` by default (`dram_eviction_policy=` to override).

### Retention — `RetentionPolicy`

| Class | Behavior |
|-------|----------|
| `UnboundedRetention` | Default. Unlimited copies per hash; pull leaves source |
| `SingleCopyPerTier` | Trim to one unheld resident copy per hash per tier |
| `ConsumeOnPull` | Remove source copy after pull if it has no holders |

```python
Engine(
    ...,
    retention_policy=ConsumeOnPull(),
)
```

`on_block_resident` runs after forward/pull (and DRAM mirrors via `HBMAndDRAM.bind_retention`). `after_pull` runs when a `LoadTask` completes.

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

recompute_work = block_recompute_work(req, block_size=…)   # see cost_model.py
              | work_per_block                         (req is None)

batch work     = batch_forward_work(batch, …)              # ForwardTask

`LookupPolicy.lookup()` is the SSOT for allocate-time block actions + eviction;
``Scheduler._allocate_blocks`` retries with preemption when lookup returns ``None``.

t_pull      = link.time_for(work_per_transfer, load(link))
t_recompute = compute.time_for(block_recompute_work(...), load(compute))  # 0 if forward reserved
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
| Placement / eviction / duplicates | **~60%** | HBM+DRAM placement, retention, spill; no SSD / composable policies |
| vLLM scheduler shape | **~85%** | Batching, preempt, chunked prefill, PD read mode |
| Sweep infrastructure | **~70%** | `sweep` CLI + CSV; workload is synthetic only (no trace replay) |

Use the gap sections below when designing experiments — running a sweep that assumes a missing feature will silently give wrong conclusions.

---

## Gaps: placement, eviction, duplicates

These are the main blockers for LMCache-style **where to put KV** and **how many copies to keep** experiments.

### Write-side placement (partial)

| Gap | Today | Needed for vLLM/LMCache alignment |
|-----|-------|-----------------------------------|
| Where computed KV goes | **Partial:** `HBMAndDRAM` mirrors + spill-on-HBM-evict | SSD / multi-tier write TBD |
| Remote tiers | DRAM tier with LRU eviction + pull source | SSD spill / remote plugins |
| Per-tier capacity | HBM + DRAM independent `Memory(size=…)` | More tiers |
| Spill on evict | **Partial:** `HBMAndDRAM.spill_on_evict` | Configurable spill vs drop policy |

Implemented via `PlacementPolicy.place_copy` and `PlacementPolicy.spill_on_evict`.

### Duplicate retention (partial)

`Memory.blocks[hash]` is a **list** of physical copies. `RetentionPolicy` enforces per-tier caps and pull source lifecycle.

| Gap | Today | Needed |
|-----|-------|--------|
| Max copies per hash | **`SingleCopyPerTier`** (per tier) | Global caps, composable policies |
| Cross-tier duplicates | Via placement + retention | e.g. HBM + DRAM + SSD simultaneously |
| Pull consumption | **`ConsumeOnPull`** when source unheld | Move semantics with held sources |
| Deduplication | Holders refcount shared blocks | Explicit canonical copy vs per-request copies |

### Eviction fidelity

| Gap | vLLM | Simulator |
|-----|------|-----------|
| Victim selection | LRU on physical blocks | Default `LRUEviction` |
| Touch on access | Updates LRU on hit / use | `Memory.touch` on local hit, pull/forward complete |
| Prefix-aware scoring | Prefer evicting unshared / low-reuse | Not modeled |
| Watermarks | Reserved blocks for decode vs prefill | Not modeled |
| Eviction timing | Often synchronous at allocation | Async `EvictTask` on compute resource (distorts pressure timing) |
| Eviction scope | HBM evict + DRAM LRU in `HBMAndDRAM` | More tiers / spill targets |

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

Aggregation CLI / sweep reporting: **`simulator.py sweep`** (see below).

---

## Policy sweep (`sweep`)

Compare policies on the **same synthetic PD workload** (`workload.py`) with fixed seeds.

```bash
python3.12 simulator.py sweep
python3.12 simulator.py sweep --presets baseline,ordered_pull,dram_tier,consume_on_pull
python3.12 simulator.py sweep --requests 128 --seeds 10 --csv results.csv
python3.12 simulator.py sweep --list-presets
```

| Preset | P engine | D engine | Extras |
|--------|----------|----------|--------|
| `baseline` | compute-only | cost-based pull from P HBM | stress default |
| `ordered_pull` | compute-only | ordered pull (no cost model) | |
| `dram_tier` | HBM+DRAM mirror/spill | cost-pull from DRAM then HBM | extra tier |
| `consume_on_pull` | baseline | baseline | `ConsumeOnPull` |
| `single_copy` | baseline | baseline | `SingleCopyPerTier` |

CSV columns include decode P50/P99 latency, pull ratio, evictions, preemptions, and `tier_used_at_end` (leak check; all zeros when idle).

**Blindspots (read before drawing conclusions):**

- **Synthetic workload only** — shared-prefix random generator, not trace replay.
- **`pull_ratio`** — decode `pulls / (pulls + computes)`; local hits excluded.
- **No in-run peak memory / duplicate count** — `dram_slots_used` is occupancy at end (expected >0 for `dram_tier`); HBM tiers must be empty.
- **PD read mode only** — same as stress tests.
- **Preset = full stack** — lookup + placement + retention bundled; not isolated single-knob sweeps yet.
- **Failed runs** — recorded in CSV with `status=fail` and `error`; CLI exits non-zero if any fail.

Extend presets in `sweep.py` (`PRESETS` dict + `build_engines`).

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

1. ~~`PlacementPolicy` on compute complete~~ (`HBMAndDRAM` + spill).
2. ~~Per-tier `Memory` with independent eviction~~ (DRAM LRU in `HBMAndDRAM`).
3. ~~`RetentionPolicy`: max copies per hash per tier / globally~~ (`SingleCopyPerTier`, `ConsumeOnPull`).
4. ~~Spill-on-evict (HBM victim → DRAM)~~ (`spill_on_evict`).
5. ~~Touch order + `LRUEviction` default~~ (done).

### P1 — trustworthy pull vs recompute (remaining)

1. Range/batch pull tasks with amortized latency.
2. In-flight source sharing.
3. Optional prefetch queue.

### P2 — experiment infrastructure

1. Workload / trace config (prefix length, sharing, arrivals) — **partial:** `WorkloadConfig` in `workload.py`.
2. Metrics aggregation CLI — **partial:** `simulator.py sweep` + CSV.
3. Policy sweep harness — **partial:** `sweep.py` presets; extend for isolated knob sweeps.

### P3 — scheduler fidelity

1. Move cursor bump to schedule time.
2. PD write mode (early spawn).
3. Workload config file for stress / benchmark runs.
