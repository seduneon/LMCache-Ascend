# KV Cache Simulator — Progress & Roadmap

Discrete-event simulator for KV cache scheduling, eviction, tiering, and PD transfer. Block-grained workflow model shaped like vLLM’s batch scheduler (not cycle-accurate). Goal: experiment with KV cache **placement / eviction / duplicate retention** and **pull vs recompute** policies aligned with vLLM + LMCache-Ascend.

Run policy comparisons: `python3.12 simulator.py sweep` (see `sweep.py` for presets).

---

## Experiment readiness

| Policy area | Readiness | Honest assessment |
|-------------|-----------|-------------------|
| Pull vs recompute (read path) | **~75%** | Cost model + bandwidth queues work; per-block pulls and in-flight `None`→preempt bias results under contention. Good for **relative** policy ordering after P1 fixes, not production numbers yet. |
| Placement / eviction / duplicates | **~55%** | HBM+DRAM hooks exist; mirrors/spills are zero-cost sync copies. No SSD write path, no global duplicate caps, eviction decoupled from lookup. HBM-only custom eviction is testable today. |
| vLLM scheduler shape | **~80%** | Batching, preempt, chunked prefill, PD read mode. Block-grain (not token-grain), cursor bumped at `apply_batch`, one in-flight batch per engine. |
| Sweep infrastructure | **~65%** | CLI + CSV on synthetic PD workload. No trace replay, no isolated single-knob sweeps, no in-run peak memory / duplicate metrics. |

**Bottom line:** Read-path pull-vs-recompute is the most mature surface. Multi-tier placement and duplicate-count experiments are **not trustworthy** until P0 lands. Do not run sweeps that assume missing features — conclusions will be silently wrong.

---

## Done (policy-relevant)

### Pull vs recompute

- `ComputeOnlyLookupPolicy`, `OrderedPullLookupPolicy`, `CostBasedPullLookupPolicy`
- Token-aware recompute cost (`cost_model.py`: prefill tokens vs decode steps)
- Per-link `BandwidthResource` with `queued_load()`; pending pulls tracked within one allocation batch
- LMCache-aligned chunk keys (`chunk_hash.py`) and chunk-aware pull work (`transfer_work_units`)
- Task DAG: evict → pull → forward (`tasks.py`, `engine.py`)

### Placement / eviction / duplicates

- `EvictionPolicy` hook; default `LRUEviction` with `last_touch` on hit / pull / forward
- `HBMOnly`, `HBMAndDRAM` (mirror on resident, spill on HBM evict, DRAM LRU eviction)
- `RetentionPolicy`: `UnboundedRetention`, `SingleCopyPerTier`, `ConsumeOnPull`
- Content-keyed `Memory` with per-hash copy lists and holder refcounts

### Scheduler & PD (enables policy experiments)

- vLLM-shaped queues: RUNNING → WAITING, unified allocate + preempt, chunked prefill
- PD read mode: late decode spawn, `WAITING_REMOTE_KV`, P holds KV until D finishes (`pd.py`)
- Micro-step event loop (one event per `step()`); regressions locked in `critical` / `stress` tests

### Experiment harness (partial)

- `sweep.py` presets: `baseline`, `ordered_pull`, `dram_tier`, `consume_on_pull`, `single_copy`
- Per-request metrics (`RequestMetrics`); CSV with decode P50/P99, pull ratio, evictions, preemptions
- Synthetic shared-prefix workload (`workload.py` + `WorkloadConfig`)

---

## Remaining gaps

### Placement, eviction, duplicates

| Gap | Today | Blocks |
|-----|-------|--------|
| Write-side tiers | HBM→DRAM mirror/spill only | SSD / composable multi-tier write |
| Spill / mirror cost | Sync, zero bandwidth | Link-cost writes; configurable spill vs drop |
| Duplicate caps | Per-tier (`SingleCopyPerTier`) | Global / cross-tier caps; canonical vs per-request copies |
| Eviction scoring | LRU on unheld blocks | Prefix-aware, shared-block preference, decode/prefill watermarks |
| Eviction timing | Async `EvictTask` on compute resource | Sync at allocate (vLLM-like pressure) |
| Joint optimization | Lookup, eviction, placement are separate hooks | Expected reuse + tier pressure + bandwidth in one decision |

### Pull vs recompute

| Gap | Impact |
|-----|--------|
| Per-block independence | Overcounts pull latency vs LMCache chunk pulls; no amortized range latency |
| In-flight source | `tier_inflight_hbm_block` → `None` → preempt; production waits or shares loads |
| No prefetch | No background overlap beyond the task DAG |
| Interconnect | Single hop per `pull_sources` entry; no NVLink/PCIe/RDMA/metadata delay model |
| Per-tier queue policy | Bandwidth queue only; no depth limits or fairness |
| Tie-break | Equal cost → pull (may over-favor tier churn) |

### Infrastructure & model fidelity

| Gap | Impact |
|-----|--------|
| Synthetic workload only | Sharing/reuse patterns not production-shaped |
| Bundled sweep presets | Cannot isolate one policy knob without editing `sweep.py` |
| Metrics | No in-run peak memory, duplicate counts, or tier occupancy time series |
| PD write mode | Not implemented |
| Block vs token grain | `block_size` abstraction; token budget vs block allocation mismatch |
| Pipeline overlap | One in-flight batch per engine undercounts vLLM overlap |

### LMCache-Ascend alignment

| Aligned | Not yet |
|---------|---------|
| Chunk key mapping (`lmcache_chunk_hash`, `chunk_blocks`) | SSD / storage backend tier |
| PD read-mode pull path (Mooncake-shaped) | Write path, connector async match, chunk-grain `LoadTask` |
| Policy hooks mirroring tier concepts | Production connector bandwidth / queue behavior |

---

## Prioritized build order

Focus: **placement / eviction / duplicates** and **pull vs recompute**. Order reflects dependency and experiment unblockers.

| Priority | Work | Unlocks |
|----------|------|---------|
| **P0** | SSD / multi-tier `PlacementPolicy` + spill with real link cost on write | Multi-tier placement experiments |
| **P0** | Global / cross-tier `RetentionPolicy` (max copies system-wide, tier preferences) | Duplicate-count algorithm sweeps |
| **P1** | Range/batch `LoadTask` + amortized latency (align with `chunk_blocks`) | Honest pull vs recompute for LMCache chunks |
| **P1** | In-flight source sharing (wait/dedupe, not `None`→preempt) | Contention-faithful pull policies |
| **P1** | Sync eviction at allocate (or vLLM-style watermark eviction) | Eviction algorithm experiments under pressure |
| **P2** | Unified placement + eviction + lookup optimizer hook (shared expected-reuse signal) | Joint policies, not three disconnected hooks |
| **P2** | Trace/workload config + per-run tier/duplicate metrics in sweep CSV | Reproducible, interpretable sweeps |
| **P3** | Prefetch queue, multi-hop links, PD write mode | Production parity (not core algorithm prototyping) |

---

## Completed roadmap items

- ~~`PlacementPolicy` on compute complete~~ (`HBMAndDRAM` + spill)
- ~~Per-tier `Memory` with independent eviction~~ (DRAM LRU in `HBMAndDRAM`)
- ~~`RetentionPolicy`: per-tier copy caps + pull consumption~~ (`SingleCopyPerTier`, `ConsumeOnPull`)
- ~~Spill-on-evict (HBM victim → DRAM)~~ (`spill_on_evict`)
- ~~Touch order + `LRUEviction` default~~
- ~~Cost-based pull vs recompute with queued bandwidth~~ (`CostBasedPullLookupPolicy`)
- ~~PD read mode + KV hold until decode completes~~ (regression tests in `test_critical.py`)

---

## What tests validate (and do not)

| Validated | Not validated |
|-----------|---------------|
| Micro-step loop, monotonic time | Full vLLM scheduler parity |
| PD KV hold/release, decode preemption under tight HBM | All placement / retention / eviction policies |
| Stress regressions (incl. seed sweeps 32–512) | Optimality of pull vs recompute decisions |
| Per-request metric invariants | Production trace accuracy |
| Sweep preset runs + CSV export | Extreme HBM configs, PD write mode, SSD tier |

Extend presets and metrics in `sweep.py` after P0/P1 — not before, or sweeps compare policies on a model that violates their assumptions.
