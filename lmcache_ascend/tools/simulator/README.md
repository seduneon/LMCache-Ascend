# KV Cache Simulator — Progress & Roadmap

Discrete-event simulator for KV cache scheduling, eviction, tiering, and PD transfer. Block-grained workflow model shaped like vLLM’s batch scheduler (not cycle-accurate). Goal: compare KV cache **placement / eviction / duplicate retention** and **pull vs recompute** policies aligned with vLLM + LMCache-Ascend.

Run policy comparisons: `python3.12 simulator.py sweep` (see `sweep.py` for presets).

---

## Experiment readiness

| Policy area | Readiness | Honest assessment |
|-------------|-----------|-------------------|
| Pull vs recompute (read path) | **~85%** | Cost model + bandwidth queues; chunk batch pulls; `"wait"` on remote in-flight; batch-local pull dedupe. |
| Placement / eviction / duplicates | **~75%** | HBM+DRAM sync + SSD paid writes; `GlobalCopyCap`; sync eviction at allocate. |
| vLLM scheduler shape | **~80%** | Batching, preempt, chunked prefill, PD read mode. Block-grain, one in-flight batch per engine. |
| Sweep infrastructure | **~70%** | CLI + CSV; `ssd_tier`, `global_cap_2`; peak duplicate metric. Synthetic workload only. |

**Bottom line:** Credible for **single-knob** policy comparisons on synthetic PD workloads. Not trustworthy for joint optimizer claims, trace replay, or production latency numbers.

---

## Architecture & sweep validity

### Sound for this tool’s purpose

- Micro-step loop + shared `TaskPool` (cross-engine bandwidth contention)
- `Scheduler` / `Engine` / `Simulator` split; `LookupResult` as allocate-time SSOT
- Pluggable policies; task DAG for evict → pull → forward → store
- `chunk_hash.py` for LMCache chunk keys (separate from vLLM block-table shape)

### Structural limits (read before interpreting sweeps)

| Limit | Effect on experiments |
|-------|------------------------|
| **No unified content id** | HBM block hashes vs tier chunk keys are mapped via helpers, not a single type. Retention/dedup/metrics can drift when adding tiers — verify with tests, not assumptions. |
| **Decoupled policies** | Lookup, eviction, placement, retention do not joint-optimize. Comparing “placement presets” may be dominated by eviction/pull behavior. Prefer isolated knobs or wait for P2 coordinator. |
| **Synthetic string hashes** | Prefix sharing is workload-shaped, not content-hash-shaped. Invalid for trace replay or collision/dedup-at-scale claims. |
| **Schedule vs execute split** | Cost model decides at `schedule()`; `BatchLoadTask` executes once per chunk. Relative pull-vs-recompute ordering is OK; absolute times are approximate. |
| **Batch-local pull dedupe** | Same chunk pulled once per batch, not across steps/engines. |
| **`"wait"` = reschedule** | Remote in-flight chunks block cursor advance; no explicit pull-future object. |
| **One in-flight batch / engine** | No pipeline overlap; block-grain not token-grain. |

### Acceptable simplifications (documented, not bugs)

- Sync eviction at allocate (`sync_evict=True` default)
- Read/write bandwidth as separate resources per tier
- Store tasks planned in `execute_batch` (ordered by task DAG, not a background write queue)
- `GlobalCopyCap` needs `req` context for cross-chunk-tier trimming; HBM-only presets are the well-tested case

### What sweeps are good for

- Relative ordering: cost-pull vs ordered-pull vs recompute-heavy baseline
- Tiering direction: `dram_tier` / `ssd_tier` vs `baseline` under same workload
- Retention direction: `consume_on_pull`, `single_copy`, `global_cap_2`
- Regression: stress/critical tests after policy changes

### What sweeps are not good for

- Absolute latency / throughput vs production
- Joint “best” placement + eviction + pull policy
- Multi-tenant trace replay or hash-collision behavior
- PD write mode or full LMCache connector semantics

---

## Done (policy-relevant)

### P0 — placement + duplicates

- `TieredPlacement`: multi-tier mirror/spill; sync tiers (DRAM) + paid async tiers (`StoreTask` on `write_links`)
- `HBMAndDRAM` preserved (sync DRAM mirror/spill)
- `GlobalCopyCap`: cross-tier resident copy limit with tier-preference eviction
- `StoreTask`: bandwidth-priced writes to downstream tiers

### P1 — pull vs recompute + eviction timing

- `BatchLoadTask`: chunk-aligned pull groups; latency/work amortized once per group
- In-flight remote source → `"wait"` (no cursor advance; holder on source inflight; no `None`→preempt)
- Batch-local pull dedupe via `(src_key, chunk_key)` registry
- Sync eviction at allocate (`sync_evict=True` default; `work_per_evict=0`)
- Paid-tier spill: HBM remove deferred until spill `StoreTask` completes

### Existing (unchanged)

- `CostBasedPullLookupPolicy`, token-aware recompute, per-link queues
- `RetentionPolicy`: `UnboundedRetention`, `SingleCopyPerTier`, `ConsumeOnPull`
- PD read mode, micro-step loop, stress/critical regressions

### Sweep presets

| Preset | Notes |
|--------|-------|
| `baseline`, `ordered_pull`, `dram_tier`, `consume_on_pull`, `single_copy` | Original set |
| `ssd_tier` | P sync DRAM + paid SSD writes; D cost-pull from SSD/DRAM/HBM |
| `global_cap_2` | `GlobalCopyCap(2)` across P and D HBM |

CSV adds `ssd_slots_used`, `peak_duplicate_count`.

---

## Remaining gaps

| Area | Still missing |
|------|----------------|
| Architecture | First-class `ContentKey`; batch-scoped task ownership; optional joint policy coordinator |
| Pull vs recompute | Prefetch queue; multi-hop interconnect; schedule/execute cost alignment |
| Placement | Background write queue; spill-vs-drop knob |
| Retention | Canonical copy semantics; `ConsumeOnPull` + chunk-tier interaction |
| Eviction | Prefix-aware scoring; decode/prefill watermarks |
| Infrastructure | Trace replay; isolated single-knob sweeps; tier occupancy time series |
| Scheduler | PD write mode; cursor at schedule time; pipeline overlap |

---

## Prioritized build order (next)

| Priority | Work | Unlocks |
|----------|------|---------|
| **P2** | `ContentKey` + batch task group | Correct cross-tier retention/metrics; sturdier batch lifecycle |
| **P2** | Unified placement + eviction + lookup hook | Joint policies |
| **P2** | Trace/workload config + tier occupancy time series in sweep | Production-shaped experiments |
| **P3** | Prefetch queue, multi-hop links, PD write mode | Production parity |

---

## Completed roadmap items

- ~~P0: multi-tier placement + paid SSD writes~~
- ~~P0: global cross-tier retention~~
- ~~P1: batch/range pulls + amortized latency~~
- ~~P1: in-flight source sharing (`"wait"`)~~
- ~~P1: sync eviction at allocate~~
- ~~P1: batch-local pull dedupe~~
- ~~`PlacementPolicy` / `RetentionPolicy` / cost-based pull~~ (prior milestones)
- ~~PD read mode + KV hold until decode completes~~

---

## What tests validate (and do not)

| Validated | Not validated |
|-----------|---------------|
| P0/P1 unit tests: `GlobalCopyCap`, `TieredPlacement`+`StoreTask`, batch pulls, remote wait, sync evict, pull dedupe | Full vLLM scheduler parity |
| Stress/critical regressions | Production trace accuracy |
| Sweep presets incl. `ssd_tier`, `global_cap_2` | Joint tier policy optimality |
| Relative policy ordering on synthetic PD workload | Absolute production latency |
