# KV Cache Simulator — Progress & Roadmap

Discrete-event simulator for KV cache scheduling, eviction, tiering, and PD transfer. Block-grained workflow model shaped like vLLM’s batch scheduler (not cycle-accurate). Goal: compare KV cache **placement / eviction / duplicate retention** and **pull vs recompute** policies aligned with vLLM + LMCache-Ascend.

Run policy comparisons: `python3.12 simulator.py sweep` (see `sweep.py` for presets).

---

## Experiment readiness

| Policy area | Readiness | Honest assessment |
|-------------|-----------|-------------------|
| Pull vs recompute (read path) | **~85%** | Cost model + bandwidth queues; chunk batch pulls amortize latency; in-flight remote wait avoids spurious preempt. |
| Placement / eviction / duplicates | **~75%** | HBM+DRAM sync + SSD paid writes via `TieredPlacement`/`StoreTask`; `GlobalCopyCap`; sync eviction at allocate. |
| vLLM scheduler shape | **~80%** | Batching, preempt, chunked prefill, PD read mode. Block-grain, one in-flight batch per engine. |
| Sweep infrastructure | **~70%** | CLI + CSV; `ssd_tier`, `global_cap_2` presets; peak duplicate metric. Synthetic workload only. |

**Bottom line:** P0/P1 policy hooks are in place for credible placement and pull-vs-recompute sweeps on synthetic PD workloads. Joint tier optimizer, trace replay, and production-faithful async write queues remain open.

---

## Done (policy-relevant)

### P0 — placement + duplicates

- `TieredPlacement`: multi-tier mirror/spill; sync tiers (DRAM) + paid async tiers (`StoreTask` on `write_links`)
- `HBMAndDRAM` preserved (sync DRAM mirror/spill)
- `GlobalCopyCap`: cross-tier resident copy limit with tier-preference eviction
- `StoreTask`: bandwidth-priced writes to downstream tiers

### P1 — pull vs recompute + eviction timing

- `BatchLoadTask`: chunk-aligned pull groups; latency/work amortized once per group
- In-flight remote source → `"local"` wait (attach holder on source inflight); no `None`→preempt
- Sync eviction at allocate (`sync_evict=True` default; `work_per_evict=0`)

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
| Pull vs recompute | Prefetch queue; multi-hop interconnect; configurable tie-break |
| Placement | Background write queue; spill-vs-drop policy knob; composable policy stacks |
| Retention | Canonical vs per-request copy semantics with held sources |
| Eviction | Prefix-aware scoring; decode/prefill watermarks |
| Optimizer | Joint placement + eviction + lookup on expected reuse |
| Infrastructure | Trace replay; isolated single-knob sweeps; tier occupancy time series |
| Scheduler | PD write mode; cursor at schedule time; pipeline overlap |

---

## Prioritized build order (next)

| Priority | Work | Unlocks |
|----------|------|---------|
| **P2** | Unified placement + eviction + lookup hook | Joint policies |
| **P2** | Trace/workload config + tier occupancy time series in sweep | Production-shaped experiments |
| **P3** | Prefetch queue, multi-hop links, PD write mode | Production parity |

---

## Completed roadmap items

- ~~P0: multi-tier placement + paid SSD writes~~
- ~~P0: global cross-tier retention~~
- ~~P1: batch/range pulls + amortized latency~~
- ~~P1: in-flight source sharing~~
- ~~P1: sync eviction at allocate~~
- ~~`PlacementPolicy` / `RetentionPolicy` / cost-based pull~~ (prior milestones)
- ~~PD read mode + KV hold until decode completes~~

---

## What tests validate (and do not)

| Validated | Not validated |
|-----------|---------------|
| P0/P1 unit tests: `GlobalCopyCap`, `TieredPlacement`+`StoreTask`, batch pulls, remote wait, sync evict | Full vLLM scheduler parity |
| Stress/critical regressions | Production trace accuracy |
| Sweep presets incl. `ssd_tier`, `global_cap_2` | Optimality of joint tier policies |
