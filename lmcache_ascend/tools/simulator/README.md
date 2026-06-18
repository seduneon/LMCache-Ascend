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
- `Scheduler` / `Engine` / `Simulator` split; `EntryPlan` / `BatchWork` as pipeline SSOT
- `KVController` + `EffectInterpreter` separate plan from execute
- Pluggable policies; task DAG for evict → pull → forward → store
- `TierAllocator` for shared downstream tier eviction (placement says *where*)
- `content_key.py` for tier-independent `ContentKey`; `chunk_hash.py` for tier slot mapping

---

## Pipeline architecture

Data flows in one direction through these modules:

```
Simulator.step()
  → Engine.schedule_batch()       # Scheduler + SimContext.capture()
  → Engine.make_work()            # ScheduleResult → BatchWork
  → BatchExecutor.execute()       # memory + TaskPool (effect_interpreter.py)
  → Engine.apply_work()           # advance request cursors
  → Simulator.dispatch()          # DecodeSpawn / KvRelease PD events
```

| Module | Role |
|--------|------|
| `plan.py` | `ScheduleResult`, `BatchWork`, `EntryPlan`, `SimContext`, `ExecuteResult` |
| `eviction.py` | `EvictionPolicy`, `LRUEviction` (shared by lookup + tier allocator) |
| `kv_controller.py` | Planning facade; future joint policy coordinator |
| `effect_interpreter.py` | `BatchExecutor` + `ResidentEffects` |
| `tier_allocator.py` | Shared downstream tier slot acquire + eviction |
| `events.py` | Explicit cross-engine messages (`DecodeSpawn`, `KvRelease`) |
| `policies.py` | Placement, retention, lookup implementations |

### Structural limits (read before interpreting sweeps)

| Limit | Effect on experiments |
|-------|------------------------|
| **Decoupled policies** | Lookup, eviction, placement, retention still plan separately inside `KVController`. Joint optimization is the next step. |
| **Synthetic string hashes** | Prefix sharing is workload-shaped, not content-hash-shaped. Invalid for trace replay or collision/dedup-at-scale claims. |
| **Schedule vs execute split** | Cost model decides at `schedule()`; `BatchLoadTask` runs once per chunk. Relative pull-vs-recompute ordering is OK; absolute times are approximate. |
| **Batch-local pull dedupe** | Same chunk pulled once per batch, not across steps/engines. |
| **`"wait"` = reschedule** | Remote in-flight chunks block cursor advance; no explicit pull-future object. |
| **One in-flight batch / engine** | No pipeline overlap; block-grain not token-grain. Batch completion tracks all pool tasks tagged with `batch_id`. |

### Acceptable simplifications (documented, not bugs)

- Sync eviction at allocate (`sync_evict=True` default)
- Read/write bandwidth as separate resources per tier
- Store tasks planned in `BatchExecutor.execute()` (ordered by task DAG, not a background write queue)
- `GlobalCopyCap` uses `ContentKey` + `req` for cross-chunk-tier trimming

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
- Batch-local pull dedupe via `(src_key, ContentKey)` registry
- `BatchContext` + `task.batch_id`: batch completes when all tagged pool tasks finish
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
| Architecture | Joint planning inside `KVController` (placement + eviction + lookup together) |
| Pull vs recompute | Prefetch queue; multi-hop interconnect; schedule/execute cost alignment via `QueueSnapshot` |
| Execution | Task callbacks → `TaskDone` events (interpreter still uses callbacks today) |
| Placement | Background write queue; spill-vs-drop knob |
| Retention | Canonical copy semantics; `ConsumeOnPull` + chunk-tier interaction |
| Eviction | Prefix-aware scoring; decode/prefill watermarks |
| Infrastructure | Trace replay; isolated single-knob sweeps; tier occupancy time series |
| Scheduler | PD write mode; pipeline overlap |

---

## Prioritized build order (next)

| Priority | Work | Unlocks |
|----------|------|---------|
| **P2** | Joint planning in `KVController` | Unified placement + eviction + lookup |
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
- ~~P2: pipeline refactor (`ScheduleResult` → `BatchWork`, `BatchExecutor`, `TierAllocator`, PD events)~~
- ~~P2: `ContentKey` + batch-scoped task ownership~~
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
