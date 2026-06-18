# KV Cache Simulator — Progress & Roadmap

Discrete-event simulator for KV cache scheduling, eviction, tiering, and PD transfer. Block-grained workflow model shaped like vLLM's batch scheduler (not cycle-accurate). Goal: compare KV cache **placement / eviction / duplicate retention** and **pull vs recompute** policies aligned with vLLM + LMCache-Ascend.

Run policy comparisons: `python3.12 simulator.py sweep` (see `sweep.py` and `presets.py`).

---

## Experiment readiness

| Policy area | Readiness | Honest assessment |
|-------------|-----------|-------------------|
| Pull vs recompute (read path) | **~85%** | Bandwidth queues; chunk batch pulls; `"wait"` on remote in-flight; batch-local pull dedupe. |
| Placement / eviction / duplicates | **~75%** | HBM+DRAM sync + SSD paid writes; `GlobalCopyCap`; sync eviction at allocate. |
| vLLM scheduler shape | **~80%** | Batching, preempt, chunked prefill, PD read mode. Block-grain, one in-flight batch per engine. |
| Sweep infrastructure | **~75%** | CLI + CSV; data-driven presets; `build_pd_engines`; peak duplicate metric. Synthetic workload only. |

**Bottom line:** Credible for **single-knob** policy comparisons on synthetic PD workloads. Not trustworthy for joint optimizer claims, trace replay, or production latency numbers.

---

## Pipeline architecture

Data flows in one direction through these modules:

```
Simulator.step()
  → Engine.try_schedule_and_execute()
      → Scheduler.schedule()           # admit + SchedulePolicy.lookup()
      → enrich_entry_plan()            # plan-time store/spill ops (EffectPolicy)
      → BatchRunner.run()              # execute.py
  → Engine.apply_plan()                # advance request cursors
  → Simulator.dispatch()               # DecodeSpawn / KvRelease PD events
```

| Module | Role |
|--------|------|
| `policies.py` | `EnginePolicies` bundle: schedule + effects |
| `schedule.py` | `SchedulePolicy`: block resolution + HBM eviction |
| `effects.py` | `EffectPolicy`: placement, retention, store/spill planning |
| `presets.py` | `PresetSpec`, `PRESETS`, `build_pd_engines()` |
| `topology.py` | Tier memory layout (`hbm_only`, `hbm_dram`, `hbm_dram_ssd`) |
| `plan.py` | `ScheduleResult`, `BatchPlan`, `EntryPlan`, `StoreOp` |
| `execute.py` | `BatchRunner`: reservations, tasks, effect callbacks |
| `placement.py` / `retention.py` | Effect implementations (mirrors, caps, consume-on-pull) |
| `eviction.py` | `EvictionPolicy`, `LRUEviction` |
| `tier_allocator.py` | Shared downstream tier slot acquire + eviction |
| `kv_content.py` | `ContentKey`, tier slot mapping |
| `lookup.py` | Compatibility shims (`ComputeOnlyLookupPolicy`, etc.) |
| `events.py` | Cross-engine messages (`DecodeSpawn`, `KvRelease`) |

### Policy model

Each engine holds an `EnginePolicies` bundle:

- **Schedule** (`SchedulePolicy`): resolve per-block actions (`compute`, `pull`, `wait`, local hit) and HBM evictions at admit time.
- **Effects** (`EffectPolicy`): execute-time mirrors/spills/retention; expands async store and spill ops into `EntryPlan` before execute.

Sweep presets are data rows in `presets.PRESETS` interpreted by `build_pd_engines()` — no per-preset Python wiring.

### Structural limits (read before interpreting sweeps)

| Limit | Effect on experiments |
|-------|------------------------|
| **Decoupled schedule vs effects** | Lookup/eviction at admit; placement/retention at execute. Joint optimization is future work. |
| **Synthetic string hashes** | Prefix sharing is workload-shaped, not content-hash-shaped. |
| **Batch-local pull dedupe** | Same chunk pulled once per batch, not across steps/engines. |
| **`"wait"` = reschedule** | Remote in-flight chunks block cursor advance. |
| **One in-flight batch / engine** | No pipeline overlap; block-grain not token-grain. |

### What sweeps are good for

- Relative ordering: ordered-pull vs compute-only baseline
- Tiering direction: `dram_tier` / `ssd_tier` vs `baseline`
- Retention direction: `consume_on_pull`, `single_copy`, `global_cap_2`
- Regression: stress/critical tests after policy changes

---

## Sweep presets

| Preset | Notes |
|--------|-------|
| `baseline`, `ordered_pull` | P compute-only; D ordered pull from P HBM |
| `dram_tier` | P sync DRAM mirror; D pull DRAM then HBM |
| `consume_on_pull`, `single_copy` | Retention variants on baseline topology |
| `ssd_tier` | P sync DRAM + async SSD writes; D pull SSD/DRAM/HBM |
| `global_cap_2` | `GlobalCopyCap(2)` across P and D HBM |

Use `python3.12 -m simulator.sweep --list-presets` for the catalog.

---

## Remaining gaps

| Area | Still missing |
|------|----------------|
| Joint planning | Unified placement + eviction + lookup in one optimizer |
| Pull vs recompute | Prefetch queue; multi-hop interconnect |
| Infrastructure | Trace replay; tier occupancy time series |
| Scheduler | PD write mode; pipeline overlap |
