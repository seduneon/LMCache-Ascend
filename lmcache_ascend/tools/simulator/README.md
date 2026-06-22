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
| Sweep infrastructure | **~80%** | CLI + CSV; data-driven presets; Mooncake trace replay via `--trace`. |

**Bottom line:** Credible for **single-knob** policy comparisons on synthetic or Mooncake trace workloads. Not trustworthy for joint optimizer claims or production latency numbers.

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
| `eviction.py` | `LRUEviction`, `FIFOEviction`, `RandomEviction`, `make_hbm_eviction` |
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
| **Synthetic string hashes** | Prefix sharing is workload-shaped, not content-hash-shaped. Trace mode uses Mooncake chunk ids as block keys. |
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
| `evict_lru`, `evict_fifo`, `evict_random` | LMCache PD: P stores prefix to DRAM on complete then frees P HBM; D pulls **DRAM only**; vLLM APC on **D HBM** only; compare LRU/FIFO/random on HBM and DRAM |

Use `--presets evict_lru,evict_fifo,evict_random` with tight `--hbm-gib` / `--dram-gib` (and `--drop-oversized` on traces) to stress tier eviction. Compare `tier_evictions`, `lifecycle_hbm_frees`, and hit ratios in CSV output — distinct from `prefill_evictions` / policy evictions at admit time.

Use `python3.12 -m simulator.sweep --list-presets` for the catalog.

`--csv` writes a wide compare table (metric rows, preset columns). Use `--raw-csv` for one row per preset/seed run.

Prefix-aware metrics (prompt KV vs decode `blk:` output slots):
- `prefix_pull_ratio` — decode prefix pulls / (prefix pulls + prefix recomputes)
- `dram_hit_rate` — decode prefix pulls from DRAM / all decode prefix pulls
- `pull_ratio` — legacy: all decode pulls / (pulls + computes); dominated by output tokens
- `decode_hit_ratio` — decode prefix local HBM hits / prefix block resolutions

### Mooncake trace replay

Bundled trace: `simulator/traces/synthetic_trace.jsonl` (from [Mooncake FAST'25 release](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/traces/synthetic_trace.jsonl)).

```bash
cd lmcache_ascend/tools
python3.12 -m simulator.sweep --trace --requests 64 --presets baseline,ordered_pull
python3.12 -m simulator.sweep --trace /path/to/synthetic_trace.jsonl --requests 128 --trace-offset 100
```

Mapping: each `hash_id` is one 512-token prefix block; `timestamp` × `--trace-time-scale` (default 0.001, ms→s); `output_length` ÷ `--tokens-per-block` (default **512**) → decode blocks. The same trace slice is replayed for every seed; seed only affects randomized eviction.

### Tier capacity (GiB → block slots)

Each tier is a `TierSpec(tier_key, capacity_gib, chunk_blocks)`. The sweep CLI sets these via `--hbm-gib`, `--dram-gib`, `--ssd-gib`, and `--dram-chunk-blocks`; tests can pass an explicit `tiers=` tuple on `SimConfig`.

```text
slots = floor(gib × 1024³ / (tokens_per_block × kv_bytes_per_token × chunk_blocks))
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--hbm-gib` | 32 | HBM per engine (`npu-0:hbm`, `npu-1:hbm`; chunk_blocks=1) |
| `--dram-gib` | 64 | DRAM tier (`npu-0:dram`) |
| `--ssd-gib` | 256 | SSD tier (`npu-0:ssd`) |
| `--kv-model` | llama3-8b | Reference K+V bytes/token (`toy`, `llama3-8b`, `llama3-70b`) |
| `--kv-bytes-per-token` | (from model) | Override bytes/token for conversion |
| `--tokens-per-block` | 512 | Tokens per KV block (trace + capacity) |
| `--dram-chunk-blocks` | 4 | HBM blocks per DRAM slot |

At sweep start, resolved slot counts are printed per tier (`tier capacity: npu-0:hbm=…; …`).

Set **`--hbm-gib`** explicitly. Use **`--drop-oversized`** with trace workloads so requests with `prefix + decode blocks > hbm slots` are skipped instead of deadlocking the run.

---

## Remaining gaps

| Area | Still missing |
|------|----------------|
| Joint planning | Unified placement + eviction + lookup in one optimizer |
| Pull vs recompute | Prefetch queue; multi-hop interconnect |
| Infrastructure | Tier occupancy time series |
| Scheduler | PD write mode; pipeline overlap |
