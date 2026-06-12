# KV Cache Simulator

Discrete-event simulator for KV cache scheduling, eviction, compute, and PD transfer. Block-grained workflow model aligned with vLLM’s batch scheduler (not a cycle-accurate GPU sim).

```bash
cd lmcache_ascend/tools/simulator
python3.12 simulator.py            # PD demo + deadlock + limits tests
python3.12 simulator.py deadlock   # preemption test only
python3.12 simulator.py limits     # max_num_seqs / token_budget test only
```

## Architecture

| File | Role |
|------|------|
| `scheduler.py` | Queues, `schedule()` → `Batch` (RUNNING → WAITING, preempt at allocate) |
| `engine.py` | `execute_batch()` (reserve, tasks), `apply_batch()` (advance state) |
| `simulator.py` | Global clock, multi-engine step, PD spawn |
| `policies.py` | `LookupPolicy` (pull/compute), `EvictionPolicy` |
| `tasks.py` | `ForwardTask` (batched compute), `LoadTask` (pull), `EvictTask` |
| `memory.py` | Content-keyed slot budget, holders, block states |

Each `Simulator.step()`:

1. `release_arrivals`
2. Per engine: `batch = schedule()` → `execute_batch(batch)`
3. Drain all batch tasks
4. `apply_batch(batch)`; spawn decode on prefill complete

Preemption frees KV and resets the request; it does not touch the task pool — the step drains before the next `schedule()`.

## Scheduling (vLLM-shaped)

`Scheduler.schedule()` per engine, once per step:

1. **RUNNING** (FCFS) — decode output: 1 token/request (`block_size`), subject to shared `token_budget`; allocate with preempt loop.
2. **WAITING** (if no preempt this step) — admit while `len(running) < max_num_seqs` and `token_budget > 0`; full prefix `lookup` (no preempt on waiting path).
3. Phase from request cursor: `req.is_prefill_chunk()` ↔ vLLM `num_computed_tokens < prompt_len`.

Engine knobs: `max_num_seqs`, `max_num_batched_tokens`, `block_size`, `enable_chunked_prefill` (default off).

## Policy hooks

### Eviction — `EvictionPolicy`

```python
class MyEviction(EvictionPolicy):
    def pick_victims(self, hbm, count, exclude) -> list[KVBlock]: ...

LookupPolicy(local_memory="npu-0:hbm", eviction_policy=MyEviction())
```

`plan()` computes slot deficit and calls `pick_victims`. Victims must be `RESIDENT` with `len(holders)==0` (`memory.can_evict_block`).

### Pull / compute — `LookupPolicy`

```python
LookupPolicy(
    local_memory="npu-1:hbm",
    pull_sources=["npu-0:hbm", "npu-0:ssd"],  # first resident source wins
)
```

Per block: local hit → skip; else pull from first source with `RESIDENT` block; else `compute`. `lookup → None` if a pull source has the block inflight (not implemented: early decode wait).

### Compute / transfer cost — `Engine` + `ForwardTask`

- One `ForwardTask` per engine batch (all compute blocks in the step).
- Prefill cost: `work_per_prefill_token × num_scheduled_tokens`
- Decode cost: `work_per_decode_req × num_decode_entries_with_compute`
- Pull: `work_per_transfer` on `BandwidthResource`; evict: `work_per_evict`

## Policy experiment readiness

Roughly **~70%** ready for evict/pull/compute policy sweeps. Not a full vLLM clone.

### Ready now

| Experiment | Hook |
|------------|------|
| Pull source order (1P1D) | `pull_sources` list order |
| Compute vs transfer cost | `work_per_prefill_token`, `work_per_decode_req`, `work_per_transfer` |
| Memory pressure + preempt | `Memory(size)`, deadlock scenario |
| Batch limits | `max_num_seqs`, `max_num_batched_tokens` |
| Custom eviction | `EvictionPolicy.pick_victims` |

### Usable with caveats

- **Eviction under load** — running requests pin blocks via `holders`; only unheld `RESIDENT` blocks evict. Pressure is often preemption-driven, not cache replacement. Fine for relative policy comparison if you understand the model.
- **Prefix hits** — implicit via `best_resident`; no ref-counted block pool like vLLM.

### Fix before trusting eviction/memory studies

1. **`free_request()` on completion** — today `release_request()` leaves `RESIDENT` KV after finish, inflating `used_size`.
2. **Metrics** — no built-in evict/pull/preempt counters yet (only `finish_time` from demos).

### Not implemented (distorts specific experiments)

| Gap | Affects |
|-----|---------|
| Chunked prefill in RUNNING | Long-prompt memory spikes |
| Early decode wait | Async PD |
| `num_computed_blocks` bumped in `apply_batch` not at schedule | Tight memory timing |
| Workload generator / metrics CLI | Large sweeps |

**Suggested path to ~85% policy-lab ready:** (1) free KV on complete, (2) run metrics, (3) chunked prefill if prefill+memory matters. Remaining gap vs vLLM is block-table/refcount fidelity.

## PD 1P1D

Memories use instance keys, not roles:

```python
memories = {
    "npu-0:hbm": Memory(size=100),
    "npu-1:hbm": Memory(size=100),
}
# spawn_decode={"npu-0": "npu-1"}  — decode request injected when prefill completes
```

Typical timeline (`work=1`, large HBM, `finish_time≈11` for 3 prefix + 3 pull + 5 decode blocks):

| Phase | Engine | What happens |
|-------|--------|----------------|
| Prefill | npu-0 | `schedule` → one `ForwardTask` for prefix blocks |
| Transfer | npu-1 | chained pull `LoadTask`s on bandwidth |
| Decode | npu-1 | one `ForwardTask` per step per output block |

Pull requires source `RESIDENT` (decode spawned after prefill completes). No shared producer-task prereqs.

## vLLM divergences (intentional simplifications)

| Area | vLLM | Simulator |
|------|------|-----------|
| Unit of work | Tokens | Blocks (`block_size` tokens/block) |
| Physical KV | Block table + refcounts | Content-keyed slots + holders |
| Scheduler output | `num_scheduled_tokens` per req | `BatchEntry.num_scheduled_tokens` |
| Preempt | `kv_cache_manager.free` | `free_request` + reset cursor |
| PD | KV connector, async match | Spawn on prefill complete; pull if resident |
| Eviction | LRU on physical blocks | Pluggable; default `FirstAvailableEviction` |

## Roadmap

1. `free_request()` on normal completion
2. Run metrics (evictions, pulls, computes, preemptions)
3. Chunked prefill in RUNNING path
4. Move cursor bump to schedule time
5. Early decode wait; workload config file
