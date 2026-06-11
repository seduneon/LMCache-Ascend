# KV Cache Simulator

Discrete-event simulator for KV cache admission, eviction, compute, and PD transfer.

```bash
cd lmcache_ascend/tools/simulator
python3.12 simulator.py            # PD demo + deadlock test
python3.12 simulator.py deadlock   # deadlock / preemption test only
```

## PD disaggregation — 1P1D event mapping

Minimal **1 prefill NPU + 1 decode NPU**, one logical request `req_id="r1"`, prefix blocks `["a","b","c"]`.

### Naming (many NPUs / HBMs)

Do **not** name memories by role (`hbm_p`, `hbm_d`). Use stable **instance / tier keys**:

```python
memories = {
    "npu-0:hbm": Memory(size=100, name="npu-0:hbm"),  # prefill in 1P1D
    "npu-1:hbm": Memory(size=100, name="npu-1:hbm"),  # decode in 1P1D
}

Engine(
    engine_id="npu-0",
    role=RequestPD.PREFILL,
    local_memory="npu-0:hbm",   # where this engine reserves/loads
    memories=memories,            # full map; lookup may read other tiers
)
```

- **`engine_id`** — NPU instance (`npu-0`, `npu-1`, …). Role (prefill/decode) lives on `Engine`, not on memory names.
- **`local_memory`** — this engine's primary KV tier (usually `"{engine_id}:hbm"`).
- **`memories` keys** — any tier on any NPU (`npu-2:ssd`, `npu-0:cpu`, …).
- **Pull source** — `("pull", "npu-0:hbm")`, not `("pull", "prefill")`. The source is a **memory key**, not a role.

1P1D is just `npu-0` = producer, `npu-1` = consumer. XpYd is more engines with the same keys.

### Components to add

| New piece | vLLM analogue |
|-----------|----------------|
| `Engine(engine_id, role, local_memory)` | vLLM instance / `kv_producer` or `kv_consumer` |
| `memories: dict[str, Memory]` | per-NPU KV pools (and later SSD/CPU tiers) |
| `LookupPolicy` | local hit skip; optional `pull_sources`; else compute |
| `LoadTask` on `BandwidthResource` (pull) | `start_load_kv` |
| Same `req_id` on both engines | `request_id` in KV connector metadata |
| `Simulator` loops all engines | global clock |

### Example workload

```
# Same req_id (vLLM: request_id links producer send ↔ consumer recv)
npu-0: Request(req_id="r1", pd=PREFILL, arrival=0,   blocks=[a,b,c])
npu-1: Request(req_id="r1", pd=DECODE,  arrival=3*, blocks=[a,b,c])  (* spawned when prefill completes)
```

Decode may also arrive early and sit in `WAITING` until blocks are `RESIDENT` on `npu-0:hbm` (like `get_num_new_matched_tokens` returning `None`).

### vLLM step → simulator events

Assume `work=1`, `bandwidth` speed `1`, no eviction, HBM large enough.

| Time | vLLM step | Simulator events |
|------|-----------|------------------|
| **0** | Client → proxy → prefill | `npu-0.release_arrivals(0)`: `r1` → `WAITING` |
| **0** | Prefill admits | `npu-0.admit()`: lookup → `{blocks:{a,b,c:compute}}`; reserve on `npu-0:hbm`; `Load(a)→Load(b)→Load(c)` |
| **0–3** | Prefill forward (3 blocks) | Compute advances; at **t=3** a,b,c `RESIDENT` on `npu-0:hbm` |
| **3** | Prefill `request_finished` | `r1` prefill → `COMPLETE`; spawn `r1` decode on `npu-1` at `now=3` |
| **3** | Decode engine sees request | `npu-1.release_arrivals(3)`: `r1` → `WAITING` |
| **3** | `get_num_new_matched_tokens` | `npu-1.admit()`: a,b,c on `npu-0:hbm` → `{blocks:{a,b,c:("pull","npu-0:hbm")}}`; reserve on `npu-1:hbm`; 3× pull `LoadTask` |
| **3–6** | `start_load_kv` (pull) | `LoadTask` on `BandwidthResource`; dst blocks `RESIDENT` on `npu-1:hbm` |
| **6** | Decode KV ready | Transfers done; `r1` decode tasks complete → `COMPLETE` |

**End-to-end finish ≈ 6 + max_output_blocks** (3 prefill compute + 3 transfer + 1 compute per generated block).

### Task DAG (req_id r1)

```
npu-0 (npu-0:hbm):
  Load(a) → Load(b) → Load(c)

npu-1 (npu-1:hbm), after t=3:
  Load(a, bandwidth) → Load(b) → Load(c)   # pull from npu-0:hbm
```

Cross-engine prereq (if decode admitted before prefill done): each pull `LoadTask` prereqs on prefill `LoadTask` for same hash.

### Lookup rules

`LookupPolicy(local_memory, pull_sources=[])` — compute-only misses. With `pull_sources` (e.g. `["npu-0:hbm"]`):

```python
if local hit or loading:                 skip
elif memories[src].best_resident(h):     ("pull", src)   # first matching src
elif memories[src].inflight_incoming(h): None  # wait
else:                                    "compute"
```

### Admission / spawn

**Option B (recommended):** prefill `COMPLETE` → inject decode request with the **same `req_id`**:

```python
def on_prefill_complete(req_id, finish_time, blocks):
    decode_pending.append(Request(req_id, finish_time, blocks, pd=DECODE, ...))
```

**Early decode (vLLM async):** decode in `WAITING` at t=0; `lookup → None` until prefill loads finish; retry each `step()`.

Pull uses the same `LoadTask` as compute on `BandwidthResource`. Schedule requires the source block to be `RESIDENT` (no cross-request task prereq on the producer).

### Simulator loop (vLLM batch scheduling)

Each `Simulator.step()`:

```python
def step(self):
    for eng in self.engines:
        eng.release_arrivals(self.now)
    for eng in self.engines:
        batch = eng.schedule()          # Scheduler: RUNNING → WAITING, preempt at allocate
        eng.execute_batch(batch)        # reserve + enqueue tasks for this batch only
    drain until all batch tasks complete
    for eng in self.engines:
        eng.apply_batch(batch)          # advance num_computed_blocks, finish requests
    spawn decode on prefill complete
```

- **`scheduler.py`** — queues, `Batch` / `BatchEntry`, preempt while building the batch (victims not in the current batch).
- **`engine.py`** — `execute_batch()` + `apply_batch()` only; no mid-step admit/schedule loops.
- **No `cancel_tasks` on preempt** — a step drains fully before the next `schedule()`; preemption only frees KV and resets request state.
- **Pull prereqs** — no shared `src_block.task` dependency; schedule requires source `RESIDENT` (decode spawned after prefill completes).

### Implemented (PD v1)

- `BandwidthResource`, pull via `LoadTask`, `LookupPolicy(pull_sources=...)`
- `Engine(engine_id, local_memory, ...)`, pull via `("pull", memory_key)`
- Multi-engine `Simulator`, `spawn_decode={"npu-0": "npu-1"}`

### Scheduling (vLLM-aligned)

`Scheduler.schedule()` per engine, once per sim step:

1. **RUNNING** — each decode request gets 1 output block; `allocate` with preempt loop (FCFS tail, skip requests already in this batch).
2. **WAITING** — admit head while block budget remains; full prefix `lookup` (no preempt on waiting path).
3. **`execute_batch`** — reserve + build task DAG for all entries; sim drains this batch before the next step.
4. **`apply_batch`** — advance `num_computed_blocks`, append output hashes, finish requests.

`num_computed_blocks` is the single cursor. Output block IDs are assigned when scheduled (`blk:{req_id}:{index}`) and appended to `block_hashes` when the batch completes.

## Known gaps and divergences from vLLM

### Critical problems (can cause wrong or fragile behavior)

**No batch / concurrency limits (`max_num_seqs`, `token_budget`)**  
vLLM caps how many sequences run and how many tokens are scheduled per step. `Scheduler.schedule()` can still place many running decode requests (one block each) in one batch with no global token cap.

**Prefill is not in the RUNNING schedule loop**  
Prefill is admitted from `WAITING` as a full-prefix batch entry. vLLM schedules running prefills incrementally (chunked prefill) with the same allocate/preempt loop.

**Preemption only while scheduling RUNNING decode blocks**  
Preempt runs in `Scheduler._allocate_running()` when decode output allocation fails. The `WAITING` path uses `lookup` without preempt (vLLM-aligned). A waiting request can starve until running requests free slots or are preempted.

**Completion vs schedule timing**  
vLLM advances `num_computed_tokens` at schedule time (`_update_after_schedule`). The simulator advances `num_computed_blocks` in `apply_batch()` after the batch drains. That still shifts memory lifetime and PD spawn timing relative to vLLM.

**Memory model: holders block eviction, not a block table**  
Eviction only picks `RESIDENT` blocks with `len(holders) == 0`. Running requests pin all their blocks via holders, so pressure is entirely preemption-driven. vLLM’s paged block pool and refcounts behave differently (prefix cache blocks, shared physical slots, connector-owned blocks).

**Completed requests leave KV resident**  
`_finish_request` calls `release_request()`, which only removes unheld `RESERVED` blocks. `RESIDENT` KV stays in memory (holders cleared). vLLM frees via `kv_cache_manager.free()` on finish/preempt. This can inflate `used_size` and change eviction/preemption dynamics.

### Major modeling divergences (usually intentional)

| Area | vLLM | Simulator |
|------|------|-----------|
| Unit of work | Tokens (variable chunk sizes) | 1 block = 1 forward |
| Physical KV | Block table + pool | Content-keyed slot budget |
| Prefix cache | `PrefixCacheBlock` / hash chain | Implicit via `best_resident` hit skip |
| Output blocks | Placeholders, spec decode, async discard | `blk:{req}:{idx}` assigned at schedule |
| Waiting admission | `allocate_slots` fails → stop | `lookup → None` (remote wait or memory) |
| Preempt victim | FCFS `running.pop()` or priority | FCFS tail of `running` |
| Preempt reset | `num_computed_tokens = 0`, `PREEMPTED` | `num_computed_blocks = 0`, trim prefix hashes |
| PD | Connector, async KV load, matched tokens | Spawn decode on prefill complete; pull when src `RESIDENT` |
| Early decode | Wait in queue until remote KV ready | Not implemented |
| Eviction policy | LRU etc. on physical blocks | `FirstAvailableEviction` on unheld resident copies |
| Resources | Per-worker GPU | Shared global `TaskPool` + one compute/bandwidth resource |

### PD-specific gaps

- **No early decode** — decode cannot sit in `WAITING` while prefill is still computing (vLLM `get_num_new_matched_tokens → None`).
- **No producer eviction after transfer** — prefill blocks stay on the producer after pull.
- **Spawn timing** — decode is injected only when prefill **completes**, not when KV is merely resident/transferable.
- **Cross-engine failure modes** — source eviction or source preempt during an in-flight pull is not modeled.

### What is reasonably aligned

- Batch scheduling: one `schedule()` per engine per step, drain batch before next step
- RUNNING before WAITING inside `Scheduler.schedule()`
- Reserve-before-forward (`append_reserved` → `LoadTask`)
- vLLM-style preempt at running allocate failure: `free_request`, reset progress, prepend to `waiting`
- Block-level output hash assigned when scheduled, appended in `apply_batch`

### Task DAG (`tasks.py`)

Readiness is derived from prereq **status** (`_is_ready`), not a `needs` counter. `cancel_tasks()` cascades only to dependents with the same `req_id`; cross-request dependents of a cancelled prereq stay pending (poisoned). Preempt does not call `cancel_tasks` — each step drains before the next schedule.

### Planned improvements (highest impact first)

1. `max_running_reqs` + per-step schedule budget (even “1 block total per engine per step”)
2. Unified allocate/preempt for chunked prefill in the RUNNING path
3. Move `num_computed_blocks` bump to schedule time, not `apply_batch`
4. `free_request()` on normal completion, not just `release_request()`

### Later

- Early decode wait (decode arrives before prefill done)
- Prefill KV eviction after transfer
- Multiple pull sources / tiers per decode policy
