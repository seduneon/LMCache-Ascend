# KV Cache Simulator

Discrete-event simulator for KV cache admission, eviction, compute, and PD transfer.

```bash
cd lmcache_ascend/tools/simulator
python3.12 simulator.py   # 1P1D PD demo (finish_time=6)
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

**End-to-end finish ≈ 6** (3 prefill compute + 3 transfer). v1 can skip decode-token compute and treat decode as transfer-only.

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

Pull uses the same `LoadTask` as compute; engine passes `BandwidthResource` and wires prereqs on the source `LoadTask`.

### Simulator loop (multi-engine)

```python
def step(self):
    for eng in self.engines:
        eng.release_arrivals(self.now)
    for eng in self.engines:
        while eng.admit(): pass
    self.pool.start_ready(self.now)
    # advance time to min(pool.next(), next arrival across engines)
    ...
    for eng in self.engines:
        while eng.admit(): pass
    for eng in self.engines:
        eng.check_completions(on_complete=spawn_decode_if_prefill)
```

### Implemented (PD v1)

- `BandwidthResource`, pull via `LoadTask`, `LookupPolicy(pull_sources=...)`
- `Engine(engine_id, local_memory, ...)`, pull via `("pull", memory_key)`
- Multi-engine `Simulator`, `spawn_decode={"npu-0": "npu-1"}`

### Later

- Early decode wait (decode arrives before prefill done)
- Decode token compute after KV pull
- Prefill KV eviction after transfer
- Multiple pull sources / tiers per decode policy
