"""KV cache policy-lab simulator (discrete-event, vLLM-shaped scheduler).

Pipeline (top → bottom):
  simulator   — event loop, PD event dispatch
  engine      — schedule_batch / make_work / execute_work / apply_work
  scheduler   — RUNNING / WAITING batching → ScheduleResult
  kv_controller — schedule-time KV planning facade
  effect_interpreter — BatchExecutor + ResidentEffects
  plan        — ScheduleResult, BatchWork, EntryPlan, SimContext
  eviction    — EvictionPolicy implementations
  tier_allocator — shared downstream tier slot acquisition
  policies    — placement, retention, lookup implementations
  events      — DecodeSpawn, KvRelease (cross-engine dataflow)
  workload    — synthetic PD workload generator
  sweep       — policy preset sweep + CSV export
  memory      — tier slot budgets
  tasks       — forward / pull / store / evict tasks
  resource    — compute and bandwidth queues
  request     — request state and metrics
  pd          — prefill/decode read-mode config
  sim_log     — optional verbose logging (``SIM_LOG=1``)
  sim_progress — progress bar for long stress runs

Allocation SSOT: ``KVController.plan_blocks()`` → ``EntryPlan`` inside ``ScheduleResult``.
"""