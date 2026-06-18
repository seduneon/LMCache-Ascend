"""KV cache policy-lab simulator (discrete-event, vLLM-shaped scheduler).

Lifecycle:
  arrive → admit (scheduler) → plan (kv_controller + lookup) → execute (BatchExecutor) → commit

Public batch API:
  Engine.try_schedule_and_execute(now) → BatchPlan | None
  Simulator applies completed batches via Engine.apply_plan

Layers:
  simulator    — event loop, PD dispatch
  engine       — thin orchestration; outcome hooks for placement/retention
  scheduler    — RUNNING / WAITING batching (Admit)
  kv_controller — schedule-time lookup + EntryPlan enrichment (Plan)
  effect_interpreter — mechanical BatchExecutor (Execute)
  plan         — BatchPlan, EntryPlan, StoreOp, SimContext
  lookup / placement / retention — plugin surface (sweep.py)
  memory, tasks, resource, request — domain types
"""
