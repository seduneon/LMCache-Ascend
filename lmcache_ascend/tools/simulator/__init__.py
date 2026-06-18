"""KV cache policy-lab simulator (discrete-event, vLLM-shaped scheduler).

Lifecycle:
  arrive → admit (scheduler) → plan (lookup) → execute (BatchExecutor) → commit

Public batch API:
  Engine.try_schedule_and_execute(now) → BatchPlan | None
  Simulator applies completed batches via Engine.apply_plan

Layers:
  simulator    — event loop, PD dispatch
  engine       — orchestration; outcome hooks for placement/retention
  scheduler    — RUNNING / WAITING batching (Admit)
  effect_interpreter — mechanical BatchExecutor (Execute)
  plan         — BatchPlan, EntryPlan, StoreOp
  lookup / placement / retention — plugin surface (sweep.py)
  memory, tasks, resource, request — domain types
"""
