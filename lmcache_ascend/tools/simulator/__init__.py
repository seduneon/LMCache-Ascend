"""KV cache policy-lab simulator (discrete-event, vLLM-shaped scheduler).

Lifecycle:
  arrive → admit (scheduler) → plan (lookup) → execute (BatchRunner) → commit

Public batch API:
  Engine.try_schedule_and_execute(now) → BatchPlan | None
  Simulator applies completed batches via Engine.apply_plan

Layers:
  simulator    — event loop, PD dispatch
  engine       — orchestration; placement/retention hooks on task completion
  scheduler    — RUNNING / WAITING batching (Admit)
  execute      — BatchRunner: reservations, tasks, effect callbacks
  plan         — BatchPlan, EntryPlan, StoreOp
  lookup / placement / retention — plugin surface (sweep.py)
  memory, tasks, resource, request — domain types
"""
