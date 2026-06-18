"""KV cache policy-lab simulator (discrete-event, vLLM-shaped scheduler).

Lifecycle:
  arrive → admit (scheduler) → plan (schedule) → enrich (effects) → execute → commit

Public batch API:
  Engine.try_schedule_and_execute(now) → BatchPlan | None
  Simulator applies completed batches via Engine.apply_plan

Layers:
  simulator    — event loop, PD dispatch
  engine       — orchestration; ``EnginePolicies`` bundle
  scheduler    — RUNNING / WAITING batching (admit)
  schedule     — block resolution + HBM eviction at admit time
  effects      — placement, retention, plan-time store/spill expansion
  policies     — ``EnginePolicies`` factories + ``enrich_entry_plan``
  presets      — data-driven ``PRESETS`` + ``build_pd_engines``
  topology     — tier memory layout for PD experiments
  execute      — ``BatchRunner``: reservations, tasks, effect callbacks
  plan         — ``BatchPlan``, ``EntryPlan``, ``StoreOp``
  lookup       — compatibility shims over ``SchedulePolicy``
  placement / retention / eviction — effect implementations
  memory, tasks, resource, request — domain types
"""
