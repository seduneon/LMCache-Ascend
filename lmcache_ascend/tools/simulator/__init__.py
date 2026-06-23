"""KV cache policy-lab simulator (discrete-event, vLLM-shaped scheduler).

Lifecycle:
  arrive → admit (scheduler) → plan (schedule) → enrich (effects) → execute → commit

Package layout:
  core          — memory, request, resource, kv_content
  model         — tier graph, capacity, topology layout
  policy        — eviction, retention, placement, read-path, schedule, effects
  runtime       — plan, tasks, scheduler, execute, engine, simulator
  bench         — presets, sweep, workload, traces
  observability — metrics, tracing, logging, analysis
  tests         — unit, critical, stress runners
"""
