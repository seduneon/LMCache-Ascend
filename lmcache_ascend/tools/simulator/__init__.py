"""KV cache policy-lab simulator (discrete-event, vLLM-shaped scheduler).

Core modules:
  simulator   — event loop
  engine      — batch execution, placement hooks
  scheduler   — RUNNING / WAITING batching
  policies    — eviction, placement, retention, lookup
  workload    — synthetic PD workload generator
  sweep       — policy preset sweep + CSV export
  memory      — tier slot budgets
  tasks       — forward / pull / store / evict tasks
  resource    — compute and bandwidth queues
  request     — request state and metrics
  pd          — prefill/decode read-mode config
  sim_log     — optional verbose logging (``SIM_LOG=1``)
  sim_progress — progress bar for long stress runs

Policies in ``policies.py``: eviction, placement, retention, lookup.
Cost math in ``cost_model.py``; allocation SSOT is ``LookupPolicy.lookup()``.
"""
