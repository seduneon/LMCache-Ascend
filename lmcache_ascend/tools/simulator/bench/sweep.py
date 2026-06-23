"""Policy sweep harness: fixed workloads, aggregated metrics, CSV output."""

from __future__ import annotations

from .cli import list_presets, main
from .presets import PRESETS
from .metrics import (
    SWEEP_CSV_FIELDS,
    SWEEP_COMPARE_FIELDS,
    SWEEP_COMPARE_NUMERIC,
    SweepRow,
    _aggregate_metrics,
    _local_tier_end_state_ok,
)
from .sweep_config import SimConfig, SweepConfig
from .sweep_run import build_engines, run_sweep, run_sweep_case

__all__ = [
    "PRESETS",
    "SimConfig",
    "SweepConfig",
    "SweepRow",
    "SWEEP_CSV_FIELDS",
    "SWEEP_COMPARE_FIELDS",
    "SWEEP_COMPARE_NUMERIC",
    "build_engines",
    "run_sweep_case",
    "run_sweep",
    "main",
    "list_presets",
    "_aggregate_metrics",
    "_local_tier_end_state_ok",
]

if __name__ == "__main__":
    main()
