"""Analyze structured simulator JSONL traces (stdlib only)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def analyze(events: list[dict]) -> dict:
    decisions = Counter()
    decision_by_tier = Counter()
    task_durations: dict[str, list[float]] = defaultdict(list)
    estimate_errors: list[float] = []
    tier_samples: list[dict] = []
    evictions = Counter()
    resource_busy: list[int] = []
    phase_times: dict[str, list[float]] = defaultdict(list)
    admitted_at: dict[tuple[str, str], float] = {}

    for event in events:
        kind = event.get("kind")
        if kind == "decision":
            decisions[event.get("chosen", "?")] += 1
            chosen = event.get("chosen", "")
            if chosen.startswith("pull:"):
                decision_by_tier[chosen.split(":", 1)[1]] += 1
        elif kind == "task_end":
            task_kind = event.get("task_kind", "?")
            duration = float(event.get("duration", 0.0))
            task_durations[task_kind].append(duration)
            err = event.get("estimate_error")
            if err is not None:
                estimate_errors.append(float(err))
        elif kind == "task_start":
            busy = event.get("resource_busy")
            if busy is not None:
                resource_busy.append(int(busy))
        elif kind == "request_phase":
            phase = event.get("phase", "")
            req_key = (event.get("engine_id", ""), event.get("req_id", ""))
            t = float(event.get("t", 0.0))
            if phase.endswith("_admitted"):
                admitted_at[req_key] = t
            elif phase.endswith("_complete") and req_key in admitted_at:
                phase_times[phase.rsplit("_", 1)[0]].append(t - admitted_at.pop(req_key))
        elif kind == "tier_sample":
            tier_samples.append(event)
        elif kind == "evict":
            evictions[event.get("tier", "?")] += 1

    occupancy_peak: dict[str, int] = defaultdict(int)
    for sample in tier_samples:
        for tier, stats in sample.get("tiers", {}).items():
            occupancy_peak[tier] = max(occupancy_peak[tier], int(stats.get("used", 0)))

    task_means = {k: mean(v) if v else 0.0 for k, v in task_durations.items()}
    latency_p50 = percentile(task_durations.get("forward", []), 50)
    latency_p99 = percentile(task_durations.get("forward", []), 99)
    busy_fraction = (
        mean(1 if b > 0 else 0 for b in resource_busy) if resource_busy else 0.0
    )
    return {
        "event_count": len(events),
        "decisions": dict(decisions),
        "pull_by_tier": dict(decision_by_tier),
        "task_duration_mean": task_means,
        "forward_latency_p50": latency_p50,
        "forward_latency_p99": latency_p99,
        "request_latency_p50": percentile(
            [v for vals in phase_times.values() for v in vals], 50
        ),
        "request_latency_p99": percentile(
            [v for vals in phase_times.values() for v in vals], 99
        ),
        "resource_busy_fraction": busy_fraction,
        "estimate_error_mean": mean(estimate_errors) if estimate_errors else 0.0,
        "estimate_error_abs_mean": (
            mean(abs(e) for e in estimate_errors) if estimate_errors else 0.0
        ),
        "estimate_error_p99": percentile([abs(e) for e in estimate_errors], 99),
        "tier_occupancy_peak": dict(occupancy_peak),
        "evictions": dict(evictions),
    }


def print_report(summary: dict) -> None:
    print(f"events: {summary['event_count']}")
    print("\n=== decisions ===")
    for key, count in sorted(summary["decisions"].items()):
        print(f"  {key}: {count}")
    if summary["pull_by_tier"]:
        print("\n=== pulls by tier ===")
        for tier, count in sorted(summary["pull_by_tier"].items()):
            print(f"  {tier}: {count}")
    print("\n=== task duration mean (s) ===")
    for kind, dur in sorted(summary["task_duration_mean"].items()):
        print(f"  {kind}: {dur:.4f}")
    print("\n=== forward latency percentiles (s) ===")
    print(f"  p50: {summary['forward_latency_p50']:.4f}")
    print(f"  p99: {summary['forward_latency_p99']:.4f}")
    print("\n=== request latency percentiles (s) ===")
    print(f"  p50: {summary['request_latency_p50']:.4f}")
    print(f"  p99: {summary['request_latency_p99']:.4f}")
    print("\n=== resource busy fraction ===")
    print(f"  {summary['resource_busy_fraction']:.3f}")
    print("\n=== pull estimate error ===")
    print(f"  mean: {summary['estimate_error_mean']:.4f}")
    print(f"  abs_mean: {summary['estimate_error_abs_mean']:.4f}")
    print(f"  abs_p99: {summary['estimate_error_p99']:.4f}")
    if summary["tier_occupancy_peak"]:
        print("\n=== tier occupancy peak ===")
        for tier, used in sorted(summary["tier_occupancy_peak"].items()):
            print(f"  {tier}: {used}")
    if summary["evictions"]:
        print("\n=== evictions ===")
        for tier, count in sorted(summary["evictions"].items()):
            print(f"  {tier}: {count}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze simulator JSONL trace")
    parser.add_argument("trace", type=Path, help="Path to JSONL trace file")
    args = parser.parse_args(argv)
    if not args.trace.is_file():
        raise SystemExit(f"trace not found: {args.trace}")
    summary = analyze(load_events(args.trace))
    print_report(summary)


if __name__ == "__main__":
    main()
