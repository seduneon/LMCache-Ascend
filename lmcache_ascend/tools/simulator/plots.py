"""Optional matplotlib plots for simulator traces."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analyze import analyze, load_events


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plots; install with: pip install matplotlib"
        ) from exc
    return plt


def plot_tier_occupancy(events: list[dict], out: Path) -> None:
    plt = _require_matplotlib()
    series: dict[str, list[tuple[float, int]]] = {}
    for event in events:
        if event.get("kind") != "tier_sample":
            continue
        t = float(event["t"])
        for tier, stats in event.get("tiers", {}).items():
            series.setdefault(tier, []).append((t, int(stats.get("used", 0))))
    if not series:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for tier, points in sorted(series.items()):
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], label=tier)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("slots used")
    ax.set_title("Tier occupancy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_decision_mix(summary: dict, out: Path) -> None:
    plt = _require_matplotlib()
    decisions = summary.get("decisions", {})
    if not decisions:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(decisions.keys())
    values = [decisions[k] for k in labels]
    ax.bar(labels, values)
    ax.set_title("Decision mix")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_latency_cdf(events: list[dict], out: Path) -> None:
    plt = _require_matplotlib()
    durations = [
        float(event["duration"])
        for event in events
        if event.get("kind") == "task_end" and event.get("task_kind") == "forward"
    ]
    if not durations:
        return
    ordered = sorted(durations)
    ys = [(i + 1) / len(ordered) for i in range(len(ordered))]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ordered, ys)
    ax.set_xlabel("forward duration (s)")
    ax.set_ylabel("CDF")
    ax.set_title("Forward latency CDF")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_cost_scatter(events: list[dict], out: Path) -> None:
    plt = _require_matplotlib()
    predicted: list[float] = []
    actual: list[float] = []
    for event in events:
        if event.get("kind") != "task_end" or event.get("task_kind") != "pull":
            continue
        err = event.get("estimate_error")
        duration = float(event.get("duration", 0.0))
        if err is None:
            continue
        predicted.append(duration - float(err))
        actual.append(duration)
    if not predicted:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(predicted, actual, alpha=0.5, s=12)
    lim = max(max(predicted), max(actual))
    ax.plot([0, lim], [0, lim], linestyle="--", color="gray")
    ax.set_xlabel("predicted pull duration (s)")
    ax.set_ylabel("actual pull duration (s)")
    ax.set_title("Pull cost: predicted vs actual")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_bandwidth_utilization(events: list[dict], out: Path) -> None:
    plt = _require_matplotlib()
    points: list[tuple[float, int]] = []
    for event in events:
        if event.get("kind") != "task_start":
            continue
        busy = event.get("resource_busy")
        if busy is None:
            continue
        points.append((float(event["t"]), int(busy)))
    if not points:
        return
    points.sort()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([p[0] for p in points], [p[1] for p in points])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("resource busy count")
    ax.set_title("Bandwidth utilization proxy")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot simulator trace charts")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    events = load_events(args.trace)
    summary = analyze(events)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_tier_occupancy(events, args.out_dir / "tier_occupancy.png")
    plot_decision_mix(summary, args.out_dir / "decision_mix.png")
    plot_latency_cdf(events, args.out_dir / "latency_cdf.png")
    plot_cost_scatter(events, args.out_dir / "cost_scatter.png")
    plot_bandwidth_utilization(events, args.out_dir / "bandwidth_utilization.png")
    print(f"wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
