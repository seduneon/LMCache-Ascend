"""Integration and unit tests for the KV cache simulator."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[2]
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


def _run_test_functions(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        getattr(mod, name)()
        print(f"{name} ok")


def run_unit_tests() -> None:
    _run_test_functions("simulator.tests.test_unit")


def run_critical_tests() -> None:
    _run_test_functions("simulator.tests.test_critical")


def run_stress_test() -> None:
    from simulator.tests.test_stress import run_stress_test as _run

    _run()


def run_stress_heavy_test() -> None:
    from simulator.tests.test_stress import run_stress_heavy_test as _run

    _run()


def run_stress_benchmark_cli() -> None:
    from simulator.tests.test_stress import run_stress_benchmark

    run_stress_benchmark()


def run_stress_seed_sweep_cli() -> None:
    from simulator.tests.test_stress import run_stress_seed_sweep

    run_stress_seed_sweep()


_ALL = {
    "unit": [run_unit_tests],
    "critical": [run_critical_tests],
    "stress": [run_stress_test],
    "stress-heavy": [run_stress_heavy_test],
    "stress-benchmark": [run_stress_benchmark_cli],
    "stress-seeds": [run_stress_seed_sweep_cli],
}


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if argv:
        key = argv[0]
        if key == "sweep":
            from simulator.bench.sweep import main as sweep_main

            sweep_main(argv[1:])
            return
        if key not in _ALL:
            raise SystemExit(f"unknown test group: {key!r} (try: {', '.join(sorted(_ALL))})")
        for test in _ALL[key]:
            test()
        return

    run_unit_tests()
    print()
    run_critical_tests()


if __name__ == "__main__":
    main()
