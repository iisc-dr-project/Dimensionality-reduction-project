from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Direction 1 common benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "common" / "benchmark_config.json",
        help="Path to the benchmark JSON config file.",
    )
    args = parser.parse_args()
    summary = run_benchmark(args.config)
    print(f"Benchmark finished. Outputs written to: {summary['output_dir']}")
    print(f"Seeds: {summary['seed_count']} ({summary['seeds']})")
    print(f"Visualization runs: {summary['visualization_runs']}")
    print(f"Downstream runs: {summary['downstream_runs']}")
    print(f"Multilabel runs: {summary['multilabel_runs']}")
    print(f"Errors: {summary['error_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
