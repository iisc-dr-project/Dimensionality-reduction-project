from __future__ import annotations

import argparse
import sys
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
BENCHMARK_ROOT = CURRENT_FILE.parents[1]
PROJECT_ROOT = BENCHMARK_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from direction_1.src.reproductions.random_projection import run_random_projection_reproduction


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the random projection paper-reproduction experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=BENCHMARK_ROOT / "configs" / "reproductions" / "random_projection_reproduction.json",
        help="Path to the random projection reproduction config file.",
    )
    args = parser.parse_args()
    summary = run_random_projection_reproduction(args.config)
    print(f"Random projection reproduction finished. Outputs written to: {summary['output_dir']}")
    print(f"Seeds: {summary['seed_count']} ({summary['seeds']})")
    print(f"Runs: {summary['run_count']}")
    print(f"Aggregated rows: {summary['aggregated_run_count']}")
    print(f"Errors: {summary['error_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
