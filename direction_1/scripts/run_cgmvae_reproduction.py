from __future__ import annotations

import argparse
import sys
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
BENCHMARK_ROOT = CURRENT_FILE.parents[1]
PROJECT_ROOT = BENCHMARK_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from direction_1.src.pipeline import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the C-GMVAE paper-reproduction configuration.")
    parser.add_argument(
        "--config",
        type=Path,
        default=BENCHMARK_ROOT / "configs" / "reproductions" / "cgmvae_reproduction.json",
        help="Path to the C-GMVAE reproduction config file.",
    )
    args = parser.parse_args()
    summary = run_benchmark(args.config)
    print(f"C-GMVAE reproduction finished. Outputs written to: {summary['output_dir']}")
    print(f"Seeds: {summary['seed_count']} ({summary['seeds']})")
    print(f"Multilabel runs: {summary['multilabel_runs']}")
    print(f"Errors: {summary['error_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
