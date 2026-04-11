from __future__ import annotations

import argparse
import shutil
from pathlib import Path


FILENAMES = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy local MNIST IDX files into the benchmark data folder.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
        help="Directory that already contains the MNIST IDX files.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw" / "mnist",
        help="Destination directory inside the benchmark project.",
    )
    args = parser.parse_args()

    args.target_root.mkdir(parents=True, exist_ok=True)

    for filename in FILENAMES:
        source = args.source_root / filename
        destination = args.target_root / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing source file: {source}")
        shutil.copy2(source, destination)
        print(f"Copied {source} -> {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
