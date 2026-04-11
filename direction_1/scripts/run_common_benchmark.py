from __future__ import annotations

import sys
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
BENCHMARK_ROOT = CURRENT_FILE.parents[1]
PROJECT_ROOT = BENCHMARK_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from direction_1.src.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
