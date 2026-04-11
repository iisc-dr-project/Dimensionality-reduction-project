from __future__ import annotations

import sys
import unittest
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BENCHMARK_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from direction_1.src.config import load_config


class ConfigLoaderTest(unittest.TestCase):
    def test_example_config_loads(self) -> None:
        config_path = BENCHMARK_ROOT / "configs" / "common" / "benchmark_config.json"
        config = load_config(config_path)
        self.assertEqual(config["seed"], 24529)
        self.assertGreaterEqual(len(config["datasets"]), 1)
        self.assertIn("_benchmark_root", config)

    def test_nested_reproduction_config_loads(self) -> None:
        config_path = BENCHMARK_ROOT / "configs" / "reproductions" / "random_projection_reproduction.json"
        config = load_config(config_path)
        self.assertEqual(config["seed"], 24529)
        self.assertEqual(Path(config["_benchmark_root"]), BENCHMARK_ROOT)


if __name__ == "__main__":
    unittest.main()
