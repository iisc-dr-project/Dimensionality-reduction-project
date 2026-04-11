from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the JSON config and attach resolved project paths used by the pipeline."""
    config_file = Path(config_path).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))

    benchmark_root = _find_benchmark_root(config_file)
    project_root = benchmark_root.parent

    config["_config_path"] = str(config_file)
    config["_benchmark_root"] = str(benchmark_root)
    config["_project_root"] = str(project_root)
    return config


def _find_benchmark_root(config_file: Path) -> Path:
    for parent in config_file.parents:
        if (parent / "src").exists() and (parent / "scripts").exists():
            return parent
    raise ValueError(f"Could not infer benchmark root from config path: {config_file}")
