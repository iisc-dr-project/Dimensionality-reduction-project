from __future__ import annotations

import csv
import importlib
import json
import random
from pathlib import Path
from typing import Any


def import_required(module_name: str) -> Any:
    """Import a required runtime dependency."""
    return importlib.import_module(module_name)


def ensure_directory(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(path: str | Path, payload: Any) -> None:
    path_obj = Path(path)
    ensure_directory(path_obj.parent)
    path_obj.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path_obj = Path(path)
    ensure_directory(path_obj.parent)
    if not rows:
        path_obj.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path_obj.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np = importlib.import_module("numpy")
    np.random.seed(seed)

    torch = importlib.import_module("torch")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_dense_array(data: Any) -> Any:
    """Convert sparse-like matrices to dense arrays when a downstream method requires it."""
    if hasattr(data, "toarray"):
        return data.toarray()
    return data


def resolve_path(anchor: Path, raw_path: str) -> Path:
    path_obj = Path(raw_path)
    if path_obj.is_absolute():
        return path_obj
    return (anchor / path_obj).resolve()
