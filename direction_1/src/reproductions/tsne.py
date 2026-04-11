from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..config import load_config
from ..datasets import load_dataset
from ..utils import ensure_directory, import_required, save_csv, save_json, seed_everything, to_dense_array


def run_tsne_reproduction(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    benchmark_root = Path(config["_benchmark_root"])
    output_dir = ensure_directory(benchmark_root / config["output_dir"])
    plots_dir = ensure_directory(output_dir / "plots")
    seeds = [int(seed) for seed in config.get("seeds", [config.get("seed", 0)])]

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for experiment in config.get("experiments", []):
        dataset_spec = experiment["dataset"]
        for seed in seeds:
            try:
                seed_everything(seed)
                dataset = load_dataset(dataset_spec, benchmark_root, seed)
                X = _prepare_pca_input(dataset["X"], int(experiment.get("pca_components", 30)), seed)
                y = dataset["y"]
            except ImportError:
                raise
            except Exception as exc:
                errors.append({"experiment": experiment["name"], "seed": seed, "stage": "dataset", "error": str(exc)})
                continue

            for method_spec in experiment.get("methods", []):
                method_name = method_spec["name"]
                method_seed = _stable_seed(seed, experiment["name"], method_name)
                start_time = time.perf_counter()
                try:
                    seed_everything(method_seed)
                    embedding = _run_embedding_method(X, method_spec, method_seed)
                    elapsed = round(time.perf_counter() - start_time, 4)
                    _save_scatter_plot(
                        plots_dir=plots_dir,
                        experiment_name=experiment["name"],
                        dataset_name=dataset["name"],
                        method_name=method_name,
                        seed=seed,
                        embedding=embedding,
                        labels=y,
                    )
                    rows.append(
                        {
                            "experiment": experiment["name"],
                            "dataset": dataset["name"],
                            "method": method_name,
                            "seed": seed,
                            "pca_components": int(experiment.get("pca_components", 30)),
                            "elapsed_seconds": elapsed,
                            "n_samples": int(len(y)),
                            "n_classes": int(len(set(y.tolist()))),
                        }
                    )
                except ImportError:
                    raise
                except Exception as exc:
                    errors.append(
                        {
                            "experiment": experiment["name"],
                            "dataset": dataset["name"],
                            "method": method_name,
                            "seed": seed,
                            "stage": "embedding",
                            "error": str(exc),
                        }
                    )

    save_csv(output_dir / "embedding_runs.csv", rows)
    save_csv(output_dir / "embedding_runs_aggregated.csv", _aggregate_rows(rows, ["experiment", "dataset", "method"]))
    save_json(output_dir / "errors.json", errors)

    summary = {
        "seeds": seeds,
        "seed_count": len(seeds),
        "run_count": len(rows),
        "error_count": len(errors),
        "output_dir": str(output_dir),
        "note": "t-SNE settings approximate the paper using scikit-learn; the exact momentum schedule is not exposed by the library.",
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def _prepare_pca_input(X: Any, pca_components: int, seed: int) -> Any:
    np = import_required("numpy")
    sklearn_decomposition = import_required("sklearn.decomposition")

    X_dense = np.asarray(to_dense_array(X), dtype=np.float32)
    reducer = sklearn_decomposition.PCA(n_components=pca_components, random_state=seed)
    return reducer.fit_transform(X_dense)


def _run_embedding_method(X: Any, method_spec: dict[str, Any], seed: int) -> Any:
    base_method = method_spec["base_method"]
    if base_method == "tsne":
        return _run_tsne(X, method_spec, seed)
    if base_method == "isomap":
        sklearn_manifold = import_required("sklearn.manifold")
        model = sklearn_manifold.Isomap(n_neighbors=int(method_spec.get("params", {}).get("n_neighbors", 12)), n_components=2)
        return model.fit_transform(X)
    if base_method == "lle":
        sklearn_manifold = import_required("sklearn.manifold")
        model = sklearn_manifold.LocallyLinearEmbedding(
            n_neighbors=int(method_spec.get("params", {}).get("n_neighbors", 12)),
            n_components=2,
            method=method_spec.get("params", {}).get("method", "standard"),
            random_state=seed,
        )
        return model.fit_transform(X)
    if base_method == "sammon":
        return _run_sammon_mapping(X, method_spec, seed)
    raise ValueError(f"Unsupported t-SNE reproduction method: {base_method}")


def _run_tsne(X: Any, method_spec: dict[str, Any], seed: int) -> Any:
    sklearn_manifold = import_required("sklearn.manifold")
    params = dict(method_spec.get("params", {}))
    params.setdefault("n_components", 2)
    params.setdefault("perplexity", 40)
    params.setdefault("max_iter", 1000)
    params.setdefault("early_exaggeration", 4.0)
    params.setdefault("learning_rate", 100.0)
    params.setdefault("init", "pca")
    params.setdefault("random_state", seed)
    return sklearn_manifold.TSNE(**params).fit_transform(X)


def _run_sammon_mapping(X: Any, method_spec: dict[str, Any], seed: int) -> Any:
    np = import_required("numpy")
    sklearn_decomposition = import_required("sklearn.decomposition")
    scipy_spatial_distance = import_required("scipy.spatial.distance")

    params = method_spec.get("params", {})
    max_iter = int(params.get("max_iter", 500))
    learning_rate = float(params.get("learning_rate", 0.3))
    epsilon = 1e-9

    X_array = np.asarray(X, dtype=np.float32)
    init = sklearn_decomposition.PCA(n_components=2, random_state=seed).fit_transform(X_array)
    y = init.astype(np.float32, copy=True)

    high_dist = scipy_spatial_distance.squareform(scipy_spatial_distance.pdist(X_array, metric="euclidean")).astype(np.float32)
    high_dist = np.maximum(high_dist, epsilon)
    scale = float(high_dist.sum()) + epsilon

    for _ in range(max_iter):
        low_dist = scipy_spatial_distance.squareform(scipy_spatial_distance.pdist(y, metric="euclidean")).astype(np.float32)
        low_dist = np.maximum(low_dist, epsilon)
        delta = (high_dist - low_dist) / (high_dist * low_dist)
        np.fill_diagonal(delta, 0.0)

        gradient = np.zeros_like(y)
        for idx in range(len(y)):
            difference = y[idx] - y
            gradient[idx] = (-2.0 / scale) * np.sum(delta[idx][:, None] * difference, axis=0)
        y -= learning_rate * gradient

    return y


def _save_scatter_plot(
    plots_dir: Path,
    experiment_name: str,
    dataset_name: str,
    method_name: str,
    seed: int,
    embedding: Any,
    labels: Any,
) -> None:
    np = import_required("numpy")
    matplotlib_pyplot = import_required("matplotlib.pyplot")

    figure = matplotlib_pyplot.figure(figsize=(7, 6))
    axis = figure.add_subplot(1, 1, 1)
    scatter = axis.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=np.asarray(labels),
        s=8,
        alpha=0.8,
        cmap="tab20",
    )
    axis.set_title(f"{dataset_name} - {method_name}")
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    figure.colorbar(scatter, ax=axis)
    figure.tight_layout()
    suffix = f"__seed_{seed}"
    figure.savefig(plots_dir / f"{dataset_name}__{experiment_name}__{method_name}{suffix}.png", dpi=200)
    matplotlib_pyplot.close(figure)


def _aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    np = import_required("numpy")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    aggregated_rows: list[dict[str, Any]] = []
    for key, group in grouped.items():
        aggregate_row = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        aggregate_row["runs"] = len(group)
        for metric_name in ("elapsed_seconds",):
            values = np.asarray([row[metric_name] for row in group], dtype=float)
            aggregate_row[f"{metric_name}_mean"] = float(values.mean())
            aggregate_row[f"{metric_name}_std"] = float(values.std(ddof=0))
        aggregated_rows.append(aggregate_row)
    return aggregated_rows


def _stable_seed(base_seed: int, *parts: str) -> int:
    value = int(base_seed)
    for part in parts:
        for char in part:
            value = (value * 131 + ord(char)) % (2**31 - 1)
    return value
