from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..config import load_config
from ..datasets import load_dataset, train_test_split_numpy
from ..metrics import distance_rank_correlation, knn_overlap
from ..utils import ensure_directory, import_required, save_csv, save_json, seed_everything, to_dense_array


def run_random_projection_reproduction(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    benchmark_root = Path(config["_benchmark_root"])
    output_dir = ensure_directory(benchmark_root / config["output_dir"])
    seeds = [int(seed) for seed in config.get("seeds", [config.get("seed", 0)])]
    methods = {method["name"]: method for method in config.get("methods", [])}

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for experiment in config.get("experiments", []):
        experiment_name = experiment["name"]
        dataset_spec = experiment["dataset"]
        for seed in seeds:
            try:
                seed_everything(seed)
                dataset = load_dataset(dataset_spec, benchmark_root, seed)
            except ImportError:
                raise
            except Exception as exc:
                errors.append({"experiment": experiment_name, "seed": seed, "stage": "dataset", "error": str(exc)})
                continue

            X = _to_dense_matrix(dataset["X"])
            y = dataset.get("y")
            for method_name in experiment.get("methods", []):
                method_spec = methods[method_name]
                for k in experiment.get("k_values", config.get("k_values", [])):
                    if int(k) >= X.shape[1]:
                        continue
                    try:
                        rows.append(
                            _run_single_projection(
                                config=config,
                                experiment=experiment,
                                method_spec=method_spec,
                                X=X,
                                y=y,
                                seed=seed,
                                target_dim=int(k),
                            )
                        )
                    except ImportError:
                        raise
                    except Exception as exc:
                        errors.append(
                            {
                                "experiment": experiment_name,
                                "dataset": dataset["name"],
                                "method": method_name,
                                "target_dim": int(k),
                                "seed": seed,
                                "stage": "projection",
                                "error": str(exc),
                            }
                        )

    save_csv(output_dir / "projection_runs.csv", rows)
    aggregated_rows = _aggregate_rows(rows, group_keys=["experiment", "dataset", "method", "target_dim"])
    save_csv(output_dir / "projection_runs_aggregated.csv", aggregated_rows)
    save_json(output_dir / "errors.json", errors)
    _save_curve_plots(output_dir, aggregated_rows)

    summary = {
        "seeds": seeds,
        "seed_count": len(seeds),
        "run_count": len(rows),
        "aggregated_run_count": len(aggregated_rows),
        "error_count": len(errors),
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def _run_single_projection(
    config: dict[str, Any],
    experiment: dict[str, Any],
    method_spec: dict[str, Any],
    X: Any,
    y: Any,
    seed: int,
    target_dim: int,
) -> dict[str, Any]:
    np = import_required("numpy")

    method_name = method_spec["name"]
    run_seed = _stable_seed(seed, experiment["name"], method_name, str(target_dim))
    seed_everything(run_seed)

    fit_start = time.perf_counter()
    transformed, reconstruction = _fit_and_transform(method_spec, X, target_dim, run_seed, experiment)
    elapsed_seconds = round(time.perf_counter() - fit_start, 6)

    row = {
        "experiment": experiment["name"],
        "dataset": experiment["dataset"]["name"],
        "domain": experiment["domain"],
        "condition": experiment.get("condition", experiment["domain"]),
        "method": method_name,
        "target_dim": target_dim,
        "seed": seed,
        "elapsed_seconds": elapsed_seconds,
    }

    sample_pairs = int(config.get("pairwise_samples", 4000))
    row["distance_rank_correlation"] = distance_rank_correlation(X, transformed, max_pairs=sample_pairs, seed=run_seed)
    row["knn_overlap"] = knn_overlap(
        X,
        transformed,
        k=int(config.get("neighborhood_k", 10)),
        max_points=int(config.get("max_pairwise_samples", 700)),
        seed=run_seed,
    )
    row["cosine_similarity_correlation"] = _cosine_similarity_correlation(X, transformed, max_pairs=sample_pairs, seed=run_seed)
    row["estimated_flops"] = _estimate_flops(method_name, X.shape[0], X.shape[1], target_dim)

    if reconstruction is not None:
        row["reconstruction_mse"] = _reconstruction_mse(X, reconstruction)
        row["relative_squared_error"] = _relative_squared_error(X, reconstruction)

    if y is not None and getattr(y, "ndim", 1) == 1:
        X_train, X_test, y_train, y_test = train_test_split_numpy(
            transformed,
            np.asarray(y),
            test_size=float(config.get("test_size", 0.2)),
            seed=run_seed,
            stratify=True,
        )
        row["downstream_accuracy"] = _linear_probe_accuracy(X_train, y_train, X_test, y_test)

    return row


def _fit_and_transform(
    method_spec: dict[str, Any],
    X: Any,
    target_dim: int,
    seed: int,
    experiment: dict[str, Any],
) -> tuple[Any, Any | None]:
    base_method = method_spec["base_method"]
    if base_method == "dct":
        return _apply_dct(X, target_dim, experiment)
    if base_method == "nmf":
        return _apply_nmf(X, target_dim, seed)
    return _apply_sklearn_projection(base_method, method_spec, X, target_dim, seed)


def _apply_sklearn_projection(base_method: str, method_spec: dict[str, Any], X: Any, target_dim: int, seed: int) -> tuple[Any, Any | None]:
    np = import_required("numpy")
    sklearn_decomposition = import_required("sklearn.decomposition")
    sklearn_projection = import_required("sklearn.random_projection")

    params = dict(method_spec.get("params", {}))
    params["n_components"] = target_dim

    if base_method == "pca":
        reducer = sklearn_decomposition.PCA(random_state=seed, **params)
        transformed = reducer.fit_transform(X)
        return transformed, reducer.inverse_transform(transformed)

    if base_method == "gaussian_random_projection":
        reducer = sklearn_projection.GaussianRandomProjection(random_state=seed, **params)
        transformed = reducer.fit_transform(X)
        reconstruction = transformed @ np.asarray(reducer.components_, dtype=np.float32)
        return transformed, reconstruction

    if base_method == "sparse_random_projection":
        reducer = sklearn_projection.SparseRandomProjection(random_state=seed, dense_output=True, **params)
        transformed = reducer.fit_transform(X)
        components = reducer.components_.toarray() if hasattr(reducer.components_, "toarray") else reducer.components_
        reconstruction = transformed @ np.asarray(components, dtype=np.float32)
        return transformed, reconstruction

    raise ValueError(f"Unsupported random projection reproduction method: {base_method}")


def _apply_dct(X: Any, target_dim: int, experiment: dict[str, Any]) -> tuple[Any, Any]:
    np = import_required("numpy")
    scipy_fft = import_required("scipy.fft")

    if experiment["domain"] == "image":
        height, width = _infer_image_hw(experiment, X.shape[1])
        image_array = X.reshape(-1, height, width)
        transformed = scipy_fft.dctn(image_array, axes=(1, 2), norm="ortho")
        mask = _low_frequency_mask(height, width, target_dim)
        reduced = transformed[:, mask]
        reconstructed_coeffs = np.zeros_like(transformed)
        reconstructed_coeffs[:, mask] = reduced
        reconstruction = scipy_fft.idctn(reconstructed_coeffs, axes=(1, 2), norm="ortho").reshape(len(X), -1)
        return reduced, reconstruction

    transformed = scipy_fft.dct(X, axis=1, norm="ortho")
    reduced = transformed[:, :target_dim]
    reconstructed = np.zeros_like(transformed)
    reconstructed[:, :target_dim] = reduced
    reconstruction = scipy_fft.idct(reconstructed, axis=1, norm="ortho")
    return reduced, reconstruction


def _apply_nmf(X: Any, target_dim: int, seed: int) -> tuple[Any, Any]:
    sklearn_decomposition = import_required("sklearn.decomposition")

    model = sklearn_decomposition.NMF(
        n_components=target_dim,
        init="nndsvda",
        random_state=seed,
        max_iter=400,
    )
    transformed = model.fit_transform(X)
    reconstruction = transformed @ model.components_
    return transformed, reconstruction


def _infer_image_hw(experiment: dict[str, Any], flat_dim: int) -> tuple[int, int]:
    input_shape = experiment.get("input_shape")
    if input_shape is not None:
        if len(input_shape) == 2:
            return int(input_shape[0]), int(input_shape[1])
        if len(input_shape) == 3:
            return int(input_shape[1]), int(input_shape[2])
    side = int(flat_dim**0.5)
    if side * side != flat_dim:
        raise ValueError("Image experiment requires square flattened images or an explicit input_shape")
    return side, side


def _low_frequency_mask(height: int, width: int, target_dim: int) -> Any:
    np = import_required("numpy")
    coords = [(row, col) for row in range(height) for col in range(width)]
    coords.sort(key=lambda item: (item[0] + item[1], item[0]))
    mask = np.zeros((height, width), dtype=bool)
    for row, col in coords[:target_dim]:
        mask[row, col] = True
    return mask


def _reconstruction_mse(X: Any, reconstruction: Any) -> float:
    np = import_required("numpy")
    return float(np.mean((np.asarray(X) - np.asarray(reconstruction)) ** 2))


def _relative_squared_error(X: Any, reconstruction: Any) -> float:
    np = import_required("numpy")
    numerator = np.sum((np.asarray(X) - np.asarray(reconstruction)) ** 2)
    denominator = np.sum(np.asarray(X) ** 2) + 1e-12
    return float(numerator / denominator)


def _cosine_similarity_correlation(X_high: Any, X_low: Any, max_pairs: int, seed: int) -> float:
    np = import_required("numpy")
    rng = np.random.default_rng(seed)
    n_samples = len(X_high)
    idx_a = rng.integers(0, n_samples, size=max_pairs)
    idx_b = rng.integers(0, n_samples, size=max_pairs)
    mask = idx_a != idx_b
    idx_a = idx_a[mask]
    idx_b = idx_b[mask]

    def cosine_pairs(matrix: Any) -> Any:
        left = matrix[idx_a]
        right = matrix[idx_b]
        numerator = np.sum(left * right, axis=1)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-12
        return numerator / denominator

    high_cos = cosine_pairs(np.asarray(X_high, dtype=np.float32))
    low_cos = cosine_pairs(np.asarray(X_low, dtype=np.float32))
    return float(np.corrcoef(high_cos, low_cos)[0, 1])


def _estimate_flops(method_name: str, n_samples: int, input_dim: int, target_dim: int) -> float:
    if method_name in {"rp", "srp"}:
        return float(n_samples * input_dim * target_dim)
    if method_name == "pca":
        return float(min(n_samples * input_dim * target_dim, input_dim * input_dim * n_samples))
    if method_name == "dct":
        return float(n_samples * input_dim * max(1, (input_dim.bit_length())))
    if method_name == "nmf":
        return float(10 * n_samples * input_dim * target_dim)
    return float(n_samples * input_dim * target_dim)


def _linear_probe_accuracy(X_train: Any, y_train: Any, X_test: Any, y_test: Any) -> float:
    sklearn_linear = import_required("sklearn.linear_model")
    sklearn_metrics = import_required("sklearn.metrics")
    classifier = sklearn_linear.LogisticRegression(max_iter=5000)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    return float(sklearn_metrics.accuracy_score(y_test, predictions))


def _to_dense_matrix(X: Any) -> Any:
    np = import_required("numpy")
    return np.asarray(to_dense_array(X), dtype=np.float32)


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

        metric_names = []
        for row in group:
            for metric_name, metric_value in row.items():
                if metric_name in group_keys or metric_name == "seed":
                    continue
                if isinstance(metric_value, (int, float)) and metric_name not in metric_names:
                    metric_names.append(metric_name)

        for metric_name in metric_names:
            values = np.asarray([row[metric_name] for row in group], dtype=float)
            aggregate_row[f"{metric_name}_mean"] = float(values.mean())
            aggregate_row[f"{metric_name}_std"] = float(values.std(ddof=0))

        aggregated_rows.append(aggregate_row)

    aggregated_rows.sort(key=lambda row: tuple(row[key] for key in group_keys))
    return aggregated_rows


def _save_curve_plots(output_dir: Path, aggregated_rows: list[dict[str, Any]]) -> None:
    if not aggregated_rows:
        return

    matplotlib_pyplot = import_required("matplotlib.pyplot")
    plots_dir = ensure_directory(output_dir / "plots")

    metrics_to_plot = [
        "reconstruction_mse_mean",
        "relative_squared_error_mean",
        "distance_rank_correlation_mean",
        "knn_overlap_mean",
        "cosine_similarity_correlation_mean",
        "downstream_accuracy_mean",
        "elapsed_seconds_mean",
        "estimated_flops_mean",
    ]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in aggregated_rows:
        grouped.setdefault((row["experiment"], row["dataset"]), []).append(row)

    for (experiment_name, dataset_name), rows in grouped.items():
        rows.sort(key=lambda item: item["target_dim"])
        methods = sorted({row["method"] for row in rows})
        for metric_name in metrics_to_plot:
            plot_rows = [row for row in rows if metric_name in row]
            if not plot_rows:
                continue
            figure = matplotlib_pyplot.figure(figsize=(7, 5))
            axis = figure.add_subplot(1, 1, 1)
            for method_name in methods:
                method_rows = [row for row in plot_rows if row["method"] == method_name]
                if not method_rows:
                    continue
                axis.plot(
                    [row["target_dim"] for row in method_rows],
                    [row[metric_name] for row in method_rows],
                    marker="o",
                    label=method_name,
                )
            axis.set_title(f"{experiment_name} - {metric_name}")
            axis.set_xlabel("Target Dimension")
            axis.set_ylabel(metric_name.replace("_mean", ""))
            axis.legend()
            axis.grid(True, alpha=0.3)
            figure.tight_layout()
            figure.savefig(plots_dir / f"{dataset_name}__{experiment_name}__{metric_name}.png", dpi=200)
            matplotlib_pyplot.close(figure)


def _stable_seed(base_seed: int, *parts: str) -> int:
    value = int(base_seed)
    for part in parts:
        for char in part:
            value = (value * 131 + ord(char)) % (2**31 - 1)
    return value
