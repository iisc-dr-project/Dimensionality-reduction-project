from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .config import load_config
from .datasets import load_dataset, take_training_fraction, train_test_split_numpy
from .metrics import (
    class_consistency_score,
    continuity_score,
    distance_rank_correlation,
    knn_overlap,
    multilabel_probe_metrics,
    neighborhood_hit_score,
    one_vs_rest_logistic_scores,
    pairwise_distance_distortion,
    silhouette_score_safe,
    single_label_probe_metrics,
    trustworthiness_score,
)
from .methods.cgmvae import fit_cgmvae, fit_vae_multilabel
from .methods.classical import fit_train_test_reducer, fit_visualization_reducer
from .methods.neural import fit_autoencoder_reducer, fit_cnn_multilabel, fit_mlp_multilabel, fit_vae_reducer
from .utils import ensure_directory, import_required, save_csv, save_json, seed_everything


def run_benchmark(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    benchmark_root = Path(config["_benchmark_root"])
    output_dir = ensure_directory(benchmark_root / config["output_dir"])
    seeds = _resolve_seeds(config)

    visualization_rows: list[dict[str, Any]] = []
    downstream_rows: list[dict[str, Any]] = []
    multilabel_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for seed in seeds:
        seed_everything(seed)
        for dataset_spec in config.get("datasets", []):
            dataset_name = dataset_spec["name"]
            try:
                dataset = load_dataset(dataset_spec, benchmark_root, seed)
                if dataset_spec["task"] == "multilabel_prediction":
                    multilabel_rows.extend(_run_multilabel_suite(config, dataset, seed))
                else:
                    vis_rows, down_rows = _run_visualization_suite(config, dataset, seed)
                    visualization_rows.extend(vis_rows)
                    downstream_rows.extend(down_rows)
            except ImportError:
                raise
            except Exception as exc:
                errors.append({"seed": seed, "dataset": dataset_name, "stage": "dataset", "error": str(exc)})

    save_csv(output_dir / "visualization_metrics.csv", visualization_rows)
    save_csv(output_dir / "downstream_metrics.csv", downstream_rows)
    save_csv(output_dir / "multilabel_metrics.csv", multilabel_rows)

    save_csv(
        output_dir / "visualization_metrics_aggregated.csv",
        _aggregate_rows(visualization_rows, group_keys=["dataset", "method"]),
    )
    save_csv(
        output_dir / "downstream_metrics_aggregated.csv",
        _aggregate_rows(downstream_rows, group_keys=["dataset", "method"]),
    )
    save_csv(
        output_dir / "multilabel_metrics_aggregated.csv",
        _aggregate_rows(multilabel_rows, group_keys=["dataset", "method", "training_fraction"]),
    )
    save_json(output_dir / "errors.json", errors)

    summary = {
        "seeds": seeds,
        "seed_count": len(seeds),
        "visualization_runs": len(visualization_rows),
        "downstream_runs": len(downstream_rows),
        "multilabel_runs": len(multilabel_rows),
        "error_count": len(errors),
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def _run_visualization_suite(
    config: dict[str, Any],
    dataset: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    visualization_rows: list[dict[str, Any]] = []
    downstream_rows: list[dict[str, Any]] = []

    benchmark_cfg = config["benchmark"]
    X = dataset["X"]
    y = dataset["y"]

    for method_spec in config.get("visualization_methods", []):
        if not _method_applies(method_spec, dataset):
            continue
        start_time = time.perf_counter()
        method_seed = _stable_seed(seed, dataset["name"], method_spec["name"], "visualization")
        try:
            seed_everything(method_seed)
            if method_spec["base_method"] == "autoencoder":
                embedding, _ = fit_autoencoder_reducer(method_spec, X)
            elif method_spec["base_method"] == "vae":
                embedding, _ = fit_vae_reducer(method_spec, X)
            else:
                embedding, _ = fit_visualization_reducer(method_spec, X)

            row = {
                "seed": seed,
                "dataset": dataset["name"],
                "method": method_spec["name"],
                "pairwise_distance_distortion": pairwise_distance_distortion(
                    X,
                    embedding,
                    max_pairs=int(benchmark_cfg["distance_pairs"]),
                    seed=method_seed,
                ),
                "distance_rank_correlation": distance_rank_correlation(
                    X,
                    embedding,
                    max_pairs=int(benchmark_cfg["distance_pairs"]),
                    seed=method_seed,
                ),
                "knn_overlap": knn_overlap(
                    X,
                    embedding,
                    k=int(benchmark_cfg["neighborhood_k"]),
                    max_points=int(benchmark_cfg["max_pairwise_samples"]),
                    seed=method_seed,
                ),
                "trustworthiness": trustworthiness_score(
                    X,
                    embedding,
                    n_neighbors=int(benchmark_cfg["neighborhood_k"]),
                    max_points=int(benchmark_cfg["max_trustworthiness_samples"]),
                    seed=method_seed,
                ),
                "continuity": continuity_score(
                    X,
                    embedding,
                    n_neighbors=int(benchmark_cfg["neighborhood_k"]),
                    max_points=int(benchmark_cfg["max_trustworthiness_samples"]),
                    seed=method_seed,
                ),
                "neighborhood_hit": neighborhood_hit_score(
                    embedding,
                    y,
                    k=int(benchmark_cfg["neighborhood_k"]),
                    max_points=int(benchmark_cfg["max_pairwise_samples"]),
                    seed=method_seed,
                ),
                "silhouette_score": silhouette_score_safe(
                    embedding,
                    y,
                    max_points=int(benchmark_cfg["max_pairwise_samples"]),
                    seed=method_seed,
                ),
                "class_consistency": class_consistency_score(
                    embedding,
                    y,
                    k=int(benchmark_cfg["neighborhood_k"]),
                    max_points=int(benchmark_cfg["max_pairwise_samples"]),
                    seed=method_seed,
                ),
                "elapsed_seconds": round(time.perf_counter() - start_time, 4),
            }
            visualization_rows.append(row)
            _save_embedding_plot(config, dataset["name"], method_spec["name"], embedding, y, seed)
        except ImportError:
            raise
        except Exception as exc:
            visualization_rows.append(
                {
                    "seed": seed,
                    "dataset": dataset["name"],
                    "method": method_spec["name"],
                    "error": str(exc),
                    "elapsed_seconds": round(time.perf_counter() - start_time, 4),
                }
            )

    X_train, X_test, y_train, y_test = train_test_split_numpy(
        X,
        y,
        test_size=float(config["split"]["test_size"]),
        seed=seed,
        stratify=True,
    )

    for method_spec in config.get("downstream_methods", []):
        if not _method_applies(method_spec, dataset):
            continue
        start_time = time.perf_counter()
        method_seed = _stable_seed(seed, dataset["name"], method_spec["name"], "downstream")
        try:
            seed_everything(method_seed)
            X_train_red, X_test_red = fit_train_test_reducer(method_spec, X_train, X_test)
            metrics = single_label_probe_metrics(X_train_red, y_train, X_test_red, y_test)
            row = {
                "seed": seed,
                "dataset": dataset["name"],
                "method": method_spec["name"],
                **metrics,
                "elapsed_seconds": round(time.perf_counter() - start_time, 4),
            }
            downstream_rows.append(row)
        except ImportError:
            raise
        except Exception as exc:
            downstream_rows.append(
                {
                    "seed": seed,
                    "dataset": dataset["name"],
                    "method": method_spec["name"],
                    "error": str(exc),
                    "elapsed_seconds": round(time.perf_counter() - start_time, 4),
                }
            )

    return visualization_rows, downstream_rows


def _run_multilabel_suite(config: dict[str, Any], dataset: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    X_train, X_test, y_train, y_test = train_test_split_numpy(
        dataset["X"],
        dataset["y"],
        test_size=float(config["split"]["test_size"]),
        seed=seed,
        stratify=False,
    )

    for method_spec in config.get("multilabel_methods", []):
        if not _method_applies(method_spec, dataset):
            continue
        fractions = method_spec.get("fractions", [1.0])
        for fraction in fractions:
            start_time = time.perf_counter()
            method_seed = _stable_seed(seed, dataset["name"], method_spec["name"], str(fraction))
            try:
                seed_everything(method_seed)
                X_train_sub, y_train_sub = take_training_fraction(X_train, y_train, float(fraction), method_seed)
                scores = _fit_multilabel_method(method_spec, X_train_sub, y_train_sub, X_test)
                metrics = multilabel_probe_metrics(y_test, scores["probabilities"])
                row = {
                    "seed": seed,
                    "dataset": dataset["name"],
                    "method": method_spec["name"],
                    "training_fraction": fraction,
                    **metrics,
                    "elapsed_seconds": round(time.perf_counter() - start_time, 4),
                }
                rows.append(row)
                if "embeddings" in scores:
                    _save_embedding_plot(
                        config,
                        dataset["name"],
                        f"{method_spec['name']}_fraction_{fraction}",
                        scores["embeddings"],
                        y_test.sum(axis=1),
                        seed,
                    )
            except ImportError:
                raise
            except Exception as exc:
                rows.append(
                    {
                        "seed": seed,
                        "dataset": dataset["name"],
                        "method": method_spec["name"],
                        "training_fraction": fraction,
                        "error": str(exc),
                        "elapsed_seconds": round(time.perf_counter() - start_time, 4),
                    }
                )
    return rows


def _fit_multilabel_method(method_spec: dict[str, Any], X_train: Any, y_train: Any, X_test: Any) -> dict[str, Any]:
    base_method = method_spec["base_method"]
    if base_method == "one_vs_rest_logistic":
        return {"probabilities": one_vs_rest_logistic_scores(X_train, y_train, X_test)}
    if base_method == "mlp_multilabel":
        return fit_mlp_multilabel(method_spec, X_train, y_train, X_test)
    if base_method == "cnn_multilabel":
        return fit_cnn_multilabel(method_spec, X_train, y_train, X_test)
    if base_method == "vae_multilabel":
        return fit_vae_multilabel(method_spec, X_train, y_train, X_test)
    if base_method == "cgmvae":
        return fit_cgmvae(method_spec, X_train, y_train, X_test)
    raise ValueError(f"Unsupported multi-label method: {base_method}")


def _save_embedding_plot(
    config: dict[str, Any],
    dataset_name: str,
    method_name: str,
    embedding: Any,
    labels: Any,
    seed: int,
) -> None:
    np = import_required("numpy")
    matplotlib_pyplot = import_required("matplotlib.pyplot")
    sklearn_decomposition = import_required("sklearn.decomposition")

    benchmark_root = Path(config["_benchmark_root"])
    plots_dir = ensure_directory(benchmark_root / config["output_dir"] / "plots")

    embedding_array = np.asarray(embedding)
    if embedding_array.ndim != 2:
        return

    if embedding_array.shape[1] > 2:
        reducer = sklearn_decomposition.PCA(n_components=2)
        embedding_array = reducer.fit_transform(embedding_array)

    figure = matplotlib_pyplot.figure(figsize=(7, 6))
    axis = figure.add_subplot(1, 1, 1)
    scatter = axis.scatter(
        embedding_array[:, 0],
        embedding_array[:, 1],
        c=np.asarray(labels),
        s=8,
        alpha=0.8,
        cmap="tab10",
    )
    axis.set_title(f"{dataset_name} - {method_name}")
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    figure.colorbar(scatter, ax=axis)
    figure.tight_layout()

    seed_suffix = f"__seed_{seed}" if len(_resolve_seeds(config)) > 1 else ""
    figure.savefig(plots_dir / f"{dataset_name}__{method_name}{seed_suffix}.png", dpi=200)
    matplotlib_pyplot.close(figure)


def _method_applies(method_spec: dict[str, Any], dataset: dict[str, Any]) -> bool:
    supported_datasets = method_spec.get("supported_datasets")
    if supported_datasets is not None and dataset["name"] not in supported_datasets:
        return False

    if method_spec.get("requires_image_like", False) and not dataset.get("is_image_like", False):
        return False

    return True


def _resolve_seeds(config: dict[str, Any]) -> list[int]:
    if "seeds" in config:
        return [int(seed) for seed in config["seeds"]]
    return [int(config.get("seed", 0))]


def _stable_seed(base_seed: int, *parts: str) -> int:
    value = int(base_seed)
    for part in parts:
        for char in part:
            value = (value * 131 + ord(char)) % (2**31 - 1)
    return value


def _aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    np = import_required("numpy")

    successful_rows = [row for row in rows if "error" not in row]
    if not successful_rows:
        return []

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in successful_rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    aggregated_rows: list[dict[str, Any]] = []
    for key, group in grouped.items():
        aggregate_row = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        aggregate_row["runs"] = len(group)

        numeric_keys: list[str] = []
        for row in group:
            for metric_name, metric_value in row.items():
                if metric_name in group_keys or metric_name == "seed" or metric_name == "error":
                    continue
                if isinstance(metric_value, (int, float)) and metric_name not in numeric_keys:
                    numeric_keys.append(metric_name)

        for metric_name in numeric_keys:
            values = np.asarray([row[metric_name] for row in group if metric_name in row], dtype=float)
            aggregate_row[f"{metric_name}_mean"] = float(values.mean())
            aggregate_row[f"{metric_name}_std"] = float(values.std(ddof=0))

        aggregated_rows.append(aggregate_row)

    return aggregated_rows
