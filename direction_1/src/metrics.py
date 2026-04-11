from __future__ import annotations

from typing import Any

from .utils import import_required, to_dense_array


def pairwise_distance_distortion(X_high: Any, X_low: Any, max_pairs: int, seed: int) -> float:
    np = import_required("numpy")
    high_distances, low_distances = _sample_pairwise_distances(X_high, X_low, max_pairs, seed)
    high_scaled = high_distances / (high_distances.mean() + 1e-12)
    low_scaled = low_distances / (low_distances.mean() + 1e-12)
    return float(np.mean(np.abs(high_scaled - low_scaled)))


def distance_rank_correlation(X_high: Any, X_low: Any, max_pairs: int, seed: int) -> float:
    np = import_required("numpy")
    high_distances, low_distances = _sample_pairwise_distances(X_high, X_low, max_pairs, seed)
    high_ranks = np.argsort(np.argsort(high_distances))
    low_ranks = np.argsort(np.argsort(low_distances))
    return float(np.corrcoef(high_ranks, low_ranks)[0, 1])


def knn_overlap(X_high: Any, X_low: Any, k: int, max_points: int, seed: int) -> float:
    np = import_required("numpy")
    high_dense, low_dense = _matching_dense_subsets(X_high, X_low, max_points, seed)
    high_dist = _pairwise_distance_matrix(high_dense)
    low_dist = _pairwise_distance_matrix(low_dense)
    np.fill_diagonal(high_dist, np.inf)
    np.fill_diagonal(low_dist, np.inf)

    high_nn = np.argpartition(high_dist, kth=k, axis=1)[:, :k]
    low_nn = np.argpartition(low_dist, kth=k, axis=1)[:, :k]

    overlaps = []
    for row_high, row_low in zip(high_nn, low_nn, strict=False):
        overlaps.append(len(set(row_high.tolist()) & set(row_low.tolist())) / float(k))
    return float(np.mean(overlaps))


def trustworthiness_score(X_high: Any, X_low: Any, n_neighbors: int, max_points: int, seed: int) -> float:
    sklearn_manifold = import_required("sklearn.manifold")
    high_dense, low_dense = _matching_dense_subsets(X_high, X_low, max_points, seed)
    return float(sklearn_manifold.trustworthiness(high_dense, low_dense, n_neighbors=n_neighbors))


def continuity_score(X_high: Any, X_low: Any, n_neighbors: int, max_points: int, seed: int) -> float:
    np = import_required("numpy")
    high_dense, low_dense = _matching_dense_subsets(X_high, X_low, max_points, seed)
    high_dist = _pairwise_distance_matrix(high_dense)
    low_dist = _pairwise_distance_matrix(low_dense)

    high_order = np.argsort(high_dist, axis=1)
    low_order = np.argsort(low_dist, axis=1)
    high_ranks = np.empty_like(high_order)

    for row_idx in range(high_order.shape[0]):
        high_ranks[row_idx, high_order[row_idx]] = np.arange(high_order.shape[1])

    penalty = 0.0
    for row_idx in range(high_order.shape[0]):
        high_neighbors = set(high_order[row_idx, 1 : n_neighbors + 1].tolist())
        low_neighbors = set(low_order[row_idx, 1 : n_neighbors + 1].tolist())
        for neighbor in low_neighbors - high_neighbors:
            penalty += float(high_ranks[row_idx, neighbor] - n_neighbors)

    n_samples = high_dense.shape[0]
    normalizer = n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1)
    if normalizer <= 0:
        return 0.0
    return float(1.0 - (2.0 * penalty / normalizer))


def neighborhood_hit_score(X_low: Any, y: Any, k: int, max_points: int, seed: int) -> float:
    np = import_required("numpy")
    X_dense, y_subset = _labels_with_dense_subset(X_low, y, max_points, seed)
    distances = _pairwise_distance_matrix(X_dense)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argpartition(distances, kth=k, axis=1)[:, :k]
    hits = [(y_subset[row] == y_subset[rows]).mean() for row, rows in enumerate(neighbors)]
    return float(np.mean(hits))


def class_consistency_score(X_low: Any, y: Any, k: int, max_points: int, seed: int) -> float:
    np = import_required("numpy")
    X_dense, y_subset = _labels_with_dense_subset(X_low, y, max_points, seed)
    distances = _pairwise_distance_matrix(X_dense)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argpartition(distances, kth=k, axis=1)[:, :k]

    correct = 0
    for row_idx, row_neighbors in enumerate(neighbors):
        neighbor_labels = y_subset[row_neighbors]
        vote = np.bincount(neighbor_labels).argmax()
        correct += int(vote == y_subset[row_idx])
    return float(correct / len(y_subset))


def silhouette_score_safe(X_low: Any, y: Any, max_points: int, seed: int) -> float:
    sklearn_metrics = import_required("sklearn.metrics")
    X_dense, y_subset = _labels_with_dense_subset(X_low, y, max_points, seed)
    if len(set(y_subset.tolist())) < 2:
        return 0.0
    return float(sklearn_metrics.silhouette_score(X_dense, y_subset))


def single_label_probe_metrics(X_train: Any, y_train: Any, X_test: Any, y_test: Any) -> dict[str, float]:
    sklearn_linear = import_required("sklearn.linear_model")
    sklearn_metrics = import_required("sklearn.metrics")
    classifier = sklearn_linear.LogisticRegression(max_iter=5000)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    return {
        "accuracy": float(sklearn_metrics.accuracy_score(y_test, predictions)),
        "macro_f1": float(sklearn_metrics.f1_score(y_test, predictions, average="macro")),
    }


def multilabel_probe_metrics(y_true: Any, y_score: Any, threshold: float = 0.5) -> dict[str, float]:
    np = import_required("numpy")
    sklearn_metrics = import_required("sklearn.metrics")

    y_pred = (y_score >= threshold).astype(np.int64)
    metrics = {
        "micro_f1": float(sklearn_metrics.f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(sklearn_metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "example_f1": _example_f1_score(y_true, y_pred),
        "hamming_loss": float(sklearn_metrics.hamming_loss(y_true, y_pred)),
        "average_precision": float(sklearn_metrics.average_precision_score(y_true, y_score, average="macro")),
        "label_ranking_loss": float(sklearn_metrics.label_ranking_loss(y_true, y_score)),
        "subset_accuracy": float(sklearn_metrics.accuracy_score(y_true, y_pred)),
    }
    metrics["hamming_accuracy"] = 1.0 - metrics["hamming_loss"]
    metrics["precision_at_1"] = _precision_at_k(y_true, y_score, k=1)
    metrics["precision_at_3"] = _precision_at_k(y_true, y_score, k=3)
    metrics["precision_at_5"] = _precision_at_k(y_true, y_score, k=5)
    return metrics


def one_vs_rest_logistic_scores(X_train: Any, y_train: Any, X_test: Any) -> Any:
    sklearn_linear = import_required("sklearn.linear_model")
    sklearn_multiclass = import_required("sklearn.multiclass")
    classifier = sklearn_multiclass.OneVsRestClassifier(
        sklearn_linear.LogisticRegression(max_iter=5000)
    )
    classifier.fit(X_train, y_train)
    return classifier.predict_proba(X_test)


def _example_f1_score(y_true: Any, y_pred: Any) -> float:
    np = import_required("numpy")
    true_array = np.asarray(y_true, dtype=np.int64)
    pred_array = np.asarray(y_pred, dtype=np.int64)

    scores = []
    for true_row, pred_row in zip(true_array, pred_array, strict=False):
        true_positive = int(((true_row == 1) & (pred_row == 1)).sum())
        predicted_positive = int((pred_row == 1).sum())
        actual_positive = int((true_row == 1).sum())

        if predicted_positive == 0 and actual_positive == 0:
            scores.append(1.0)
            continue

        denominator = predicted_positive + actual_positive
        scores.append(0.0 if denominator == 0 else (2.0 * true_positive / denominator))

    return float(np.mean(scores)) if scores else 0.0


def _precision_at_k(y_true: Any, y_score: Any, k: int) -> float:
    np = import_required("numpy")
    true_array = np.asarray(y_true, dtype=np.int64)
    score_array = np.asarray(y_score, dtype=np.float32)

    if score_array.ndim != 2 or score_array.shape[1] == 0:
        return 0.0

    top_k = min(k, score_array.shape[1])
    top_indices = np.argpartition(-score_array, kth=top_k - 1, axis=1)[:, :top_k]
    precision_scores = [float(true_array[row_idx, indices].mean()) for row_idx, indices in enumerate(top_indices)]
    return float(np.mean(precision_scores)) if precision_scores else 0.0


def _sample_pairwise_distances(X_high: Any, X_low: Any, max_pairs: int, seed: int) -> tuple[Any, Any]:
    np = import_required("numpy")
    high_dense, low_dense = _matching_dense_subsets(X_high, X_low, None, seed)
    n_samples = high_dense.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least two samples to compare pairwise distances")

    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, n_samples, size=max_pairs)
    idx_b = rng.integers(0, n_samples, size=max_pairs)
    mask = idx_a != idx_b
    idx_a = idx_a[mask]
    idx_b = idx_b[mask]

    high_distances = np.linalg.norm(high_dense[idx_a] - high_dense[idx_b], axis=1)
    low_distances = np.linalg.norm(low_dense[idx_a] - low_dense[idx_b], axis=1)
    return high_distances, low_distances


def _pairwise_distance_matrix(X: Any) -> Any:
    np = import_required("numpy")
    squared_norms = np.sum(X * X, axis=1, keepdims=True)
    distances = squared_norms + squared_norms.T - 2.0 * X @ X.T
    distances = np.maximum(distances, 0.0)
    return np.sqrt(distances)


def _matching_dense_subsets(X_high: Any, X_low: Any, max_points: int | None, seed: int) -> tuple[Any, Any]:
    np = import_required("numpy")
    high_dense = np.asarray(to_dense_array(X_high))
    low_dense = np.asarray(to_dense_array(X_low))
    if max_points is None or len(high_dense) <= max_points:
        return high_dense, low_dense
    rng = np.random.default_rng(seed)
    indices = np.arange(len(high_dense))
    rng.shuffle(indices)
    chosen = indices[:max_points]
    return high_dense[chosen], low_dense[chosen]


def _labels_with_dense_subset(X: Any, y: Any, max_points: int, seed: int) -> tuple[Any, Any]:
    np = import_required("numpy")
    dense = np.asarray(to_dense_array(X))
    labels = np.asarray(y)
    if max_points is None or len(labels) <= max_points:
        return dense, labels
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    chosen = indices[:max_points]
    return dense[chosen], labels[chosen]
