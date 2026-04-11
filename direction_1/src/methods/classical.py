from __future__ import annotations

from typing import Any

from ..utils import import_required, to_dense_array


def fit_visualization_reducer(method_spec: dict[str, Any], X: Any) -> tuple[Any, dict[str, Any]]:
    """Fit a reducer on the full dataset for visualization metrics."""
    base_method = method_spec["base_method"]
    params = dict(method_spec.get("params", {}))
    reducer = _build_reducer(base_method, params)

    if base_method == "tsne":
        X_dense = to_dense_array(X)
        embedding = reducer.fit_transform(X_dense)
        return embedding, {"supports_transform": False}

    if base_method == "umap":
        embedding = reducer.fit_transform(X)
        return embedding, {"supports_transform": True}

    transformed = reducer.fit_transform(X)
    return transformed, {"supports_transform": hasattr(reducer, "transform"), "model": reducer}


def fit_train_test_reducer(method_spec: dict[str, Any], X_train: Any, X_test: Any) -> tuple[Any, Any]:
    """Fit a transformable reducer on training data and apply it to test data."""
    base_method = method_spec["base_method"]
    if base_method == "tsne":
        raise ValueError("t-SNE does not support a clean train/test transform in this benchmark")

    params = dict(method_spec.get("params", {}))
    reducer = _build_reducer(base_method, params)

    if base_method == "autoencoder":
        from .neural import fit_autoencoder_reducer

        return fit_autoencoder_reducer(method_spec, X_train, X_test)

    X_train_emb = reducer.fit_transform(X_train)
    if not hasattr(reducer, "transform"):
        raise ValueError(f"Reducer '{base_method}' cannot transform held-out data")
    X_test_emb = reducer.transform(X_test)
    return X_train_emb, X_test_emb


def _build_reducer(base_method: str, params: dict[str, Any]) -> Any:
    if base_method == "autoencoder":
        return object()

    if base_method == "pca":
        sklearn_decomposition = import_required("sklearn.decomposition")
        return sklearn_decomposition.PCA(**params)

    if base_method == "truncated_svd":
        sklearn_decomposition = import_required("sklearn.decomposition")
        return sklearn_decomposition.TruncatedSVD(**params)

    if base_method == "gaussian_random_projection":
        sklearn_rp = import_required("sklearn.random_projection")
        return sklearn_rp.GaussianRandomProjection(**params)

    if base_method == "sparse_random_projection":
        sklearn_rp = import_required("sklearn.random_projection")
        return sklearn_rp.SparseRandomProjection(**params)

    if base_method == "tsne":
        sklearn_manifold = import_required("sklearn.manifold")
        return sklearn_manifold.TSNE(**params)

    if base_method == "umap":
        umap_module = import_required("umap")
        return umap_module.UMAP(**params)

    raise ValueError(f"Unsupported reducer: {base_method}")
