from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from .utils import import_required, resolve_path


def load_dataset(spec: dict[str, Any], benchmark_root: str | Path, seed: int) -> dict[str, Any]:
    loader_name = spec["loader"]
    if loader_name == "mnist_local":
        return _load_mnist_local(spec, benchmark_root, seed)
    if loader_name == "olivetti_faces":
        return _load_olivetti_faces(spec, seed)
    if loader_name == "coil20_local":
        return _load_coil20_local(spec, benchmark_root)
    if loader_name == "twenty_newsgroups":
        return _load_twenty_newsgroups(spec, seed)
    if loader_name == "synthetic_multilabel":
        return _load_synthetic_multilabel(spec, seed)
    if loader_name == "synthetic_image_multilabel":
        return _load_synthetic_image_multilabel(spec, seed)
    if loader_name == "multilabel_csv":
        return _load_multilabel_csv(spec, benchmark_root)
    if loader_name == "digits":
        return _load_digits(spec, seed)
    if loader_name == "swiss_roll":
        return _load_swiss_roll(spec, seed)
    raise ValueError(f"Unsupported dataset loader: {loader_name}")


def train_test_split_numpy(
    X: Any,
    y: Any,
    test_size: float,
    seed: int,
    stratify: bool = True,
) -> tuple[Any, Any, Any, Any]:
    np = import_required("numpy")
    indices = np.arange(len(y))
    rng = np.random.default_rng(seed)

    if stratify and getattr(y, "ndim", 1) == 1:
        train_parts: list[Any] = []
        test_parts: list[Any] = []
        for label in np.unique(y):
            label_indices = indices[y == label]
            rng.shuffle(label_indices)
            split_at = max(1, int(round(len(label_indices) * (1.0 - test_size))))
            train_parts.append(label_indices[:split_at])
            test_parts.append(label_indices[split_at:])
        train_indices = np.concatenate(train_parts)
        test_indices = np.concatenate(test_parts)
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)
    else:
        rng.shuffle(indices)
        split_at = int(round(len(indices) * (1.0 - test_size)))
        train_indices = indices[:split_at]
        test_indices = indices[split_at:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def subsample_rows(X: Any, y: Any, limit: int | None, seed: int, stratify: bool = True) -> tuple[Any, Any]:
    if limit is None or limit >= len(y):
        return X, y

    np = import_required("numpy")
    rng = np.random.default_rng(seed)

    if stratify and getattr(y, "ndim", 1) == 1:
        chosen_indices: list[Any] = []
        labels = np.unique(y)
        quota = max(1, limit // len(labels))
        for label in labels:
            label_indices = np.where(y == label)[0]
            rng.shuffle(label_indices)
            chosen_indices.append(label_indices[:quota])
        merged = np.concatenate(chosen_indices)
        if len(merged) < limit:
            remaining = np.setdiff1d(np.arange(len(y)), merged, assume_unique=False)
            rng.shuffle(remaining)
            merged = np.concatenate([merged, remaining[: limit - len(merged)]])
        rng.shuffle(merged)
        return X[merged], y[merged]

    indices = np.arange(len(y))
    rng.shuffle(indices)
    selected = indices[:limit]
    return X[selected], y[selected]


def take_training_fraction(X: Any, y: Any, fraction: float, seed: int) -> tuple[Any, Any]:
    np = import_required("numpy")
    if fraction >= 1.0:
        return X, y
    count = max(1, int(round(len(y) * fraction)))
    indices = np.arange(len(y))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    chosen = indices[:count]
    return X[chosen], y[chosen]


def _read_idx_images(path: Path) -> Any:
    np = import_required("numpy")
    with path.open("rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number in {path}: {magic}")
        raw = handle.read()
    images = np.frombuffer(raw, dtype=np.uint8).reshape(count, rows * cols)
    return images.astype(np.float32) / 255.0


def _read_idx_labels(path: Path) -> Any:
    np = import_required("numpy")
    with path.open("rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number in {path}: {magic}")
        raw = handle.read()
    labels = np.frombuffer(raw, dtype=np.uint8)
    if labels.shape[0] != count:
        raise ValueError(f"Label count mismatch in {path}")
    return labels.astype(np.int64)


def _load_mnist_local(spec: dict[str, Any], benchmark_root: str | Path, seed: int) -> dict[str, Any]:
    np = import_required("numpy")
    params = spec.get("params", {})
    root = Path(benchmark_root)
    train_images = _read_idx_images(resolve_path(root, params["train_images"]))
    train_labels = _read_idx_labels(resolve_path(root, params["train_labels"]))

    if params.get("merge_train_and_test", True):
        test_images = _read_idx_images(resolve_path(root, params["test_images"]))
        test_labels = _read_idx_labels(resolve_path(root, params["test_labels"]))
        X = np.concatenate([train_images, test_images], axis=0)
        y = np.concatenate([train_labels, test_labels], axis=0)
    else:
        X = train_images
        y = train_labels

    noise_std = float(params.get("noise_std", 0.0))
    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        X = np.clip(X + rng.normal(0.0, noise_std, size=X.shape).astype(np.float32), 0.0, 1.0)

    if not params.get("flatten", True):
        side = int(np.sqrt(X.shape[1]))
        X = X.reshape(-1, side, side)

    X, y = subsample_rows(X, y, spec.get("limit"), seed, stratify=True)
    image_shape = [1, int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))] if params.get("flatten", True) else [1, X.shape[1], X.shape[2]]
    return {
        "name": spec["name"],
        "task": spec["task"],
        "X": X,
        "y": y,
        "multilabel": False,
        "is_image_like": True,
        "image_shape": image_shape,
    }


def _load_twenty_newsgroups(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    np = import_required("numpy")
    sklearn_datasets = import_required("sklearn.datasets")
    sklearn_feature_text = import_required("sklearn.feature_extraction.text")

    params = spec.get("params", {})
    bunch = sklearn_datasets.fetch_20newsgroups(
        subset=params.get("subset", "train"),
        categories=params.get("categories"),
        remove=tuple(params.get("remove", [])),
    )

    vectorizer = sklearn_feature_text.TfidfVectorizer(
        stop_words="english",
        max_features=params.get("max_features", 2000),
    )
    X = vectorizer.fit_transform(bunch.data)
    y = np.asarray(bunch.target, dtype=np.int64)
    X, y = subsample_rows(X, y, spec.get("limit"), seed, stratify=True)
    return {
        "name": spec["name"],
        "task": spec["task"],
        "X": X,
        "y": y,
        "multilabel": False,
        "is_image_like": False,
        "label_names": list(bunch.target_names),
    }


def _load_olivetti_faces(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    sklearn_datasets = import_required("sklearn.datasets")
    params = spec.get("params", {})
    dataset = sklearn_datasets.fetch_olivetti_faces(shuffle=True, random_state=seed)
    X = dataset.data
    y = dataset.target
    X, y = subsample_rows(X, y, spec.get("limit"), seed, stratify=True)
    return {
        "name": spec["name"],
        "task": spec["task"],
        "X": X,
        "y": y,
        "multilabel": False,
        "is_image_like": True,
        "image_shape": [1, 64, 64],
        "label_names": [str(idx) for idx in range(int(y.max()) + 1)],
        "metadata": {"source": params.get("source", "sklearn.fetch_olivetti_faces")},
    }


def _load_coil20_local(spec: dict[str, Any], benchmark_root: str | Path) -> dict[str, Any]:
    np = import_required("numpy")
    matplotlib_image = import_required("matplotlib.image")

    params = spec.get("params", {})
    root = resolve_path(Path(benchmark_root), params["root"])
    pattern = params.get("pattern", "*.png")
    image_paths = sorted(root.glob(pattern))
    if not image_paths:
        raise FileNotFoundError(f"No COIL-20 images found in {root} matching pattern '{pattern}'")

    X_rows: list[Any] = []
    labels: list[int] = []
    image_shape: list[int] | None = None

    for image_path in image_paths:
        label = _parse_coil20_label(image_path.name)
        image = matplotlib_image.imread(image_path)
        if image.ndim == 3:
            image = image[..., :3].mean(axis=2)
        image = np.asarray(image, dtype=np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        if image_shape is None:
            image_shape = [1, int(image.shape[0]), int(image.shape[1])]
        X_rows.append(image.reshape(-1))
        labels.append(label)

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return {
        "name": spec["name"],
        "task": spec["task"],
        "X": X,
        "y": y,
        "multilabel": False,
        "is_image_like": True,
        "image_shape": image_shape,
        "label_names": [str(label) for label in sorted(set(labels))],
    }


def _load_synthetic_multilabel(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    sklearn_datasets = import_required("sklearn.datasets")
    params = spec.get("params", {})
    X, y = sklearn_datasets.make_multilabel_classification(
        n_samples=params.get("n_samples", 2000),
        n_features=params.get("n_features", 256),
        n_classes=params.get("n_classes", 10),
        n_labels=params.get("n_labels", 3),
        length=params.get("length", 50),
        allow_unlabeled=params.get("allow_unlabeled", False),
        sparse=params.get("sparse", False),
        random_state=seed,
    )
    return {"name": spec["name"], "task": spec["task"], "X": X, "y": y, "multilabel": True}


def _load_synthetic_image_multilabel(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    np = import_required("numpy")

    params = spec.get("params", {})
    n_samples = int(params.get("n_samples", 2500))
    image_size = int(params.get("image_size", 16))
    n_classes = int(params.get("n_classes", 6))
    min_labels = int(params.get("min_labels", 1))
    max_labels = int(params.get("max_labels", 3))
    noise_std = float(params.get("noise_std", 0.1))

    if n_classes > 8:
        raise ValueError("synthetic_image_multilabel supports up to 8 labels")

    rng = np.random.default_rng(seed)
    basis = _make_multilabel_image_basis(image_size)[:n_classes]

    X = np.zeros((n_samples, image_size * image_size), dtype=np.float32)
    y = np.zeros((n_samples, n_classes), dtype=np.int64)

    for sample_idx in range(n_samples):
        label_count = int(rng.integers(min_labels, max_labels + 1))
        active = rng.choice(n_classes, size=label_count, replace=False)
        image = np.zeros((image_size, image_size), dtype=np.float32)
        y[sample_idx, active] = 1
        for label_idx in active:
            image += basis[label_idx]
        image += rng.normal(0.0, noise_std, size=image.shape).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        X[sample_idx] = image.reshape(-1)

    return {
        "name": spec["name"],
        "task": spec["task"],
        "X": X,
        "y": y,
        "multilabel": True,
        "is_image_like": True,
        "image_shape": [1, image_size, image_size],
        "label_names": [f"pattern_{idx}" for idx in range(n_classes)],
    }


def _load_multilabel_csv(spec: dict[str, Any], benchmark_root: str | Path) -> dict[str, Any]:
    np = import_required("numpy")
    pandas = import_required("pandas")
    params = spec.get("params", {})
    csv_path = resolve_path(Path(benchmark_root), params["path"])
    frame = pandas.read_csv(csv_path)

    label_columns = params.get("label_columns")
    if not label_columns:
        raise ValueError("multilabel_csv datasets must specify 'label_columns' in the config")

    feature_columns = params.get("feature_columns")
    if feature_columns is None:
        feature_columns = [column for column in frame.columns if column not in label_columns]

    X = frame[feature_columns].to_numpy(dtype=np.float32)
    y = frame[label_columns].to_numpy(dtype=np.int64)
    return {
        "name": spec["name"],
        "task": spec["task"],
        "X": X,
        "y": y,
        "multilabel": True,
        "is_image_like": False,
        "label_names": list(label_columns),
    }


def _load_digits(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    np = import_required("numpy")
    sklearn_datasets = import_required("sklearn.datasets")
    params = spec.get("params", {})
    dataset = sklearn_datasets.load_digits()
    X = dataset.data.astype(np.float32)
    noise_std = float(params.get("noise_std", 0.0))
    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        X = np.clip(X + rng.normal(0.0, noise_std, size=X.shape), 0.0, 16.0)
    y = dataset.target
    X, y = subsample_rows(X, y, spec.get("limit"), seed, stratify=True)
    return {
        "name": spec["name"],
        "task": spec["task"],
        "X": X,
        "y": y,
        "multilabel": False,
        "is_image_like": True,
        "image_shape": [1, 8, 8],
    }


def _load_swiss_roll(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    np = import_required("numpy")
    sklearn_datasets = import_required("sklearn.datasets")

    params = spec.get("params", {})
    X, position = sklearn_datasets.make_swiss_roll(
        n_samples=params.get("n_samples", 2500),
        noise=params.get("noise", 0.05),
        random_state=seed,
    )

    n_bins = int(params.get("n_bins", 8))
    bins = np.linspace(position.min(), position.max(), n_bins + 1)
    y = np.digitize(position, bins[1:-1]).astype(np.int64)
    X, y = subsample_rows(X.astype(np.float32), y, spec.get("limit"), seed, stratify=True)
    return {"name": spec["name"], "task": spec["task"], "X": X, "y": y, "multilabel": False, "is_image_like": False}


def _make_multilabel_image_basis(image_size: int) -> list[Any]:
    np = import_required("numpy")
    basis: list[Any] = []
    grid = np.zeros((image_size, image_size), dtype=np.float32)
    stripe = max(2, image_size // 6)
    center_low = image_size // 3
    center_high = image_size - center_low

    pattern = grid.copy()
    pattern[:, :stripe] = 1.0
    basis.append(pattern)

    pattern = grid.copy()
    pattern[:, -stripe:] = 1.0
    basis.append(pattern)

    pattern = grid.copy()
    pattern[:stripe, :] = 1.0
    basis.append(pattern)

    pattern = grid.copy()
    pattern[-stripe:, :] = 1.0
    basis.append(pattern)

    pattern = grid.copy()
    np.fill_diagonal(pattern, 1.0)
    basis.append(pattern)

    pattern = grid.copy()
    np.fill_diagonal(np.fliplr(pattern), 1.0)
    basis.append(pattern)

    pattern = grid.copy()
    pattern[center_low:center_high, center_low:center_high] = 1.0
    basis.append(pattern)

    pattern = grid.copy()
    pattern[[0, -1], :] = 1.0
    pattern[:, [0, -1]] = 1.0
    basis.append(pattern)

    return basis


def _parse_coil20_label(filename: str) -> int:
    stem = Path(filename).stem.lower()
    digits = "".join(char for char in stem if char.isdigit())
    if not digits:
        raise ValueError(f"Could not infer COIL-20 object id from filename: {filename}")
    if "__" in stem:
        object_part = stem.split("__", 1)[0]
        object_digits = "".join(char for char in object_part if char.isdigit())
        if object_digits:
            return int(object_digits)
    return int(digits)
