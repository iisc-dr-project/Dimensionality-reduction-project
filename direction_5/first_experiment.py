import csv
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10, cifar100, fashion_mnist

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
TEXT_OUTPUT_PATH = OUTPUT_DIR / "first_experiment_results.txt"
CSV_OUTPUT_PATH = OUTPUT_DIR / "first_experiment_results.csv"


def log(message, log_lines):
    print(message)
    log_lines.append(message)


def build_cnn(input_shape, num_classes):
    """Upgraded CNN with deeper architecture, Batch Norm, and Dropout."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_pipeline(name, train_data, y_train, test_data, y_test, input_shape, num_classes, log_lines):
    log(f"  -> Training {name}...", log_lines)
    model = build_cnn(input_shape, num_classes)

    start_train = time.time()
    model.fit(train_data, y_train, epochs=5, batch_size=128, verbose=0)
    train_time = time.time() - start_train

    start_infer = time.time()
    _, accuracy = model.evaluate(test_data, y_test, verbose=0)
    infer_time = time.time() - start_infer

    log(
        f"     Accuracy: {accuracy:.4f} | Train Time: {train_time:.2f}s | Infer Time: {infer_time:.2f}s",
        log_lines,
    )
    return {
        "method": name,
        "accuracy": float(accuracy),
        "train_time_seconds": train_time,
        "infer_time_seconds": infer_time,
    }


def write_outputs(log_lines, rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_OUTPUT_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    with CSV_OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "dataset",
                "method",
                "accuracy",
                "train_time_seconds",
                "infer_time_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


datasets = {
    "Fashion-MNIST": {
        "loader": fashion_mnist.load_data,
        "shape": (28, 28, 1),
        "classes": 10,
        "pca_k": [64, 128],
    },
    "CIFAR-10": {
        "loader": cifar10.load_data,
        "shape": (32, 32, 3),
        "classes": 10,
        "pca_k": [64, 128],
    },
    "CIFAR-100": {
        "loader": lambda: cifar100.load_data(label_mode="fine"),
        "shape": (32, 32, 3),
        "classes": 100,
        "pca_k": [64, 128],
    },
}


def main():
    log_lines = []
    rows = []

    for ds_name, config in datasets.items():
        log("", log_lines)
        log("=" * 40, log_lines)
        log(f"DATASET: {ds_name}", log_lines)
        log("=" * 40, log_lines)

        (x_train, y_train), (x_test, y_test) = config["loader"]()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        x_train = x_train.reshape(-1, *config["shape"])
        x_test = x_test.reshape(-1, *config["shape"])

        flat_dim = int(np.prod(config["shape"]))
        x_train_flat = x_train.reshape(-1, flat_dim)
        x_test_flat = x_test.reshape(-1, flat_dim)

        baseline_metrics = evaluate_pipeline(
            "Baseline (Raw Data)",
            x_train,
            y_train,
            x_test,
            y_test,
            config["shape"],
            config["classes"],
            log_lines,
        )
        rows.append({"dataset": ds_name, **baseline_metrics})

        for k in config["pca_k"]:
            log(f"  -> Applying PCA (k={k}) to {ds_name}...", log_lines)
            pca = PCA(n_components=k)

            x_train_pca = pca.fit_transform(x_train_flat)
            x_train_recon = pca.inverse_transform(x_train_pca)

            x_test_pca = pca.transform(x_test_flat)
            x_test_recon = pca.inverse_transform(x_test_pca)

            x_train_cnn = x_train_recon.reshape(-1, *config["shape"])
            x_test_cnn = x_test_recon.reshape(-1, *config["shape"])

            pca_metrics = evaluate_pipeline(
                f"PCA ({k}) + CNN",
                x_train_cnn,
                y_train,
                x_test_cnn,
                y_test,
                config["shape"],
                config["classes"],
                log_lines,
            )
            rows.append({"dataset": ds_name, **pca_metrics})

    log("", log_lines)
    log("Outputs written to:", log_lines)
    log(f"  - {TEXT_OUTPUT_PATH}", log_lines)
    log(f"  - {CSV_OUTPUT_PATH}", log_lines)
    write_outputs(log_lines, rows)


if __name__ == "__main__":
    main()
