import csv
import gc
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
TEXT_OUTPUT_PATH = OUTPUT_DIR / "last_experiment_results.txt"
CSV_OUTPUT_PATH = OUTPUT_DIR / "last_experiment_results.csv"

# ===============================
# CONFIGURATION
# ===============================
MAX_LEN = 100
D_BOTTLENECK = 128
K_VALUES = [200, 300, 500]


def log(message, log_lines):
    print(message)
    log_lines.append(message)


# ===============================
# OOM SAFE GENERATOR (For LSTM)
# ===============================
class SequenceGenerator(tf.keras.utils.Sequence):
    """Expands 2D features to 3D sequences ON-THE-FLY for the LSTM."""

    def __init__(self, x_data, y_data, max_len, batch_size=256):
        self.x_data = x_data
        self.y_data = y_data
        self.max_len = max_len
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x_data) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x_data[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y_data[idx * self.batch_size : (idx + 1) * self.batch_size]
        return np.repeat(batch_x[:, None, :], self.max_len, axis=1), batch_y


# ===============================
# MODEL BUILDERS
# ===============================
def build_lstm(input_dim):
    model = models.Sequential([
        layers.Input(shape=(MAX_LEN, input_dim)),
        layers.LSTM(64),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_ae(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    bottleneck = layers.Dense(D_BOTTLENECK, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(bottleneck)
    out = layers.Dense(input_dim)(x)

    ae = models.Model(inp, out)
    enc = models.Model(inp, bottleneck)
    ae.compile(optimizer="adam", loss="mse")
    return ae, enc


def evaluate_lstm(name, x_train_2d, x_test_2d, y_train, y_test, log_lines):
    log(f"  -> Training LSTM on {name}...", log_lines)
    model = build_lstm(x_train_2d.shape[1])
    train_gen = SequenceGenerator(x_train_2d, y_train, MAX_LEN)
    test_gen = SequenceGenerator(x_test_2d, y_test, MAX_LEN)

    start_train = time.time()
    model.fit(train_gen, epochs=5, verbose=0)
    train_time = time.time() - start_train

    start_eval = time.time()
    _, acc = model.evaluate(test_gen, verbose=0)
    eval_time = time.time() - start_eval

    log(
        f"  ✅ {name} Final Accuracy: {acc:.4f} | LSTM Train Time: {train_time:.2f}s | Eval Time: {eval_time:.2f}s\n",
        log_lines,
    )

    del model, train_gen, test_gen
    tf.keras.backend.clear_session()
    gc.collect()

    return {
        "pipeline": name,
        "accuracy": float(acc),
        "lstm_train_time_seconds": train_time,
        "evaluation_time_seconds": eval_time,
    }


def write_outputs(log_lines, rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_OUTPUT_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    with CSV_OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "pipeline",
                "pca_components",
                "autoencoder_bottleneck",
                "autoencoder_train_time_seconds",
                "accuracy",
                "lstm_train_time_seconds",
                "evaluation_time_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []
    rows = []

    log("Loading 20 Newsgroups (TF-IDF)...", log_lines)
    news = fetch_20newsgroups(subset="all")
    labels = (news.target < 10).astype(int)

    vec = TfidfVectorizer(max_features=5000, dtype=np.float32)
    x = vec.fit_transform(news.data).toarray()

    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = labels[:split], labels[split:]

    log("\n" + "=" * 50, log_lines)
    log(f"TRUE BASELINE: Pure PCA ({D_BOTTLENECK} dimensions)", log_lines)
    log("=" * 50, log_lines)

    pca_base = IncrementalPCA(n_components=D_BOTTLENECK)
    train_pca_base = pca_base.fit_transform(x_train)
    test_pca_base = pca_base.transform(x_test)
    baseline_metrics = evaluate_lstm(
        f"Pure PCA ({D_BOTTLENECK})",
        train_pca_base,
        test_pca_base,
        y_train,
        y_test,
        log_lines,
    )
    rows.append(
        {
            "pca_components": D_BOTTLENECK,
            "autoencoder_bottleneck": "",
            "autoencoder_train_time_seconds": "",
            **baseline_metrics,
        }
    )

    log("=" * 50, log_lines)
    log(f"CASCADED PIPELINES (Targeting {D_BOTTLENECK} dimensions)", log_lines)
    log("=" * 50, log_lines)

    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    for k in K_VALUES:
        log(f"--- Testing PCA({k}) -> AE({D_BOTTLENECK}) ---", log_lines)

        pca = IncrementalPCA(n_components=k)
        train_pca_k = pca.fit_transform(x_train)
        test_pca_k = pca.transform(x_test)

        log(f"  -> Training Autoencoder (k={k} to d={D_BOTTLENECK})...", log_lines)
        ae, enc = build_ae(k)

        ae_start = time.time()
        ae.fit(
            train_pca_k,
            train_pca_k,
            epochs=100,
            batch_size=256,
            validation_split=0.2,
            callbacks=[early_stopper],
            verbose=0,
        )
        ae_train_time = time.time() - ae_start

        train_ae_final = enc.predict(train_pca_k, batch_size=1024, verbose=0)
        test_ae_final = enc.predict(test_pca_k, batch_size=1024, verbose=0)

        pipeline_metrics = evaluate_lstm(
            f"PCA({k}) + AE({D_BOTTLENECK})",
            train_ae_final,
            test_ae_final,
            y_train,
            y_test,
            log_lines,
        )
        rows.append(
            {
                "pca_components": k,
                "autoencoder_bottleneck": D_BOTTLENECK,
                "autoencoder_train_time_seconds": ae_train_time,
                **pipeline_metrics,
            }
        )

        del ae, enc, train_pca_k, test_pca_k, train_ae_final, test_ae_final
        tf.keras.backend.clear_session()
        gc.collect()

    log("Experiment Complete!", log_lines)
    log("", log_lines)
    log("Outputs written to:", log_lines)
    log(f"  - {TEXT_OUTPUT_PATH}", log_lines)
    log(f"  - {CSV_OUTPUT_PATH}", log_lines)
    write_outputs(log_lines, rows)


if __name__ == "__main__":
    main()
