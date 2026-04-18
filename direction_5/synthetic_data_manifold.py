import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
TEXT_OUTPUT_PATH = OUTPUT_DIR / "synthetic_data_manifold_results.txt"
CSV_OUTPUT_PATH = OUTPUT_DIR / "synthetic_data_manifold_results.csv"
PLOT_OUTPUT_PATH = OUTPUT_DIR / "synthetic_data_manifold_plots.png"


def log(message, log_lines):
    print(message)
    log_lines.append(message)


class CGMVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10, n_clusters=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.fc_logits = nn.Linear(64, n_clusters)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        logits = self.fc_logits(hidden)
        z = self.reparameterize(mu, logvar)
        y = F.gumbel_softmax(logits, tau=1.0, hard=False)
        return self.decoder(z), mu, logvar, y


def train_cgmvae(data, input_dim, latent_dim=10, n_clusters=5, epochs=400):
    model = CGMVAE(input_dim, latent_dim, n_clusters)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x_tensor = torch.FloatTensor(data)

    for _ in range(epochs):
        optimizer.zero_grad()
        recon, mu, logvar, y = model(x_tensor)
        loss = F.mse_loss(recon, x_tensor) + 0.005 * (
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.shape[0]
        )
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, mu, _, y = model(x_tensor)
    return mu.numpy(), torch.argmax(y, dim=1).numpy()


def benchmark_method(name, data_in, method_func, results, y_true, n_clusters_true, log_lines):
    log(f"Running {name}...", log_lines)
    start = time.time()
    coords = method_func(data_in)
    y_pred = KMeans(n_clusters=n_clusters_true, n_init=10).fit_predict(coords)
    ari = adjusted_rand_score(y_true, y_pred)
    results[name] = (ari, time.time() - start, coords)


def save_plot(results, y_true):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    plot_order = [
        "Pure PCA",
        "Pure t-SNE",
        "Pure UMAP",
        "Cascade: PCA -> t-SNE",
        "Cascade: PCA -> UMAP",
        "Cascade: PCA -> C-GMVAE",
    ]

    for index, name in enumerate(plot_order):
        if name in results:
            coords = results[name][2]
            ari = results[name][0]
            axes[index].scatter(coords[:, 0], coords[:, 1], c=y_true, cmap="tab10", s=3, alpha=0.5)
            axes[index].set_title(f"{name}\nARI: {ari:.3f}")
            axes[index].set_xticks([])
            axes[index].set_yticks([])

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(log_lines, results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_OUTPUT_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    with CSV_OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["method", "ari_score", "time_seconds"],
        )
        writer.writeheader()
        for name in sorted(results, key=lambda item: results[item][0], reverse=True):
            ari, duration, _ = results[name]
            writer.writerow(
                {
                    "method": name,
                    "ari_score": float(ari),
                    "time_seconds": duration,
                }
            )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []
    results = {}

    log("Generating 110D Noisy Dataset...", log_lines)
    n_clusters_true = 5
    x_sig, y_true = make_blobs(
        n_samples=2000,
        n_features=10,
        centers=n_clusters_true,
        cluster_std=1.2,
        random_state=42,
    )
    x_noise = np.random.normal(0, 5, size=(2000, 100))
    x_raw = StandardScaler().fit_transform(np.hstack([x_sig, x_noise]))

    benchmark_method(
        "Pure PCA",
        x_raw,
        lambda x: PCA(n_components=2).fit_transform(x),
        results,
        y_true,
        n_clusters_true,
        log_lines,
    )
    benchmark_method(
        "Pure t-SNE",
        x_raw,
        lambda x: TSNE(n_components=2).fit_transform(x),
        results,
        y_true,
        n_clusters_true,
        log_lines,
    )
    benchmark_method(
        "Pure UMAP",
        x_raw,
        lambda x: umap.UMAP(n_components=2).fit_transform(x),
        results,
        y_true,
        n_clusters_true,
        log_lines,
    )

    pca_pre = PCA(n_components=25)
    x_cleaned = pca_pre.fit_transform(x_raw)

    benchmark_method(
        "Cascade: PCA -> t-SNE",
        x_cleaned,
        lambda x: TSNE(n_components=2).fit_transform(x),
        results,
        y_true,
        n_clusters_true,
        log_lines,
    )
    benchmark_method(
        "Cascade: PCA -> UMAP",
        x_cleaned,
        lambda x: umap.UMAP(n_components=2).fit_transform(x),
        results,
        y_true,
        n_clusters_true,
        log_lines,
    )

    log("Running Cascade: PCA -> C-GMVAE...", log_lines)
    start_ae = time.time()
    latent, y_pred_ae = train_cgmvae(x_cleaned, input_dim=25)
    results["Cascade: PCA -> C-GMVAE"] = (
        adjusted_rand_score(y_true, y_pred_ae),
        time.time() - start_ae,
        latent,
    )

    log("", log_lines)
    log("=" * 80, log_lines)
    log(f"{'Method Architecture':<30} | {'ARI Score':<12} | {'Time (s)':<8}", log_lines)
    log("-" * 80, log_lines)
    for name in sorted(results, key=lambda item: results[item][0], reverse=True):
        ari, duration, _ = results[name]
        log(f"{name:<30} | {ari:<12.4f} | {duration:<8.2f}", log_lines)
    log("=" * 80, log_lines)

    save_plot(results, y_true)

    log("", log_lines)
    log("Outputs written to:", log_lines)
    log(f"  - {TEXT_OUTPUT_PATH}", log_lines)
    log(f"  - {CSV_OUTPUT_PATH}", log_lines)
    log(f"  - {PLOT_OUTPUT_PATH}", log_lines)
    write_outputs(log_lines, results)


if __name__ == "__main__":
    main()
