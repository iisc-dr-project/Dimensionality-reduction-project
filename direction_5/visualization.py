import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.random_projection import GaussianRandomProjection

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
TEXT_OUTPUT_PATH = OUTPUT_DIR / "visualization_results.txt"
CSV_OUTPUT_PATH = OUTPUT_DIR / "visualization_results.csv"
PLOT_OUTPUT_PATH = OUTPUT_DIR / "visualization_comparison.png"


def log(message, log_lines):
    print(message)
    log_lines.append(message)


def write_outputs(log_lines, results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_OUTPUT_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    with CSV_OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["pipeline", "time_seconds", "silhouette"],
        )
        writer.writeheader()
        for name, metrics in results.items():
            writer.writerow(
                {
                    "pipeline": name,
                    "time_seconds": metrics["time"],
                    "silhouette": metrics["silhouette"],
                }
            )


def save_plot(results, y_numeric):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Dimensionality Reduction Pipeline Comparison", fontsize=16)

    axes = axes.flatten()
    scatter = None
    for index, (name, metrics) in enumerate(results.items()):
        ax = axes[index]
        scatter = ax.scatter(
            metrics["data"][:, 0],
            metrics["data"][:, 1],
            c=y_numeric,
            cmap="tab10",
            s=5,
            alpha=0.7,
        )
        title = f"{name}\nTime: {metrics['time']:.2f}s | Silhouette: {metrics['silhouette']:.3f}"
        ax.set_title(title)
        ax.axis("off")

    if scatter is not None:
        legend = axes[3].legend(
            *scatter.legend_elements(),
            title="Digits",
            loc="upper right",
            bbox_to_anchor=(1.3, 1),
        )
        axes[3].add_artist(legend)

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []
    results = {}

    log("Loading dataset...", log_lines)
    x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
    x, y = x[:3000] / 255.0, y[:3000]
    y_numeric = y.astype(int)

    pca_components = 100
    tsne_perplexity = 30

    log("Running Pipeline 1: PCA -> 2D", log_lines)
    start_time = time.time()
    pca_2d = PCA(n_components=2).fit_transform(x)
    results["PCA 2D"] = {
        "data": pca_2d,
        "time": time.time() - start_time,
        "silhouette": silhouette_score(pca_2d, y_numeric),
    }

    log("Running Pipeline 2: t-SNE -> 2D", log_lines)
    start_time = time.time()
    tsne_2d = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42).fit_transform(x)
    results["t-SNE 2D"] = {
        "data": tsne_2d,
        "time": time.time() - start_time,
        "silhouette": silhouette_score(tsne_2d, y_numeric),
    }

    log(f"Running Pipeline 3: PCA ({pca_components}) -> t-SNE", log_lines)
    start_time = time.time()
    x_pca = PCA(n_components=pca_components).fit_transform(x)
    pca_tsne_2d = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42).fit_transform(x_pca)
    results["PCA + t-SNE"] = {
        "data": pca_tsne_2d,
        "time": time.time() - start_time,
        "silhouette": silhouette_score(pca_tsne_2d, y_numeric),
    }

    log(f"Running Pipeline 4: Random Projection ({pca_components}) -> t-SNE", log_lines)
    start_time = time.time()
    x_rp = GaussianRandomProjection(n_components=pca_components, random_state=42).fit_transform(x)
    rp_tsne_2d = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42).fit_transform(x_rp)
    results["RP + t-SNE"] = {
        "data": rp_tsne_2d,
        "time": time.time() - start_time,
        "silhouette": silhouette_score(rp_tsne_2d, y_numeric),
    }

    log("", log_lines)
    log("=" * 80, log_lines)
    log(f"{'Pipeline':<20} | {'Time (s)':<10} | {'Silhouette':<10}", log_lines)
    log("-" * 80, log_lines)
    for name, metrics in results.items():
        log(f"{name:<20} | {metrics['time']:<10.2f} | {metrics['silhouette']:<10.4f}", log_lines)
    log("=" * 80, log_lines)

    save_plot(results, y_numeric)

    log("", log_lines)
    log("Outputs written to:", log_lines)
    log(f"  - {TEXT_OUTPUT_PATH}", log_lines)
    log(f"  - {CSV_OUTPUT_PATH}", log_lines)
    log(f"  - {PLOT_OUTPUT_PATH}", log_lines)
    write_outputs(log_lines, results)


if __name__ == "__main__":
    main()
