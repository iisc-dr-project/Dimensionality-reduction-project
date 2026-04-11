# Direction 1: Common Benchmarks and Paper Reproductions

This repository has two explicit experiment tracks:

1. `common benchmark`
   One shared evaluation framework across random projection, t-SNE, and C-GMVAE.

2. `paper reproductions`
   Separate, paper-shaped experiment pipelines that try to mirror each paper's original setup, claims, and plots more directly.

A shared benchmark is for the objective of comparing the methods under common metric standards while paper reproduction is to scrutinize if the original claims of the three papers survive when we redo the paper’s experiments by ourselves. 

## Repository Layout

- `configs/common/`
  Common-benchmark configurations.
- `configs/reproductions/`
  Paper-specific reproduction configurations.
- `scripts/run_common_benchmark.py`
  Main entry point for the shared benchmark.
- `scripts/run_random_projection_reproduction.py`
  Entry point for the random projection reproduction pipeline.
- `scripts/run_cgmvae_reproduction.py`
  Entry point for the C-GMVAE reproduction configuration.
- `scripts/run_tsne_reproduction.py`
  Entry point for the t-SNE reproduction pipeline.
- `src/`
  Reusable shared-benchmark package.
- `src/reproductions/`
  Paper-specific reproduction pipelines kept separate from the common benchmark.
- `outputs/direction_1_run/`
  Shared benchmark outputs.
- `outputs/reproductions/`
  Paper-reproduction outputs.

## Common Benchmark

The common benchmark translates all three papers into one evaluation language:

- distance preservation after reduction
- neighborhood preservation in low-dimensional visualizations
- downstream predictive utility

Run it with:

```bash
python direction_1/scripts/run_common_benchmark.py
```

The default config is:

```text
direction_1/configs/common/benchmark_config.json
```

This track is for:

- comparing RP, PCA, autoencoders, t-SNE, and UMAP under one set of metrics
- checking shared qualitative claims

## Paper Reproductions

### Random Projection


Run it with:

```bash
python direction_1/scripts/run_random_projection_reproduction.py
```

The default config is:

```text
direction_1/configs/reproductions/random_projection_reproduction.json
```

Current reproduction outputs include:

- per-seed projection runs
- aggregated mean/std tables across seeds
- error and runtime curve plots

### C-GMVAE

Run it with:

```bash
python direction_1/scripts/run_cgmvae_reproduction.py
```

The default config is:

```text
direction_1/configs/reproductions/cgmvae_reproduction.json
```

Current reproduction config includes:

- `cgmvae_full`
- `cgmvae_no_contrastive`
- `cgmvae_no_mixture`
- 10%, 50%, and 100% training-fraction runs
- multiple seeds

### t-SNE

Run it with:

```bash
python direction_1/scripts/run_tsne_reproduction.py
```

The default config is:

```text
direction_1/configs/reproductions/tsne_reproduction.json
```

Current reproduction outputs include:

- per-method scatterplots after PCA-to-30D preprocessing
- per-run timing rows
- aggregated timing summaries

The implementation uses scikit-learn for t-SNE, so perplexity, iterations, early exaggeration, and learning rate are matched directly, while the paper's exact momentum schedule is only approximated.

## Status

The repository supports both objectives structurally:

- `common benchmarking`
- `paper-specific reproduction`

For random projection, both tracks now exist in code.

For C-GMVAE, both tracks now exist in code and config, but a fully paper-faithful reproduction still requires the original datasets and external baselines such as eBird, MIRFLICKR, NUS-WIDE, MPVAE, and LaMP.


## Setup Notes

1. Install the project requirements.
2. Stage local datasets into `data/raw/`.
3. Choose whether you want:
   - the shared benchmark
   - a paper-specific reproduction run
4. Run the corresponding script.
5. Use `reports/claim_verification_template.md` and the outputs in `outputs/` to build the final write-up.
