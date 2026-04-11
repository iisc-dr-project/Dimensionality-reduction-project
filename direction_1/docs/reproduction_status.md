# Reproduction Status

## Goal Split

This repository intentionally separates:

- `common benchmark`
- `paper reproduction`

The common benchmark asks whether all methods can be compared under one shared evaluation language.

The paper-reproduction tracks ask whether the original papers' experiment structures and claims can be recreated more directly.

## Random Projection

### Common benchmark status

Implemented.

The shared benchmark already includes:

- Gaussian random projection
- sparse random projection
- shared structure-preservation metrics
- shared downstream evaluation

### Paper-reproduction status

Implemented as a separate pipeline.

The reproduction track now includes:

- image clean condition
- image noisy condition
- text condition
- dimension sweeps over target dimension
- RP, SRP, PCA, DCT, and NMF for noisy-image comparison
- per-seed runs
- aggregated mean/std tables
- reconstruction, structure, runtime, and estimated-cost outputs

### Remaining gaps

- The exact original paper datasets are not matched yet.
- `MF` in the paper is approximated here with `NMF`, which is a reasonable matrix-factorization proxy but not a claim of exact paper identity.
- Confidence-interval reporting is currently mean/std tables and plots, not explicit 95% CI bands.

## C-GMVAE

### Common benchmark status

Implemented.

The shared benchmark includes:

- C-GMVAE
- VAE baseline
- multilabel MLP baseline
- data-fraction comparisons
- shared multilabel metrics

### Paper-reproduction status

Partially implemented as a separate configuration on top of the shared multilabel pipeline.

The reproduction track now includes:

- multiple seeds
- ablations:
  - full C-GMVAE
  - no contrastive loss
  - no mixture prior
- 10%, 50%, and 100% training-fraction experiments
- paper-style metrics such as example-F1 and precision@k

### Remaining gaps

- Original datasets are not present:
  - eBird
  - MIRFLICKR
  - NUS-WIDE
- Original comparison baselines are not present:
  - MPVAE
  - LaMP
  - ASL
  - RBCC
- The current reproduction is therefore structurally aligned, but not yet fully paper-faithful.

## t-SNE

### Common benchmark status

Implemented.

### Paper-reproduction status

Implemented as a separate pipeline.

The reproduction track now includes:

- MNIST
- Olivetti faces
- COIL-20 via local staged images
- PCA to 30D before 2D embedding
- t-SNE, Isomap, LLE, and Sammon mapping
- per-method scatterplot generation

### Remaining gaps

- The t-SNE implementation uses scikit-learn, so the paper's perplexity, iterations, early exaggeration, and learning rate are matched, but the exact momentum schedule is not directly controllable.
- COIL-20 must be staged locally under `data/raw/coil20`.
- The Sammon implementation is a practical in-repo optimizer, not a claim of exact code-level equivalence to the paper's original Newton implementation.
