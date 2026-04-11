# Raw Data Dependencies

This folder stores the non-Python data dependencies required to run the benchmark and reproduction pipelines end to end.

## Local datasets required on disk

### MNIST

Required by:

- common benchmark
- random projection reproduction
- t-SNE reproduction

Expected files:

- `mnist/train-images-idx3-ubyte`
- `mnist/train-labels-idx1-ubyte`
- `mnist/t10k-images-idx3-ubyte`
- `mnist/t10k-labels-idx1-ubyte`

### COIL-20

Required by:

- t-SNE reproduction

Expected files:

- `coil20/*.png`

Filename requirement:

- filenames must contain the object id
- examples:
  - `obj1__0.png`
  - `obj01__000.png`

## Datasets fetched through scikit-learn

These are not stored in this repository by default, but some configs depend on them and scikit-learn will try to download/cache them when the experiment runs.

### 20 Newsgroups

Required by:

- common benchmark
- random projection reproduction

Loader:

- `fetch_20newsgroups`

### Olivetti faces

Required by:

- t-SNE reproduction

Loader:

- `fetch_olivetti_faces`

## Summary by experiment track

### Common benchmark

Needs:

- local MNIST
- scikit-learn download access for 20 Newsgroups

### Random projection reproduction

Needs:

- local MNIST
- scikit-learn download access for 20 Newsgroups

### t-SNE reproduction

Needs:

- local MNIST
- local COIL-20
- scikit-learn download access for Olivetti faces

### C-GMVAE reproduction

Needs:

- no external datasets beyond the Python environment
- current config uses synthetic datasets generated in code

## Note

`requirements.txt` can list Python packages, but not datasets. This file is the authoritative data-dependency manifest for the project.
