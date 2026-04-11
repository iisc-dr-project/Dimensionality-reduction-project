# Paper Claims Mapped to a Shared Benchmark

## 1. Random Projection in Dimensionality Reduction

Operational claim for this project:
Random projection should preserve enough of the original geometry that pairwise relationships and neighborhood structure remain usable after dimensionality reduction, while remaining computationally lightweight.

What we measure:

- pairwise distance distortion
- rank correlation between sampled high-dimensional and reduced-space distances
- k-nearest-neighbor overlap before and after projection
- downstream linear-probe performance on reduced features

Why this fits the paper:
The paper focuses on image and text data, compares random projection to other reduction methods, and emphasizes approximation quality with lower computational cost.

## 2. Visualizing Data using t-SNE

Operational claim for this project:
t-SNE should generate stronger two-dimensional visualizations than older baselines by preserving local neighborhoods and producing clearer cluster structure.

What we measure:

- trustworthiness
- continuity
- neighborhood hit
- silhouette score
- class-consistency in the 2D embedding

Why this fits the paper:
The paper explicitly claims better visualization quality, better local structure preservation, and clearer multi-scale structure than earlier techniques.

## 3. Gaussian Mixture Variational Autoencoder with Contrastive Learning

Operational claim for this project:
C-GMVAE should give strong multi-label prediction performance, leverage a more meaningful multimodal latent space than simpler baselines, and stay competitive even when trained with only part of the data.

What we measure:

- micro-F1
- macro-F1
- hamming loss
- average precision
- label ranking loss
- 50% versus 100% training-fraction comparisons

Why this fits the paper:
The paper claims strong multi-label performance, useful feature and label embeddings, and competitive performance with only half of the training data.

## Common Evaluation Principle

All three papers are translated into the same experiment language:

1. Preserve structure when reducing dimension.
2. Preserve local neighborhoods in 2D visualizations.
3. Preserve predictive utility after reduction or latent modeling.
