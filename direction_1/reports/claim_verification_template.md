# Direction 1 Claim Verification Template

Use this file after running the experiments and filling in the generated numbers from `outputs/`.

## Research Question

Can the main empirical claims of the three assigned papers be recovered under a shared benchmark?

## Dataset Summary

- Visualization datasets:
- Text datasets:
- Multi-label datasets:

## Random Projection

### Expected trend

Random projection should preserve structure better than aggressive truncation while remaining lightweight.

### Evidence from the benchmark

- Pairwise distance distortion:
- Distance rank correlation:
- k-nearest-neighbor overlap:
- Downstream classification:

### Conclusion

Claim verified / partially verified / not verified:

## t-SNE

### Expected trend

t-SNE should outperform older visualization baselines on local-neighborhood preservation and cluster clarity in 2D.

### Evidence from the benchmark

- Trustworthiness:
- Continuity:
- Neighborhood hit:
- Silhouette score:
- Class-consistency:

### Conclusion

Claim verified / partially verified / not verified:

## C-GMVAE

### Expected trend

C-GMVAE should be competitive or better on multi-label metrics and remain strong when using only part of the training set.

### Evidence from the benchmark

- Micro-F1:
- Macro-F1:
- Hamming loss:
- Average precision:
- Label ranking loss:
- 50% versus 100% training comparison:

### Conclusion

Claim verified / partially verified / not verified:

## Final Overall Answer

Summarize whether a shared benchmark recovered the expected claim trends across all three papers.
