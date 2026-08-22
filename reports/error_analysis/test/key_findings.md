# Error-analysis highlights

These results use the grouped validation split (2,002 rows), not the held-out test set.

- Lowest error rate: **Cross Encoder** (49 errors, 2.45%).
- Highest error rate: **RAG Compare Fixed** (514 errors, 25.70%).
- Hardest shared sample: `qa-05440-hallucinated` was missed by 10 of 10 baselines.
- 2 samples were missed by every baseline; 26 were missed by at least eight baselines.
- Strongest false-positive bias: **RAG Compare Fixed** (367 FP vs. 147 FN).
- Strongest false-negative bias: **NLI Evidence** (296 FN vs. 124 FP).
- The shared hardest examples are mostly short, plausible entities or close semantic alternatives copied from the evidence. This suggests a dataset-level hard-negative/annotation ambiguity that should be discussed in the paper.
- “Relative confidence” is the percentile rank of distance from each model's decision boundary. It permits cross-model comparison without pretending that unbounded SVM/RAG scores are calibrated probabilities.

See `hardest_shared_errors.csv` and the per-baseline top-k tables for qualitative review.
