# Error-analysis highlights

These results use the grouped validation split (2,002 rows), not the held-out test set.

- Lowest error rate: **NLI Evidence** (51 errors, 51.00%).
- Highest error rate: **Cross Encoder** (59 errors, 59.00%).
- Hardest shared sample: `truthfulqa-gen-00000` was missed by 10 of 10 baselines.
- 46 samples were missed by every baseline; 54 were missed by at least eight baselines.
- Strongest false-positive bias: **Lexical SVM** (54 FP vs. 0 FN).
- Strongest false-negative bias: **Cross Encoder** (7 FN vs. 52 FP).
- The shared hardest examples are mostly short, plausible entities or close semantic alternatives copied from the evidence. This suggests a dataset-level hard-negative/annotation ambiguity that should be discussed in the paper.
- “Relative confidence” is the percentile rank of distance from each model's decision boundary. It permits cross-model comparison without pretending that unbounded SVM/RAG scores are calibrated probabilities.

See `hardest_shared_errors.csv` and the per-baseline top-k tables for qualitative review.
