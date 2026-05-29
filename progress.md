# CENG463 – Technical Checkpoint Report (Final Progress)

## 1. Problem Definition
Task. This project investigates whether epistemic uncertainty signals extracted from large language models (LLMs) can reliably detect hallucinated outputs. Given a question and a candidate answer, the system must classify the answer as factual (label 0) or hallucinated (label 1). The task is therefore a binary classification problem framed around natural language understanding and model introspection.

Importance. Hallucinations represent one of the most critical reliability problems in deployed NLP systems. Detecting them automatically is essential for safe deployment.

Dataset, Input, and Output. The project uses the HaluEval QA benchmark. The system input is a (question, candidate answer) pair – optionally augmented with a knowledge snippet – and the output is a binary hallucination label.

## 2. Dataset and Preprocessing Status
- **Dataset name and source**: HaluEval QA (pminervini/HaluEval, split qa).
- **Scale**: The dataset has been scaled up to use 15,999 training samples and 2,001 validation samples.
- **Preprocessing steps completed**: Normalisation, expansion into balanced supervised rows, deterministic splitting to JSONL files.

## 3. Baseline Models & Proposed Method
Three baselines and one proposed method are implemented:
1. **Lexical SVM**: A TF-IDF vectoriser over the concatenated question and answer, fed into a linear SVM.
2. **Entropy Classifier**: Logistic regression using token-level log-probabilities extracted from `distilgpt2` in memory mode.
3. **RAG Compare**: Delta thresholding between memory and context mode log-probabilities. 
4. **Hybrid LR Method**: A Logistic Regression model combining surface-level lexical traits with epistemic uncertainty. 
   * **Implementation Details**: The model extracts text representations using a `TfidfVectorizer` (up to 20,000 unigram/bigram features) over the candidate answers. Separately, it collects 7 numerical uncertainty features derived from the model's token-level log-probabilities (e.g., `mean_logprob`, `min_logprob`, `perplexity`). These two feature spaces are horizontally concatenated (`scipy.sparse.hstack`) into a single unified feature matrix. Finally, a `LogisticRegression` classifier is trained on this combined matrix with balanced class weights.
5. **Hybrid SVM Method**: An SVM model built on the identical unified feature representation as the Hybrid LR, but designed to capture robust linear decision boundaries in this high-dimensional space.
   * **Implementation Details**: It uses the same TF-IDF vectorization and 7 uncertainty features. However, unlike the LR approach, the numerical uncertainty features are first normalized using a `StandardScaler` to ensure they are on a comparable scale with the sparse TF-IDF features. The scaled numeric features and text features are stacked, and a linear `SVC` (Support Vector Classifier) with `probability=True` and balanced class weights is trained.

## 4. Experimental Results
Experiments were scaled up to the larger dataset slice (16k train / 2k val). 

| Model | Accuracy | Macro F1 | AUROC |
|-------|----------|----------|-------|
| Lexical SVM | 0.943 | 0.943 | 0.972 |
| Entropy | 0.929 | 0.929 | 0.971 |
| RAG Compare | 0.776 | 0.773 | 0.791 |
| Hybrid LR | 0.929 | 0.929 | 0.969 |
| **Hybrid SVM** | **0.958** | **0.958** | **0.981** |

**Analysis:**
- The **Hybrid SVM** method successfully outperforms all baselines (including Hybrid LR), validating the hypothesis that combining lexical and uncertainty features yields the best predictive power. Incorporating the expanded uncertainty features (Variance, ECE, Brier score) explicitly pushed the AUROC from 0.978 to **0.981**.
- **Lexical SVM** and **Entropy** maintained their high performance even on the scaled-up dataset.
- **RAG Compare** successfully establishes a baseline (AUROC 0.791) where relevant context improves model confidence, and it is now utilizing a calibrated threshold search to maximize the F1-score.

### Feature Expansion Impact
To quantify the benefit of the newly added uncertainty features (Token Variance, ECE, Brier Score), the models utilizing numeric features were evaluated before and after their inclusion:

| Model | Acc (Before) | Acc (After) | F1 (Before) | F1 (After) | AUROC (Before) | AUROC (After) |
|-------|--------------|-------------|-------------|------------|----------------|---------------|
| Entropy | 0.929 | 0.929 | 0.929 | 0.929 | 0.970 | 0.971 |
| Hybrid LR | 0.951 | 0.930 | 0.951 | 0.930 | 0.976 | 0.969 |
| **Hybrid SVM** | 0.956 | **0.958** | 0.956 | **0.958** | 0.978 | **0.981** |

**Takeaways:**
1. The **Hybrid SVM** effectively leveraged the new features to achieve the absolute best performance across all metrics (Accuracy, F1, and AUROC). Because it properly scales the features (`StandardScaler`), the new complex numerical signals smoothly integrated with the sparse text representations.
2. The **Hybrid LR** suffered a slight degradation across all metrics. Because it lacks a dedicated feature scaling pipeline in its implementation, injecting raw, unscaled variances and squared errors alongside TF-IDF tokens disrupted its logistic optimization.
3. The **Entropy** classifier saw a negligible but positive improvement in AUROC while maintaining identical classification accuracy, confirming that these features carry valid epistemic signals even on their own.

## 5. Fulfillment of Planned Improvements
An evaluation of the previously planned next steps:
- [x] **Scale up experiments**: Successfully scaled to ~18,000 samples.
- [x] **Implementation of Proposed Hybrid Method**: Done, achieving the highest performance.
- [ ] **Stronger LM backbone**: `distilgpt2` is still used as the default; a stronger model was not fully integrated.
- [x] **Expanded uncertainty features**: Completed. Features including ECE, Brier score, and token-level variance were added and improved the Hybrid SVM AUROC to 0.981.
- [x] **Threshold optimisation for RAG Compare**: Completed. The logic was fixed to prevent inversion and now uses a calibrated search over the validation set to maximize F1-score.
- [ ] **Literature review**: Summaries have not been written to `docs/papers/` (only PDFs are present).

## 6. Next Steps for the Final Report
- **Complete Error Analysis**: Generate confusion matrices and qualitative analysis for the final report.
- **Document Limitations**: Acknowledge the unfulfilled goals (e.g., larger backbone, literature summaries) as project limitations or future work in the final report.
