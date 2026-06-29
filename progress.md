# CENG463 Technical Progress Report

## 1. Problem Definition

The project studies whether lexical artifacts, model uncertainty, semantic similarity, and external evidence can detect hallucinated LLM answers. Given a question and candidate answer, the system predicts:

- `0`: factual
- `1`: hallucinated

The optional knowledge passage is used by context-sensitive and evidence-grounded methods.

## 2. Dataset and Preprocessing

- **Dataset:** HaluEval QA (`pminervini/HaluEval`, `qa`)
- **Construction:** Each original QA item produces one factual and one hallucinated row.
- **Split policy:** Grouped by `original_sample_id`; paired answers cannot cross splits.
- **Balance:** Every split contains equal numbers of factual and hallucinated rows.

| Split | Original questions | Rows |
|---|---:|---:|
| Train | 7,999 | 15,998 |
| Validation | 1,001 | 2,002 |
| Test | 1,000 | 2,000 |

The grouped split replaces the earlier row-level split, which allowed paired-question leakage.

## 3. Ten-Baseline Matrix

### Shared Feature Definitions

The baselines are organized around four information sources:

- **Lexical signal:** Surface words and short phrases represented with TF-IDF.
- **Uncertainty signal:** Statistics calculated from the causal language model's token log-probabilities for the candidate answer.
- **Semantic signal:** Dense transformer embeddings representing the meanings of the question and answer.
- **Evidence signal:** Measurements of whether the provided knowledge supports the candidate answer.

The uncertainty-based models use the following ten numerical features:

| Feature | Description |
|---|---|
| `negative_mean_logprob` | Mean token surprisal; larger values mean the LM assigned lower probability to the answer |
| `mean_logprob` | Mean log-probability of the candidate-answer tokens |
| `sum_logprob` | Sum of token log-probabilities across the answer |
| `min_logprob` | Lowest token log-probability in the answer |
| `max_logprob` | Highest token log-probability in the answer |
| `perplexity` | Exponential transformation of negative mean log-probability |
| `token_count` | Number of scored answer tokens |
| `token_variance` | Variance of token log-probabilities |
| `ece` | Mean token-confidence error heuristic used by the current implementation |
| `brier_score` | Mean squared token-confidence error heuristic used by the current implementation |

The current `ece` and `brier_score` fields are per-answer token-confidence heuristics. They are not the standard dataset-level calibration metrics calculated from final classifier predictions.

### Group 1: Lexical and Artifact Signals

1. **Lexical SVM (`lexical_svm`)**
   - **Input:** `candidate_answer` only. It deliberately excludes the question, knowledge passage, and LM uncertainty features.
   - **Text representation:** `TfidfVectorizer` converts the training answers into a sparse matrix of unigram and bigram weights.
   - **Vectorizer settings:** Maximum 20,000 features, `ngram_range=(1, 2)`, and `min_df=1`.
   - **Classifier:** Linear-kernel `SVC` with balanced class weights and probability estimation enabled.
   - **Output:** The predicted class and `P(hallucinated)` from `predict_proba`.
   - **Experimental role:** This is the artifact anchor. High performance indicates that answer wording, length, or construction style alone separates the HaluEval classes.

2. **Lexical Hybrid SVM (`lexical_hybrid_svm`)**
   - **Inputs:** `candidate_answer` plus the ten uncertainty features extracted in memory mode.
   - **Lexical branch:** Uses the same 20,000-feature unigram/bigram TF-IDF representation as Lexical SVM.
   - **Uncertainty branch:** The ten numerical features are standardized with `StandardScaler`, fitted only on training data.
   - **Fusion:** The sparse TF-IDF matrix and standardized numerical matrix are concatenated horizontally. The resulting representation contains approximately 20,010 features.
   - **Classifier:** Linear-kernel `SVC` with balanced class weights, probability estimation, and a 5,000-iteration limit.
   - **Output:** Predicted label and probability-based hallucination score.
   - **Experimental role:** Directly measures whether causal-LM confidence adds useful information beyond lexical artifacts.
   - **Current limitation:** The validation run reached the iteration limit, so its convergence status must be reported with its metrics.

### Group 2: Generative Uncertainty

3. **Base Entropy Classifier (`entropy_base`)**
   - **Input:** Question and candidate answer during feature extraction, but no TF-IDF or semantic embedding features are given to the classifier.
   - **Scoring prompt:** `distilgpt2` receives the question in memory mode and scores the supplied candidate-answer tokens through teacher forcing.
   - **Representation:** The token log-probabilities are reduced to the ten numerical uncertainty features listed above.
   - **Preprocessing:** All ten features are standardized using a training-fitted `StandardScaler`.
   - **Classifier:** Logistic regression with balanced class weights and up to 5,000 optimization iterations.
   - **Output:** Predicted label and logistic-regression probability for the hallucinated class.
   - **Experimental role:** Tests whether model confidence alone can identify hallucinations without access to lexical or semantic text representations.

4. **Upgraded Entropy Classifier (`entropy_upgraded`)**
   - **Input and classifier:** Uses exactly the same ten-feature representation, scaling pipeline, and logistic-regression classifier as Entropy Base.
   - **Backbone change:** Replaces `distilgpt2` with an instruction-tuned causal LM. The default integrated checkpoint is `Qwen/Qwen2.5-1.5B-Instruct`.
   - **Prompt formatting:** The tokenizer's chat template can be applied so answer probabilities are conditioned on the model's intended instruction format.
   - **Compute support:** The scorer supports CPU/GPU selection and `float32`, `float16`, or `bfloat16` model loading.
   - **Output:** Predicted label and probability of hallucination.
   - **Experimental role:** Isolates the effect of the uncertainty backbone. Because Baselines 3 and 4 share the same classifier, their difference measures whether instruction tuning and stronger language modeling produce more factual confidence signals.
   - **Completed run:** Training and validation features were generated with `Qwen/Qwen2.5-1.5B-Instruct`, and the logistic-regression classifier converged in 28 iterations.
   - **Observed result:** The upgraded backbone did not improve detection performance. It reached 0.9231 accuracy, 0.9231 Macro F1, and 0.9654 AUROC, all below the `distilgpt2` entropy baseline.

### Group 3: Semantic TF-IDF Replacements

5. **Semantic SVM (`semantic_svm`)**
   - **Inputs:** `question` and `candidate_answer`; knowledge and uncertainty features are excluded.
   - **Encoder:** A frozen transformer bi-encoder, by default `sentence-transformers/all-MiniLM-L6-v2`.
   - **Pooling:** Question and answer are encoded separately. Token hidden states are attention-mask mean pooled and L2 normalized.
   - **Interaction representation:** If `q` and `a` are the question and answer embeddings, the final vector is:

     ```text
     [q, a, |q-a|, q*a, cosine(q,a)]
     ```

   - **Feature dimension:** MiniLM produces 384-dimensional embeddings, so the interaction vector has `4 * 384 + 1 = 1,537` features.
   - **Classifier:** `LinearSVC` with balanced class weights and a 20,000-iteration limit.
   - **Output:** Predicted label and the SVM decision function as the hallucination score.
   - **Experimental role:** Replaces TF-IDF with meaning-based features and tests whether semantic relationships outperform word-frequency artifacts.

6. **Semantic Hybrid SVM (`semantic_hybrid_svm`)**
   - **Inputs:** Question, candidate answer, and the ten memory-mode uncertainty features.
   - **Semantic branch:** Uses the same frozen bi-encoder and 1,537-dimensional interaction representation as Semantic SVM.
   - **Uncertainty branch:** Standardizes the ten numerical uncertainty features using training statistics.
   - **Fusion:** Dense semantic and uncertainty matrices are concatenated, producing `1,537 + 10 = 1,547` features with the default MiniLM encoder.
   - **Classifier:** Balanced `LinearSVC`.
   - **Output:** Predicted label and SVM decision score.
   - **Experimental role:** Tests whether uncertainty adds complementary information after the model already understands the question-answer meaning. It is the semantic counterpart of Lexical Hybrid SVM.

### Group 4: Context and Evidence

7. **Fixed RAG Compare (`rag_compare_fixed`)**
   - **Inputs:** The same question and candidate answer are scored twice: once without knowledge and once with the HaluEval knowledge passage.
   - **Memory mode:** The causal LM scores the answer using only the question.
   - **Context mode:** The causal LM scores the answer after receiving the knowledge passage and question.
   - **Core feature:**

     ```text
     context_improvement = memory_NLL - context_NLL
     hallucination_score = -context_improvement
     ```

     A large positive improvement means the context made the answer more probable. Its negative is therefore treated as the hallucination score.
   - **Decision rule:** No supervised classifier is trained. A scalar threshold is selected on training examples to maximize Macro F1 and is then frozen for validation/test prediction.
   - **Leakage correction:** Both training and evaluation have real memory/context scores. The score direction and threshold are no longer selected from validation labels.
   - **Experimental role:** Tests whether sensitivity to supporting context is sufficient for hallucination detection.

8. **Zero-Shot NLI Evidence (`nli_evidence`)**
   - **Inputs:** `knowledge` and `candidate_answer`. The question and uncertainty features are not used by the decision rule.
   - **NLI formulation:**

     ```text
     premise = knowledge
     hypothesis = candidate_answer
     ```

   - **Model:** Frozen `FacebookAI/roberta-large-mnli`.
   - **Representation:** Three probabilities: entailment, neutral, and contradiction.
   - **Hallucination score:** `1 - P(entailment)`.
   - **Decision rule:** Predict hallucinated when the score is at least 0.5. The model is not fine-tuned on HaluEval.
   - **Output:** Prediction, hallucination score, and all three NLI probabilities.
   - **Experimental role:** Measures how well direct evidence support works without learning dataset-specific lexical or uncertainty patterns.

9. **Evidence-Aware Hybrid (`evidence_aware_hybrid`)**
   - **Inputs:** Question, candidate answer, knowledge passage, and memory-mode uncertainty features.
   - **Semantic branch:** The 1,537-dimensional question-answer interaction vector from Semantic SVM.
   - **Evidence branch:** Three frozen NLI probabilities for entailment, neutral, and contradiction.
   - **Uncertainty branch:** Ten standardized causal-LM uncertainty features.
   - **Fusion:** All branches are concatenated:

     ```text
     1,537 semantic + 3 NLI + 10 uncertainty = 1,550 features
     ```

   - **Classifier:** Balanced `LinearSVC`.
   - **Output:** Predicted label and SVM decision score.
   - **Experimental role:** Tests the central combined hypothesis: hallucination detection should improve when the model considers what was said, how confident the LM was, and whether external knowledge supports the answer.
   - **Efficiency:** Semantic and NLI matrices are cached and reused by the pure and hybrid baselines.

### Group 5: Upper Bound

10. **Cross-Encoder (`cross_encoder`)**
    - **Inputs:** Question and candidate answer as a paired transformer input. Knowledge and precomputed uncertainty features are not included.
    - **Architecture:** Unlike the bi-encoder, both texts pass through the same transformer simultaneously. Self-attention can directly compare tokens across the question and answer.
    - **Initialization:** `distilroberta-base` with a two-class sequence-classification head.
    - **Tokenization:** Paired truncation with a default maximum sequence length of 256 tokens and dynamic batch padding.
    - **Training:** End-to-end fine-tuning with AdamW, a default learning rate of `2e-5`, configurable batch size, epochs, and gradient accumulation.
    - **Output:** Softmax probability of class 1 as the hallucination score; predictions use a 0.5 threshold.
    - **Experimental role:** Serves as the supervised upper bound because joint attention can model detailed question-answer compatibility more directly than frozen bi-encoder features.
    - **Cost:** This is the most computationally expensive baseline because transformer weights are updated rather than used only for frozen feature extraction.

## 4. Unified Experiment Runner

`src/main.py` now controls the complete experiment:

1. Dataset preparation
2. Base memory/context scoring
3. Upgraded-LM memory scoring
4. All ten baseline runs
5. Prediction and metric persistence
6. Individual diagnostic plots
7. Cross-baseline comparison tables and plots

Completed outputs are reused unless `--overwrite` is passed. Baseline subsets can be selected for development or lower-compute runs.

## 5. Validation Results

All ten baselines have completed validation runs on the grouped split:

| Baseline | Accuracy | Macro F1 | AUROC |
|---|---:|---:|---:|
| Lexical SVM | 0.9381 | 0.9380 | 0.9729 |
| Lexical Hybrid SVM | 0.9530 | 0.9530 | 0.9805 |
| Entropy Base | 0.9426 | 0.9425 | 0.9765 |
| Entropy Upgraded | 0.9231 | 0.9231 | 0.9654 |
| Semantic SVM | 0.9695 | 0.9695 | 0.9870 |
| Semantic Hybrid SVM | 0.9675 | 0.9675 | 0.9878 |
| RAG Compare Fixed | 0.7522 | 0.7490 | 0.7723 |
| NLI Evidence | 0.7877 | 0.7866 | 0.7943 |
| Evidence-Aware Hybrid | 0.9770 | 0.9770 | **0.9951** |
| Cross-Encoder | **0.9815** | **0.9815** | 0.9895 |

## 6. Current Findings

1. **The instruction-tuned uncertainty backbone did not outperform `distilgpt2`.**
   - Entropy Upgraded reaches 0.9231 accuracy versus 0.9426 for Entropy Base.
   - Macro F1 decreases from 0.9425 to 0.9231.
   - AUROC decreases from 0.9765 to 0.9654.
   - A stronger generative model therefore does not automatically produce a better hallucination-confidence signal.
   - The most likely explanation is that teacher-forced likelihood captures local fluency and stylistic plausibility more directly than factual correctness.
   - Instruction tuning may sharpen that mismatch by rewarding helpful, well-formed responses even when they are false.
   - Chat-template conditioning and tokenization differences also change the scale and geometry of the extracted log-probability features, so the upgraded model is not simply a stronger version of the same uncertainty signal.
   - The degradation is especially consistent with the short-answer failure mode: a brief but plausible hallucination can still receive high confidence from a stronger chat-oriented surrogate.
   - This means `distilgpt2` should be framed as a cheap reproducible surrogate baseline, not as the project's final uncertainty answer.

2. **Semantic representations clearly outperform TF-IDF.**
   - Semantic SVM improves accuracy from 0.9381 to 0.9695.
   - Its AUROC improves from 0.9729 to 0.9870.

3. **Uncertainty improves the lexical baseline.**
   - Lexical Hybrid SVM improves accuracy by approximately 1.5 percentage points.
   - Its AUROC increases from 0.9729 to 0.9805.

4. **The semantic hybrid does not improve classification accuracy over semantic-only.**
   - Semantic SVM accuracy is 0.9695.
   - Semantic Hybrid accuracy is 0.9675.
   - The hybrid has a slightly higher AUROC, 0.9878 versus 0.9870.

5. **Evidence is most useful when combined with other signals.**
   - Zero-shot NLI alone reaches only 0.7943 AUROC.
   - Evidence-Aware Hybrid reaches the best AUROC, 0.9951.

6. **The cross-encoder is the strongest classifier by accuracy and Macro F1.**
   - Accuracy and Macro F1 are both approximately 0.9815.

7. **Context delta alone remains weak.**
   - Fixed RAG Compare reaches 0.7723 AUROC.
   - Correcting threshold leakage made the result more methodologically valid but did not make the signal sufficient by itself.

8. **Lexical Hybrid SVM still reports non-convergence.**
   - It reached the configured 5,000-iteration limit.
   - Its result should be interpreted with this optimization limitation noted.

## 7. Comparison and Diagnostic Assets

The reporting pipeline now generates:

- A consolidated metric table for all completed baselines
- Accuracy, Macro F1, and AUROC comparison bars
- A combined ROC comparison
- Per-baseline confusion matrices
- Per-baseline ROC curves
- Per-baseline calibration plots

Current aggregate files:

```text
reports/tables/baseline_results_val.csv
reports/tables/baseline_comparison_val.csv
reports/figures/matrix/baseline_comparison_metrics_val.png
reports/figures/matrix/baseline_comparison_roc_val.png
```

## 8. Implementation Status

- [x] Leakage-free grouped dataset splitting
- [x] Lexical SVM
- [x] Lexical Hybrid SVM
- [x] Base Entropy Classifier
- [x] Upgraded Entropy implementation and scoring support
- [x] Semantic SVM
- [x] Semantic Hybrid SVM
- [x] Fixed RAG Compare with training-derived threshold
- [x] Zero-Shot NLI Evidence
- [x] Evidence-Aware Hybrid
- [x] Cross-Encoder Classifier
- [x] Unified `main.py` experiment runner
- [x] Semantic/NLI feature caching
- [x] Aggregate comparison tables and plots
- [x] Offline unit tests
- [x] Upgraded-entropy feature generation and validation run
- [ ] Run the final matrix on the untouched test split
- [ ] Repeat stochastic experiments across multiple seeds

## 9. Next Evaluation Steps

1. Investigate why the instruction-tuned uncertainty features underperform the `distilgpt2` features through feature-distribution and correlation analysis.
2. Reframe `distilgpt2` as a low-cost surrogate baseline and compare it against stronger uncertainty families such as semantic entropy or self-consistency sampling.
3. Freeze all model, feature, and threshold choices using the completed validation matrix.
4. Run the ten baselines once on the untouched test split.
5. Report confidence intervals or paired bootstrap significance tests.
5. Add length-aware evaluation and modeling for the short-answer failure mode, including per-length metrics and short-answer-specific thresholds or features.
