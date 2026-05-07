# Progress Report Template

## 1. Project Title, Group Members, and Problem Definition

**Project Title:** Epistemic Uncertainty in LLM Hallucinations

**Group Members:**
- Name Surname - Student ID
- Name Surname - Student ID

### Problem Definition

This project studies hallucination behavior in large language models and focuses on **epistemic uncertainty**, which is uncertainty caused by incomplete knowledge, limited evidence, or model uncertainty rather than randomness in the generation process alone.

Answer the following in this section:
- What exact task are we solving?
- Why does this task matter for reliable NLP systems?
- Which dataset or benchmark do we use to measure hallucinations?
- What is the system input?
- What is the system output?
- Is the task framed as generation, classification, question answering, uncertainty estimation, calibration, or multi-step evaluation?
- What makes the task technically challenging?

Suggested project-specific framing:
- Input: a prompt, question, or claim given to an LLM.
- Output: the model response plus a hallucination-related label or score such as factuality, correctness, confidence, entropy, self-consistency, or uncertainty estimate.
- Challenge: hallucinations are often subtle, dataset definitions vary, and uncertainty signals may not align perfectly with correctness.

If the topic has changed since the proposal:
- Explain what changed.
- Explain why the change improves feasibility or technical clarity.

## 2. Dataset and Preprocessing Status

### Dataset Summary
- **Dataset name:**
- **Source / URL:**
- **License / access conditions:**
- **Task type:**
- **Number of samples:**
- **Input fields:**
- **Target fields / labels:**

### Current Status
- Downloaded: Yes / No
- Cleaned: Yes / No
- Split into train/validation/test: Yes / No
- Ready for baseline experiments: Yes / No

### Preprocessing Completed
- [ ] Convert raw files into a unified tabular format
- [ ] Remove duplicates
- [ ] Handle missing values
- [ ] Normalize label names
- [ ] Filter unusable samples
- [ ] Create train/validation/test split
- [ ] Save processed artifacts under `data/processed/`

### Known Issues
- Class imbalance:
- Noisy labels:
- Long-context examples:
- Access limitations:
- Annotation ambiguity:

### Coding-Relevant Notes
- Record exact dataset versions.
- Keep a raw copy under `data/raw/`.
- Make preprocessing deterministic.
- Save processed outputs with metadata so experiments are reproducible.

## 3. Literature Review Progress

Summarize the papers or methods reviewed so far.

For each reviewed work, explain:
- What problem it studies
- What method it uses
- How it relates to hallucination detection or uncertainty estimation
- Whether it can serve as a baseline, evaluation reference, or design inspiration

Suggested literature buckets:
- LLM hallucination detection
- Factuality evaluation
- Confidence estimation and calibration
- Self-consistency / sampling-based uncertainty
- Semantic entropy or disagreement-based uncertainty
- Selective prediction and abstention

### Current Technical Direction

Use this subsection to connect the literature to your implementation plan:
- Which approaches seem feasible with your compute budget?
- Which methods can be implemented first as baselines?
- Which uncertainty signals are most promising for your dataset?

## 4. Baseline Models

The final project should compare at least three baselines. A good structure for this project is:

| Baseline | Type | Status | Why it matters |
|---|---|---|---|
| Deterministic prompting baseline | Generation + evaluation | Planned / In progress / Done | Establishes base hallucination rate without uncertainty modeling |
| Multi-sample self-consistency baseline | Sampling-based uncertainty | Planned / In progress / Done | Measures disagreement across multiple generations |
| Confidence / entropy baseline | Token-level uncertainty | Planned / In progress / Done | Uses model probabilities or proxy confidence signals |
| Semantic similarity / contradiction baseline | Response comparison | Planned / In progress / Done | Detects unstable or conflicting answers |

For each baseline, report:
- Short description
- Why it is suitable
- Implementation status
- Initial observations
- Problems encountered

## 5. Initial Experimental Results

### Metrics to Consider
- Accuracy
- Precision / Recall / F1
- AUROC for hallucination detection
- Expected Calibration Error (ECE)
- Brier score
- Selective risk / coverage
- Runtime per sample

### Preliminary Results Table

| Model | Main Metric | Calibration Metric | Runtime | Notes |
|---|---|---|---|---|
| Baseline 1 | - | - | - | Implemented / In progress |
| Baseline 2 | - | - | - | Implemented / In progress |
| Baseline 3 | - | - | - | Implemented / In progress |

### Interpretation

Briefly explain:
- Which baseline currently looks most promising
- Whether uncertainty correlates with hallucinations
- Whether results are too early to conclude anything
- What the results suggest about the next experiments

## 6. Planned Improvements and Technical Direction

Describe how the next phase will improve the pipeline.

Possible directions:
- Improve prompt design or response parsing
- Add stronger uncertainty features
- Compare white-box vs black-box uncertainty signals
- Add calibration or threshold tuning
- Improve dataset cleaning
- Use better evaluation metrics
- Test stronger open-source or API-based LLMs
- Add retrieval or evidence-aware variants if the project scope allows

This section should explain the next technical step, not just broad intentions.

## 7. Ablation and Error Analysis Plan

### Ablation Plan

Potential ablations for this topic:
- Single response vs multiple sampled responses
- With vs without entropy-based features
- With vs without semantic clustering of answers
- Different decoding temperatures
- Different uncertainty thresholds
- Different prompt templates

### Error Analysis Plan

State how you will inspect failures:
- False negatives: hallucinations the system misses
- False positives: correct answers flagged as uncertain
- Overconfident hallucinations
- Cases where all samples agree but are wrong
- Dataset slices by topic, difficulty, or answer length

## 8. Visualization and Interpretation

Useful visualizations for this project:
- Confusion matrix
- Precision-recall curve
- ROC curve
- Reliability diagram / calibration plot
- Risk-coverage curve
- Uncertainty histogram for correct vs incorrect answers
- Scatter plot of uncertainty vs correctness
- Example table of representative failures

State which ones you already support and which ones are planned.

## 9. GitHub and Reproducibility Status

### Repository
- **GitHub link:**
- **Current branch workflow:**
- **Commit activity summary:**

### Structure

Reference the current repo layout and explain:
- where raw data lives
- where processed data lives
- where training / inference scripts live
- where outputs and figures are stored
- how to reproduce one experiment end-to-end

### Reproducibility Checklist
- [ ] README explains setup
- [ ] Dependencies pinned
- [ ] Dataset instructions written
- [ ] Scripts have CLI arguments
- [ ] Random seeds controlled
- [ ] Output directories standardized
- [ ] Results saved in machine-readable form

## 10. Current Challenges and Next Steps

### Current Challenges
- Dataset selection or access
- Ground-truth ambiguity
- Compute limits
- API cost or rate limits
- Missing probability outputs from some LLM APIs
- Evaluation design for hallucination severity

### Next Steps Before Final Submission
- Finalize dataset
- Implement all baselines
- Run controlled experiments
- Perform ablation study
- Complete error analysis
- Add visualizations
- Write final report with reproducible results

## Appendix: What To Fill Next

Before the progress report deadline, make sure the following become concrete:
- exact dataset choice
- exact three baselines
- exact evaluation metrics
- at least one runnable experiment script
- at least one saved result table
- repository link and setup instructions
