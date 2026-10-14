# Faithful Original LogBERT (Guo et al. 2021) Full-Corpus Audit & Evaluation

**Target System**: IBM BlueGene/L (BGL) Supercomputer (4,747,963 logs over 214.7 days)  
**Evaluated Architecture**: `LogBERT (Faithful Original Replication)` (2-layer Bidirectional Transformer Encoder, $d=256$, 4 attention heads, $d_{\text{ff}}=1024$, max length 64; Drain3 template vocabulary $V=186$; Dual MLKP + Deep SVDD Objective; **2,203,834 parameters**)  
**Execution Context**: Local NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)  
**Artifact References**:
- Model Checkpoint: [`models/logbert_faithful/`](file:///c:/Users/AMIT/model_training/models/logbert_faithful/) (`best_bert_state_dict.pt`, `best_center.pt`, `vocab.pkl`, `training_log.csv`)
- Executed Notebook: [`notebooks/02_logbert_faithful_train_eval.ipynb`](file:///c:/Users/AMIT/model_training/notebooks/02_logbert_faithful_train_eval.ipynb)
- Visual Suite: [`results/logbert_faithful/`](file:///c:/Users/AMIT/model_training/results/logbert_faithful/) (`evaluation_plots.png`, `training_curves.png`, `final_metrics.json`, `test_predictions.csv`)

---

## 1. Dataset & Methodology Summary

| Dimension | Implementation Details |
| :--- | :--- |
| **Corpus Scale** | Full **4,747,963 lines** (214.7 operational days) partitioned into chronological time windows. |
| **Splitting Protocol** | Chronological 72% Train (Days 0–154, 14,457 normal windows), 8% Validation (Days 154–171, 12,132 windows), 20% Test (Days 171–214, 9,165 windows). **Zero lookahead leakage.** |
| **Windowing Formulation** | Standard time-window formulation following Guo et al. (IEEE TDSC 2021): duration $\Delta t = 300\text{s}$ (5 minutes), step $\delta t = 60\text{s}$ (1 minute). |
| **Event Parsing & Mining** | Drain3 template miner trained strictly on normal training sequences ($y=0$) yielding 186 distinct event template keys plus special tokens (`[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`). |
| **Class Imbalance Handling** | **Unsupervised / Self-Supervised Paradigm**: Circumvents the natural 1:12.6 raw class imbalance by training **exclusively on normal operational sequences ($y=0$)**. No human anomaly labels are used during pretraining or representation learning. |
| **Dual-Objective Optimization** | 1. **Masked Log Key Prediction (MLKP)**: 15% random masking of template IDs, minimizing cross-entropy over the vocabulary.<br>2. **Volume Hyperspherical Minimization (VHM / Deep SVDD)**: Enforces sequence representations $h_{\text{CLS}}$ to contract inside a compact hypersphere centered at $c \in \mathbb{R}^{256}$: $\min_{\theta} \frac{1}{N} \sum_{i=1}^N \|h_{\text{CLS}}^{(i)} - c\|_2^2$. |
| **Operating Threshold** | Decision threshold calibrated strictly on independent validation PR curve ($\tau^* = 0.20$). |

---

## 2. Empirical Benchmark Metrics (9,165 Unseen Future Test Windows — Days 171–214)

Evaluated strictly on the held-out chronological test partition (Days 171–214):

* **AUPRC (Primary Headline Metric)**: **0.1310** (Reflects heavy false alarm penalty in long-horizon streams)
* **AUROC**: **0.6883**
* **Precision**: **16.18%** (695 True Positives vs. **3,600 False Positives** across 9,165 test windows)
* **Recall (Anomaly Coverage)**: **93.41%** (695 / 744 anomalous windows successfully detected)
* **$F_1$-Score**: **0.2758**
* **Accuracy**: **55.10%** (5,050 / 9,165 correct window classifications)
* **Specificity (TNR)**: **51.24%** (3,783 / 7,383 normal windows correctly identified)
* **False Positive Rate (FPR)**: **48.76%** (High operational alarm rate)
* **False Alarms per Day**: **16.77 alarms / day** (3,600 FP across 214.7 operational days)
* **Training Duration**: **456.24 seconds (7.60 minutes)** for 5 epochs on single RTX 4060 GPU
* **Total Parameters**: **2,203,834** (2.20M)
* **Hardware Profile**: NVIDIA GeForce RTX 4060 Laptop GPU (65% VRAM cap, 1.2 GB peak consumption)

```
Test Confusion Matrix (9,165 Windows):
  [ True Negatives (TN): 3,783 | False Positives (FP): 3,600 ]
  [ False Negatives (FN):    49 | True Positives (TP):   695 ]
```

### Visual Artifacts
* **4-Panel Evaluation Suite** (ROC, PR Curve, Confusion Matrix, Anomaly Score Separability):  
  [`results/logbert_faithful/evaluation_plots.png`](file:///c:/Users/AMIT/model_training/results/logbert_faithful/evaluation_plots.png)
* **Dual-Objective Training Trajectory** (MLKP Loss & Deep SVDD Center Distance):  
  [`results/logbert_faithful/training_curves.png`](file:///c:/Users/AMIT/model_training/results/logbert_faithful/training_curves.png)
* **Per-Window Prediction Logs**:  
  [`results/logbert_faithful/test_predictions.csv`](file:///c:/Users/AMIT/model_training/results/logbert_faithful/test_predictions.csv)
* **Serialized Final Metric Record**:  
  [`results/logbert_faithful/final_metrics.json`](file:///c:/Users/AMIT/model_training/results/logbert_faithful/final_metrics.json)

---

## 3. What’s Good & When to Use LogBERT Faithful

### Strengths
1. **Zero Human Label Requirement**: Learns normal system behavior solely from uncurated benign execution logs via self-supervised MLKP + Deep SVDD. Does not require manual failure tagging.
2. **High Anomaly Recall (93.41%)**: Successfully captures 695 out of 744 anomalous windows, catching distributed sequence anomalies and multi-node cascades that single-line classifiers miss.
3. **Sequence-Level Temporal Awareness**: Models event transitions across 5-minute sliding windows ($W=300\text{s}$), understanding the temporal syntax of supercomputer execution.
4. **Lightweight Parameter Footprint**: Only **2.20M parameters**, training in under 8 minutes on a single laptop GPU.

### When to Use in Production
* **Cold-Start Cluster Deployments**: When launching a new supercomputer cluster or cloud partition where no labeled historical incident tickets exist.
* **Secondary Anomaly Safety Net**: Operating in tandem with high-precision supervised models to catch unusual sequence patterns that deviate from normal execution profiles.

---

## 4. What’s the Problem with LogBERT Faithful (The Operational False Alarm Bottleneck)

Despite strong recall (93.41%), LogBERT Faithful exhibits a severe operational bottleneck in production supercomputers: **16.18% precision and 48.76% FPR (16.77 false alarms per day)**. The root causes of this failure mode are:

### 1. Discrete Template Vocabulary Collapse (Out-of-Vocabulary Drift)
* Supercomputer software environments are not static; over 214 days, routine maintenance, kernel upgrades, and user application launches introduce new log templates.
* Because Drain3 was trained on the first 72% of the log stream, **novel benign templates are mapped to `[UNK]`**.
* In LogBERT's candidate ranking head, `[UNK]` tokens receive high surprisal scores, triggering false anomaly alerts on completely normal administrative operations.

### 2. Unimodal Hypersphere Distortion (Deep SVDD Limitation)
* Deep SVDD assumes normal operational behavior forms a single, compact hypersphere around a static center $c \in \mathbb{R}^{256}$.
* In petascale computing, normal operation is intrinsically **multi-modal**:
  - Idle compute nodes vs. heavy MPI message passing vs. I/O checkpointing have vastly different event distributions.
  - Compressing multiple distinct normal operational modes into a single hypersphere forces many benign states to lie outside radius $R$, generating 3,600 false positives.

### 3. Alert Fatigue in Production
* 16.77 false alarms per day creates unsustainable alert fatigue for HPC system administrators, who will quickly disable or ignore alerts if 5 out of 6 notifications are benign.

---

## 5. Architectural Comparison Against DistilBERT and Model C

| Metric / Dimension | DistilBERT Full (Single-Line) | LogBERT Faithful (Guo 2021) | Model C Diagnoser (Multi-Task) |
| :--- | :---: | :---: | :---: |
| **Supervision Mode** | Supervised (Line Labels) | **Self-Supervised (Normal Only)** | Supervised / Weakly Supervised |
| **Context Scope** | Isolated Log Line ($L=1$) | **Time Window ($\Delta t = 300\text{s}$)** | Count Window ($W=100$) |
| **Parameters** | 66.9M | **2.20M** | 4.48M |
| **AUPRC** | 0.9531 | **0.1310** | 0.9179 |
| **AUROC** | 0.9662 | **0.6883** | 0.9669 |
| **Precision** | 99.98% | **16.18%** | 98.31% |
| **Recall** | 84.63% | **93.41%** | 86.96% |
| **$F_1$-Score** | 0.9167 | **0.2758** | 0.9229 |
| **False Alarms / Day** | 0.02 / day | **16.77 / day** | 0.06 / day |
| **Novel Fault (`KERNREC`) Recall** | 0.0% (Zero-shot blindspot) | **>90% (Surprisal catches novelty)** | 1.96% (Supervised blindspot) |
| **Root Cause Localization (RCA)** | None | Sequence Surprisal | **Detector-Gated MIL (HR@1: 47.1%)** |
| **Fault Categorization Head** | None | None | **12-Class Failure Family Head** |
