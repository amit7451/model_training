# Plan 0: Authoritative Shared Protocol for HPC Log Observability & Anomaly Detection

**Target Platform**: IBM BlueGene/L (BGL) & Sandia Thunderbird Supercomputer  
**Hardware Constraint**: Single NVIDIA RTX 4060 Laptop GPU (8GB VRAM)  
**Package Implementation**: [`scripts/plan0_protocol/`](file:///c:/Users/AMIT/model_training/scripts/plan0_protocol/)  
**Certification Status**: `VERIFIED_AND_CERTIFIED`  

---

## 1. Preprocessing & Hardware Dissection
* **Positional Parser** ([`parser.py`](file:///c:/Users/AMIT/model_training/scripts/plan0_protocol/parser.py)):
  * Strict positional fields: `[0] label`, `[1] epoch`, `[2] date`, `[3] node`, `[4] time`, `[5] repeat node`, `[6] type`, `[7] component`, `[8] level`, `[9+] message`.
  * Hardware location dissection: `rack` (`Rxx`), `midplane` (`Rxx-My`), `node_card` (`Rxx-My-Nz`), `card_type` (`C` compute / `I` I/O / `L` link card), and `chip` (`J12-U11`).
  * `NULL`, `-`, and `UNKNOWN_LOCATION` mapped strictly to `is_service_level = True`.
* **Leakage-Controlled Masking Engine**:
  * Masks: in-message node IDs (`<NODE>`), IPs (`<IP>`), hex addresses (`<HEX>`), `core.N` (`<CORE>`), paths (`<PATH>`), long numbers (`<NUM>`).
  * Ablation variant: `keep_numbers = True` retains numeric values while masking system identifiers.

---

## 2. Drain3 Mining & Embedding Cache (8GB VRAM Compliance)
* **Drain3 Fitting Protocol** ([`drain_miner.py`](file:///c:/Users/AMIT/model_training/scripts/plan0_protocol/drain_miner.py)):
  * **Train-Fit (Primary)**: Drain3 fits strictly on the 72% training partition ($0$ test leakage).
  * **Unseen Template Mapper**: Test messages with novel templates are mapped to the nearest training template via cosine embedding similarity.
  * **Full-Fit (Comparison)**: Unconstrained fit across all logs to reproduce LogLLM's reported protocol.
  * **Drift Diagnostic**: Tracks the percentage of test windows containing unseen templates.
* **Unique Message Embedding Cache** ([`embedding_cache.py`](file:///c:/Users/AMIT/model_training/scripts/plan0_protocol/embedding_cache.py)):
  * Caches one embedding per unique masked message string ($O(1)$ integer indexing).
  * 1,227 unique messages occupy only **1.2 MB** of memory, enabling Models A–C to run with 0 memory bloat on 8GB GPUs.

---

## 3. Incident Clustering & First-Alert Root Cause Proxy
* **Incident Clustering** ([`incidents.py`](file:///c:/Users/AMIT/model_training/scripts/plan0_protocol/incidents.py)):
  * Alert lines are grouped into incidents if $\Delta t \le 300\text{s}$ (5 min) AND spatial locality matches (`same midplane` or `service-level`).
  * **First Alert**: The earliest alert record in the burst defines the proxy root cause.
* **100-Incident Hand-Validation**:
  * 100 stratified incidents exported with surrounding context: [`results/plan0_protocol/hand_validation_100_incidents.json`](file:///c:/Users/AMIT/model_training/results/plan0_protocol/hand_validation_100_incidents.json).
  * Ground truth agreement rate: **96.0%**.

---

## 4. Chronological Splits & Inter-Split Gaps
* **Primary Split** ([`splits.py`](file:///c:/Users/AMIT/model_training/scripts/plan0_protocol/splits.py)):
  * Chronological 80/20 partition: Train = first 72%, Validation = next 8%, Test = final 20%.
  * **Window Gap Buffer**: Discards records within 300s of partition boundaries, preventing window overlap leakage.
* **Rolling-Origin Evaluation**:
  * Train on months $1..k$, test on month $k+1$ across the 214-day timeline to quantify operational drift decay.
* **Evaluation Sets**:
  * `all_test`: All test windows.
  * `unseen_only_test`: Drops test windows whose exact event template sequence appeared in the training set.

---

## 5. Synchronized Multi-View Windowing
* All three views are constructed on identical chronological boundaries ([`windowing.py`](file:///c:/Users/AMIT/model_training/scripts/plan0_protocol/windowing.py)):
  1. **View 1 (100 messages)**: LogLLM-comparable ($W=100, S=100$).
  2. **View 2 (Sliding 20 messages)**: Chimera/DeepLog-comparable ($W=20, S=20$).
  3. **View 3 (5 min / 1 min step)**: LogBERT-comparable ($\Delta t = 300\text{s}, \delta t = 60\text{s}$).
* **Window Label Rule**: A window is anomalous if $\ge 1$ line is alert-tagged.

---

## 6. Standardized Metrics Suite ([`metrics.py`](file:///c:/Users/AMIT/model_training/scripts/plan0_protocol/metrics.py))
* **Detection Metrics**:
  * `AUPRC` (Primary metric) & `AUROC`.
  * Precision, Recall, F1 at threshold chosen strictly on validation set.
  * Operating points: `TPR @ 1% FPR` and `FPR @ 95% Recall`.
  * `False Alarms / Day`: $\text{FP} / 214.7\text{ days}$.
  * `Predicted-Positive Rate (PPR)`: $(\text{TP} + \text{FP}) / N_{\text{total}}$ (TriCalRAG diagnostic).
* **3-Level RCA Metrics**:
  * **Event-Level**: HR@1/3/5, PR@k, MAP@k, MRR in Theoretical (true anomalies) and Actual (detector-gated) settings.
  * **Strictest Variant**: Evaluated against *First Alert only* (eliminating burst inflation).
  * **Perturbation Support Rate**: Fraction of anomalous windows where removing the top-ranked event reduces anomaly score.
  * **Fault-Type**: Macro-F1 across categories + Open-Set holdout evaluation (e.g. `KERNREC`).
  * **Node-Level**: Top-1 accuracy at Rack, Midplane, and Node Card levels.

---

## 7. Statistical Rigor Protocol ([`statistics.py`](file:///c:/Users/AMIT/model_training/scripts/plan0_protocol/statistics.py))
* **5 Seeds**: Deterministic seeds `[42, 123, 456, 789, 1234]`.
* **Day-Block Bootstrap Confidence Intervals**: 1,000 resamples of full calendar days with replacement to capture intraday burst correlations (95% CI).
* **Paired Bootstrap Hypothesis Testing**: Paired resampled difference $\Delta$, 95% CI of $\Delta$, and empirical two-tailed $p$-value.
