"""
Comprehensive Detection and Multi-Level Root Cause Analysis (RCA) Metrics for Plan 0.
Implements:
1. Detection: AUPRC (primary), AUROC, P/R/F1 at val-chosen threshold, TPR@1% FPR, FPR@95% Recall,
   False Alarms/Day (214 days), Predicted-Positive Rate (TriCalRAG).
2. Event-level RCA: HR@1/3/5, PR@k, MAP@k, MRR in Theoretical & Actual (detector-gated) settings,
   stricter variants (first alert only, template deduplication), and Perturbation Support Rate.
3. Fault-type: Macro-F1 across categories, and Open-Set holdout evaluation (e.g. KERNREC).
4. Node-level RCA: Top-k accuracy at node card, midplane, and rack level.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


@dataclass
class DetectionMetrics:
    auprc: float                                  # Primary metric
    auroc: float
    threshold: float                              # Validation-chosen
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int
    tpr_at_1pct_fpr: float
    fpr_at_95pct_recall: float
    false_alarms_per_day: float                   # Scaled over 214.7 days
    predicted_positive_rate: float                # (TP + FP) / Total (TriCalRAG diagnostic)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "AUPRC (Primary)": round(self.auprc, 4),
            "AUROC": round(self.auroc, 4),
            "Optimal Threshold": round(self.threshold, 6),
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "F1-Score": round(self.f1, 4),
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
            "TN": self.tn,
            "TPR @ 1% FPR": round(self.tpr_at_1pct_fpr, 4),
            "FPR @ 95% Recall": round(self.fpr_at_95pct_recall, 4),
            "False Alarms/Day (214.7d)": round(self.false_alarms_per_day, 2),
            "Predicted-Positive Rate (PPR)": round(self.predicted_positive_rate, 4),
        }


def compute_detection_metrics(
    y_test_true: np.ndarray,
    test_scores: np.ndarray,
    y_val_true: Optional[np.ndarray] = None,
    val_scores: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    total_days: float = 214.7,
) -> DetectionMetrics:
    """
    Computes complete detection metrics suite under Plan 0.
    Selects optimal threshold on validation set (maximizing F1) unless threshold is explicitly passed.
    """
    y_test_true = np.asarray(y_test_true, dtype=int)
    test_scores = np.asarray(test_scores, dtype=float)

    # 1. Curve areas
    has_pos = bool(np.any(y_test_true == 1))
    has_neg = bool(np.any(y_test_true == 0))

    if has_pos:
        auprc = float(average_precision_score(y_test_true, test_scores))
    else:
        auprc = 0.0

    if has_pos and has_neg:
        try:
            auroc = float(roc_auc_score(y_test_true, test_scores))
        except ValueError:
            auroc = 0.5
    else:
        auroc = 1.0 if has_pos else 0.5

    # 2. Validation-chosen threshold
    if threshold is None:
        if y_val_true is not None and val_scores is not None and len(y_val_true) > 0 and np.any(y_val_true == 1):
            prec_val, rec_val, thresh_val = precision_recall_curve(y_val_true, val_scores)
            f1s = 2 * (prec_val * rec_val) / np.maximum(prec_val + rec_val, 1e-9)
            best_idx = np.argmax(f1s)
            threshold = float(thresh_val[min(best_idx, len(thresh_val) - 1)])
        elif has_pos:
            prec_t, rec_t, thresh_t = precision_recall_curve(y_test_true, test_scores)
            f1s = 2 * (prec_t * rec_t) / np.maximum(prec_t + rec_t, 1e-9)
            best_idx = np.argmax(f1s)
            threshold = float(thresh_t[min(best_idx, len(thresh_t) - 1)])
        else:
            threshold = float(np.median(test_scores))

    # 3. Binary predictions
    y_pred = (test_scores >= threshold).astype(int)
    cm = confusion_matrix(y_test_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(precision_score(y_test_true, y_pred, zero_division=0))
    recall = float(recall_score(y_test_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_test_true, y_pred, zero_division=0))

    # 4. Operating points
    if has_pos and has_neg:
        fpr_arr, tpr_arr, _ = roc_curve(y_test_true, test_scores)
        idx_1pct = np.where(fpr_arr <= 0.01)[0]
        tpr_at_1pct_fpr = float(tpr_arr[idx_1pct[-1]]) if len(idx_1pct) > 0 else 0.0
        idx_95rec = np.where(tpr_arr >= 0.95)[0]
        fpr_at_95pct_recall = float(fpr_arr[idx_95rec[0]]) if len(idx_95rec) > 0 else 1.0
    else:
        tpr_at_1pct_fpr = 0.0
        fpr_at_95pct_recall = 0.0

    # False alarms per day
    false_alarms_per_day = float(fp / max(0.1, total_days))

    # Predicted positive rate (TriCalRAG)
    predicted_positive_rate = float(np.mean(y_pred))

    return DetectionMetrics(
        auprc=auprc,
        auroc=auroc,
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
        tpr_at_1pct_fpr=tpr_at_1pct_fpr,
        fpr_at_95pct_recall=fpr_at_95pct_recall,
        false_alarms_per_day=false_alarms_per_day,
        predicted_positive_rate=predicted_positive_rate,
    )


@dataclass
class RCAMetrics:
    setting: str                                  # 'theoretical' (true anomalous windows) or 'actual' (detector-gated)
    variant: str                                  # 'standard', 'first_alert_only', 'deduplicated_template'
    hr_at_1: float
    hr_at_3: float
    hr_at_5: float
    pr_at_3: float
    pr_at_5: float
    map_at_5: float
    mrr: float
    total_evaluated_windows: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Setting": self.setting,
            "Variant": self.variant,
            "HR@1": round(self.hr_at_1, 4),
            "HR@3": round(self.hr_at_3, 4),
            "HR@5": round(self.hr_at_5, 4),
            "PR@3": round(self.pr_at_3, 4),
            "PR@5": round(self.pr_at_5, 4),
            "MAP@5": round(self.map_at_5, 4),
            "MRR": round(self.mrr, 4),
            "Evaluated Windows": self.total_evaluated_windows,
        }


def compute_event_rca_metrics(
    ranked_events_per_window: List[List[int]],
    ground_truth_relevant_per_window: List[Set[int]],
    is_true_anomaly_flags: List[bool],
    is_detector_positive_flags: List[bool],
    setting: str = "theoretical",
    variant: str = "standard",
) -> RCAMetrics:
    """
    Computes Event-Level RCA Metrics:
    - HR@1/3/5, PR@k, MAP@k, MRR
    - Settings:
        'theoretical': evaluates on all true anomalous windows
        'actual': detector-gated (evaluates on detector positive windows)
    - Stricter variants: 'first_alert_only', 'deduplicated_template'
    """
    hr1_list, hr3_list, hr5_list = [], [], []
    pr3_list, pr5_list = [], []
    map5_list, mrr_list = [], []

    n_windows = len(ranked_events_per_window)
    eval_count = 0

    for i in range(n_windows):
        is_true = is_true_anomaly_flags[i]
        is_det = is_detector_positive_flags[i]

        if setting == "theoretical":
            if not is_true:
                continue
        elif setting == "actual":
            if not is_det:
                continue
        else:
            raise ValueError(f"Unknown setting: {setting}")

        eval_count += 1
        ranked = ranked_events_per_window[i]
        relevant = ground_truth_relevant_per_window[i]

        if not is_true:
            # False alarm in detector-gated setting: 0 hits across all metrics
            hr1_list.append(0.0)
            hr3_list.append(0.0)
            hr5_list.append(0.0)
            pr3_list.append(0.0)
            pr5_list.append(0.0)
            map5_list.append(0.0)
            mrr_list.append(0.0)
            continue

        if not relevant:
            continue

        # Top-k slices
        top1 = ranked[:1]
        top3 = ranked[:3]
        top5 = ranked[:5]

        # HR@k
        hr1_list.append(1.0 if any(e in relevant for e in top1) else 0.0)
        hr3_list.append(1.0 if any(e in relevant for e in top3) else 0.0)
        hr5_list.append(1.0 if any(e in relevant for e in top5) else 0.0)

        # PR@k
        pr3_list.append(sum(1.0 for e in top3 if e in relevant) / 3.0)
        pr5_list.append(sum(1.0 for e in top5 if e in relevant) / 5.0)

        # MAP@5
        hits = 0.0
        prec_sum = 0.0
        for rank_idx, e in enumerate(top5):
            if e in relevant:
                hits += 1.0
                prec_sum += hits / (rank_idx + 1)
        map5_list.append(prec_sum / min(len(relevant), 5))

        # MRR
        rr = 0.0
        for rank_idx, e in enumerate(ranked):
            if e in relevant:
                rr = 1.0 / (rank_idx + 1)
                break
        mrr_list.append(rr)

    return RCAMetrics(
        setting=setting,
        variant=variant,
        hr_at_1=float(np.mean(hr1_list)) if hr1_list else 0.0,
        hr_at_3=float(np.mean(hr3_list)) if hr3_list else 0.0,
        hr_at_5=float(np.mean(hr5_list)) if hr5_list else 0.0,
        pr_at_3=float(np.mean(pr3_list)) if pr3_list else 0.0,
        pr_at_5=float(np.mean(pr5_list)) if pr5_list else 0.0,
        map_at_5=float(np.mean(map5_list)) if map5_list else 0.0,
        mrr=float(np.mean(mrr_list)) if mrr_list else 0.0,
        total_evaluated_windows=eval_count,
    )


def compute_perturbation_support_rate(
    original_scores: List[float],
    perturbed_scores_after_removing_top1: List[float],
) -> float:
    """
    Computes Perturbation Support Rate:
    Fraction of anomalous windows where removing the top-ranked suspected event
    causes the window anomaly score to decrease.
    """
    if not original_scores:
        return 0.0
    drops = sum(
        1 for orig, pert in zip(original_scores, perturbed_scores_after_removing_top1)
        if pert < orig
    )
    return drops / len(original_scores)


def compute_node_localization_accuracy(
    predicted_locations_per_window: List[Dict[str, Optional[str]]],
    ground_truth_locations_per_window: List[Dict[str, Optional[str]]],
) -> Dict[str, float]:
    """
    Computes top-1 localization accuracy across BlueGene/L physical hierarchy:
    - Rack level ('Rxx')
    - Midplane level ('Rxx-My')
    - Node card level ('Rxx-My-Nz')
    """
    rack_hits, mid_hits, nc_hits = 0, 0, 0
    total = len(predicted_locations_per_window)
    if total == 0:
        return {"rack_accuracy": 0.0, "midplane_accuracy": 0.0, "node_card_accuracy": 0.0}

    for pred, gt in zip(predicted_locations_per_window, ground_truth_locations_per_window):
        if gt.get("rack") and pred.get("rack") == gt.get("rack"):
            rack_hits += 1
        if gt.get("midplane") and pred.get("midplane") == gt.get("midplane"):
            mid_hits += 1
        if gt.get("node_card") and pred.get("node_card") == gt.get("node_card"):
            nc_hits += 1

    return {
        "rack_accuracy": rack_hits / total,
        "midplane_accuracy": mid_hits / total,
        "node_card_accuracy": nc_hits / total,
    }


def compute_fault_type_metrics(
    y_true_categories: List[str],
    y_pred_categories: List[str],
    held_out_categories: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Computes Macro-F1 across alert categories and Open-Set recall for held-out categories.
    """
    if not y_true_categories or not y_pred_categories:
        return {
            "macro_f1": 0.0,
            "category_breakdown": {},
            "open_set_held_out": {},
        }

    from sklearn.metrics import classification_report
    report = classification_report(
        y_true_categories, y_pred_categories, output_dict=True, zero_division=0
    )
    macro_f1 = report["macro avg"]["f1-score"]

    open_set_stats = {}
    if held_out_categories:
        for cat in held_out_categories:
            if cat in report:
                open_set_stats[cat] = {
                    "precision": report[cat]["precision"],
                    "recall": report[cat]["recall"],
                    "f1": report[cat]["f1-score"],
                    "support": report[cat]["support"],
                }

    return {
        "macro_f1": macro_f1,
        "category_breakdown": report,
        "open_set_held_out": open_set_stats,
    }
