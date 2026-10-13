"""
Statistical Rigor Engine for Plan 0 Protocol:
- Multi-seed orchestration (5 seeds: [42, 123, 456, 789, 1234])
- Day-Block Bootstrap Confidence Intervals (95% CI preserving temporal clustering)
- Paired Bootstrap Hypothesis Testing for model comparisons
"""

import random
from collections import defaultdict
from typing import List, Dict, Callable, Tuple, Any, Optional
import numpy as np
from sklearn.metrics import average_precision_score, f1_score


PLAN0_DEFAULT_SEEDS = [42, 123, 456, 789, 1234]


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    scores: np.ndarray,
    day_indices: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = average_precision_score,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Computes Day-Block Bootstrap Confidence Intervals.
    Resamples whole days with replacement to preserve diurnal and burst correlations.
    """
    rng = np.random.RandomState(seed)
    unique_days = np.unique(day_indices)
    n_days = len(unique_days)

    # Pre-index windows by day
    day_to_indices = {d: np.where(day_indices == d)[0] for d in unique_days}

    # Point estimate
    base_val = float(metric_fn(y_true, scores))

    bootstrap_estimates = []
    for _ in range(n_bootstraps):
        sampled_days = rng.choice(unique_days, size=n_days, replace=True)
        sampled_idx = np.concatenate([day_to_indices[d] for d in sampled_days])
        
        y_b = y_true[sampled_idx]
        s_b = scores[sampled_idx]
        
        # Check that both classes exist
        if len(np.unique(y_b)) < 2:
            continue
        val = metric_fn(y_b, s_b)
        bootstrap_estimates.append(val)

    if not bootstrap_estimates:
        return {"estimate": base_val, "ci_lower": base_val, "ci_upper": base_val, "std_err": 0.0}

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(bootstrap_estimates, 100 * alpha))
    ci_upper = float(np.percentile(bootstrap_estimates, 100 * (1.0 - alpha)))
    std_err = float(np.std(bootstrap_estimates))

    return {
        "estimate": round(base_val, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "std_err": round(std_err, 4),
    }


def paired_bootstrap_test(
    y_true: np.ndarray,
    scores_model_a: np.ndarray,
    scores_model_b: np.ndarray,
    day_indices: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = average_precision_score,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes Paired Day-Block Bootstrap Hypothesis Test comparing Model A vs Model B.
    Calculates difference point estimate, 95% CI of delta, and empirical p-value.
    """
    rng = np.random.RandomState(seed)
    unique_days = np.unique(day_indices)
    n_days = len(unique_days)
    day_to_indices = {d: np.where(day_indices == d)[0] for d in unique_days}

    val_a = float(metric_fn(y_true, scores_model_a))
    val_b = float(metric_fn(y_true, scores_model_b))
    delta_base = val_a - val_b

    deltas = []
    for _ in range(n_bootstraps):
        sampled_days = rng.choice(unique_days, size=n_days, replace=True)
        sampled_idx = np.concatenate([day_to_indices[d] for d in sampled_days])
        y_b = y_true[sampled_idx]
        if len(np.unique(y_b)) < 2:
            continue
        ma = metric_fn(y_b, scores_model_a[sampled_idx])
        mb = metric_fn(y_b, scores_model_b[sampled_idx])
        deltas.append(ma - mb)

    if not deltas:
        return {
            "model_a_estimate": round(val_a, 4),
            "model_b_estimate": round(val_b, 4),
            "delta": round(delta_base, 4),
            "ci_lower": round(delta_base, 4),
            "ci_upper": round(delta_base, 4),
            "p_value": 1.0,
            "is_significant": False,
        }

    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    
    # Two-tailed empirical p-value
    p_pos = np.mean([d <= 0 for d in deltas])
    p_neg = np.mean([d >= 0 for d in deltas])
    p_value = float(2 * min(p_pos, p_neg))

    return {
        "model_a_estimate": round(val_a, 4),
        "model_b_estimate": round(val_b, 4),
        "delta": round(delta_base, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(min(1.0, p_value), 4),
        "is_significant": bool(ci_lower > 0 or ci_upper < 0),
    }


def run_multiseed_evaluation(
    train_eval_fn: Callable[[int], Dict[str, float]],
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Executes a training/evaluation workflow across 5 standard seeds and aggregates stats.
    """
    seeds = seeds or PLAN0_DEFAULT_SEEDS
    results_by_seed = {}
    aggregated_metrics = defaultdict(list)

    for s in seeds:
        res = train_eval_fn(s)
        results_by_seed[s] = res
        for k, v in res.items():
            if isinstance(v, (int, float)):
                aggregated_metrics[k].append(v)

    summary = {}
    for k, vals in aggregated_metrics.items():
        summary[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }

    return {
        "seeds_evaluated": seeds,
        "per_seed_results": results_by_seed,
        "summary": summary,
    }
