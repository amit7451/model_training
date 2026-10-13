"""
Plan 0: Shared Observability & Anomaly Detection Protocol for Supercomputer Logs
Authoritative, leakage-controlled protocol for BGL & HPC log anomaly detection and root cause analysis.
"""

from .parser import (
    BGLRecord,
    parse_bgl_line,
    parse_location,
    mask_log_message,
    LocationInfo,
)
from .drain_miner import (
    TrainFitDrainMiner,
    DrainFitComparison,
)
from .embedding_cache import (
    MessageEmbeddingCache,
)
from .incidents import (
    Incident,
    cluster_incidents,
    extract_first_alerts,
    sample_hand_validation_incidents,
)
from .windowing import (
    LogWindow,
    build_100_msg_windows,
    build_sliding_20_windows,
    build_time_windows_5m1m,
    build_all_window_views,
)
from .splits import (
    ChronologicalSplitResult,
    create_chronological_splits,
    create_rolling_origin_splits,
    filter_unseen_windows,
)
from .metrics import (
    DetectionMetrics,
    RCAMetrics,
    compute_detection_metrics,
    compute_event_rca_metrics,
    compute_node_localization_accuracy,
    compute_fault_type_metrics,
    compute_perturbation_support_rate,
)
from .statistics import (
    bootstrap_confidence_interval,
    paired_bootstrap_test,
    run_multiseed_evaluation,
)

__version__ = "1.0.0"
__all__ = [
    "BGLRecord",
    "parse_bgl_line",
    "parse_location",
    "mask_log_message",
    "LocationInfo",
    "TrainFitDrainMiner",
    "DrainFitComparison",
    "MessageEmbeddingCache",
    "Incident",
    "cluster_incidents",
    "extract_first_alerts",
    "sample_hand_validation_incidents",
    "LogWindow",
    "build_100_msg_windows",
    "build_sliding_20_windows",
    "build_time_windows_5m1m",
    "build_all_window_views",
    "ChronologicalSplitResult",
    "create_chronological_splits",
    "create_rolling_origin_splits",
    "filter_unseen_windows",
    "DetectionMetrics",
    "RCAMetrics",
    "compute_detection_metrics",
    "compute_event_rca_metrics",
    "compute_node_localization_accuracy",
    "compute_fault_type_metrics",
    "compute_perturbation_support_rate",
    "bootstrap_confidence_interval",
    "paired_bootstrap_test",
    "run_multiseed_evaluation",
]
