"""
Multi-View Windowing Engine for Plan 0 Protocol.
Builds three synchronized window views on identical chronological boundaries:
1. 100 messages (LogLLM-comparable)
2. Sliding 20 messages (Chimera-comparable)
3. 5 minutes window with 1 minute step (LogBERT-comparable)
Enforces split gaps and alert-tagged anomaly definitions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import defaultdict
import hashlib

from .parser import BGLRecord


@dataclass
class LogWindow:
    window_id: int
    view_name: str                      # 'view_100_msg', 'view_sliding_20', 'view_5m1m_time'
    split_name: str                     # 'train', 'val', 'test'
    start_line_idx: int
    end_line_idx: int                   # exclusive
    line_indices: List[int]
    template_ids: List[int]
    timestamps: List[int]
    is_anomaly: bool
    alert_line_indices: List[int]
    first_alert_line_indices: List[int]
    fault_category: Optional[str] = None
    first_alert_rack: Optional[str] = None
    first_alert_midplane: Optional[str] = None
    first_alert_node_card: Optional[str] = None
    has_unseen_template: bool = False

    @property
    def sequence_tuple(self) -> Tuple[int, ...]:
        return tuple(self.template_ids)

    @property
    def sequence_hash(self) -> str:
        s = ",".join(map(str, self.template_ids))
        return hashlib.md5(s.encode('ascii')).hexdigest()

    @property
    def length(self) -> int:
        return len(self.line_indices)

    @property
    def duration_sec(self) -> int:
        if not self.timestamps:
            return 0
        return self.timestamps[-1] - self.timestamps[0]


def _create_window_object(
    w_records: List[BGLRecord],
    w_template_ids: List[int],
    w_unseen_flags: List[bool],
    window_id: int,
    view_name: str,
    split_name: str,
    first_alert_indices: Set[int],
) -> LogWindow:
    line_indices = [r.line_idx for r in w_records]
    timestamps = [r.epoch for r in w_records]
    
    alert_indices = [r.line_idx for r in w_records if r.is_anomaly]
    first_alerts = [idx for idx in alert_indices if idx in first_alert_indices]
    
    is_anomaly = len(alert_indices) > 0
    fault_cat = None
    rack = None
    midplane = None
    node_card = None

    if is_anomaly:
        # Find first anomaly record in window
        first_anom_rec = next(r for r in w_records if r.is_anomaly)
        fault_cat = first_anom_rec.raw_label
        rack = first_anom_rec.location.rack
        midplane = first_anom_rec.location.midplane
        node_card = first_anom_rec.location.node_card

    has_unseen = any(w_unseen_flags)

    return LogWindow(
        window_id=window_id,
        view_name=view_name,
        split_name=split_name,
        start_line_idx=line_indices[0] if line_indices else 0,
        end_line_idx=line_indices[-1] + 1 if line_indices else 0,
        line_indices=line_indices,
        template_ids=w_template_ids,
        timestamps=timestamps,
        is_anomaly=is_anomaly,
        alert_line_indices=alert_indices,
        first_alert_line_indices=first_alerts,
        fault_category=fault_cat,
        first_alert_rack=rack,
        first_alert_midplane=midplane,
        first_alert_node_card=node_card,
        has_unseen_template=has_unseen,
    )


def build_100_msg_windows(
    records: List[BGLRecord],
    template_ids: List[int],
    unseen_flags: List[bool],
    split_name: str = "train",
    first_alert_indices: Optional[Set[int]] = None,
    step_size: int = 100,
) -> List[LogWindow]:
    """
    Constructs 100-message count-based windows (LogLLM-comparable).
    """
    first_alert_indices = first_alert_indices or set()
    n = len(records)
    windows = []
    w_id = 0

    for start in range(0, n, step_size):
        end = min(start + 100, n)
        if end - start < 10:  # Drop degenerate windows with fewer than 10 messages
            continue
        w = _create_window_object(
            records[start:end],
            template_ids[start:end],
            unseen_flags[start:end],
            window_id=w_id,
            view_name="view_100_msg",
            split_name=split_name,
            first_alert_indices=first_alert_indices,
        )
        windows.append(w)
        w_id += 1

    return windows


def build_sliding_20_windows(
    records: List[BGLRecord],
    template_ids: List[int],
    unseen_flags: List[bool],
    split_name: str = "train",
    first_alert_indices: Optional[Set[int]] = None,
    step_size: int = 20,
) -> List[LogWindow]:
    """
    Constructs sliding 20-message windows (Chimera/DeepLog-comparable).
    """
    first_alert_indices = first_alert_indices or set()
    n = len(records)
    windows = []
    w_id = 0

    for start in range(0, n, step_size):
        end = min(start + 20, n)
        if end - start < 5:
            continue
        w = _create_window_object(
            records[start:end],
            template_ids[start:end],
            unseen_flags[start:end],
            window_id=w_id,
            view_name="view_sliding_20",
            split_name=split_name,
            first_alert_indices=first_alert_indices,
        )
        windows.append(w)
        w_id += 1

    return windows


def build_time_windows_5m1m(
    records: List[BGLRecord],
    template_ids: List[int],
    unseen_flags: List[bool],
    split_name: str = "train",
    first_alert_indices: Optional[Set[int]] = None,
    win_sec: int = 300,
    step_sec: int = 60,
) -> List[LogWindow]:
    """
    Constructs 5-minute time windows with 1-minute step (LogBERT-comparable).
    """
    first_alert_indices = first_alert_indices or set()
    n = len(records)
    if n == 0:
        return []

    timestamps = [r.epoch for r in records]
    start_time = timestamps[0]
    end_time = start_time + win_sec
    si, ei = 0, 0

    for cur in timestamps:
        if cur < end_time:
            ei += 1
        else:
            break

    pairs = set()
    if ei > si:
        pairs.add((si, ei))

    while ei < n:
        start_time += step_sec
        end_time = start_time + win_sec
        while si < n and timestamps[si] < start_time:
            si += 1
        while ei < n and timestamps[ei] < end_time:
            ei += 1
        if si != ei:
            pairs.add((si, ei))

    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    windows = []
    w_id = 0

    for (s, e) in sorted_pairs:
        # Require minimum window density
        if e - s < 3:
            continue
        w = _create_window_object(
            records[s:e],
            template_ids[s:e],
            unseen_flags[s:e],
            window_id=w_id,
            view_name="view_5m1m_time",
            split_name=split_name,
            first_alert_indices=first_alert_indices,
        )
        windows.append(w)
        w_id += 1

    return windows


def build_all_window_views(
    records: List[BGLRecord],
    template_ids: List[int],
    unseen_flags: List[bool],
    split_name: str,
    first_alert_indices: Optional[Set[int]] = None,
) -> Dict[str, List[LogWindow]]:
    """
    Builds all three standardized window views across the given chronological partition.
    """
    return {
        "view_100_msg": build_100_msg_windows(
            records, template_ids, unseen_flags, split_name, first_alert_indices
        ),
        "view_sliding_20": build_sliding_20_windows(
            records, template_ids, unseen_flags, split_name, first_alert_indices
        ),
        "view_5m1m_time": build_time_windows_5m1m(
            records, template_ids, unseen_flags, split_name, first_alert_indices
        ),
    }
