"""
Chronological Splitting, Rolling-Origin Slices, and Unseen Sequence Partitioning for Plan 0.
Enforces strict chronological partitioning:
- 80/20 primary split (train = first 72%, val = next 8%, test = final 20%)
- Inter-split window gap buffer to eliminate temporal leakage
- Rolling-origin monthly evaluation (train on months 1..k, test on month k+1)
- 'All-test' vs 'Unseen-only test' (filtering exact training sequence matches)
- Drift diagnostic (share of test windows with unseen templates)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta

from .parser import BGLRecord
from .windowing import LogWindow


@dataclass
class ChronologicalSplitResult:
    train_records: List[BGLRecord]
    val_records: List[BGLRecord]
    test_records: List[BGLRecord]
    gap1_records_dropped: int
    gap2_records_dropped: int
    train_time_range: Tuple[int, int]
    val_time_range: Tuple[int, int]
    test_time_range: Tuple[int, int]


def create_chronological_splits(
    records: List[BGLRecord],
    train_ratio: float = 0.72,
    val_ratio: float = 0.08,
    test_ratio: float = 0.20,
    gap_sec: int = 300,
) -> ChronologicalSplitResult:
    """
    Creates leakage-controlled chronological splits:
    - Train: first 72%
    - Gap 1: 300s window duration gap
    - Val: next 8%
    - Gap 2: 300s window duration gap
    - Test: final 20%
    """
    n = len(records)
    if n == 0:
        raise ValueError("Cannot split empty record list.")

    # Chronologically sort to guarantee stream monotonicity
    sorted_recs = sorted(records, key=lambda r: (r.epoch, r.line_idx))

    idx_train_end = int(n * train_ratio)
    train_recs = sorted_recs[:idx_train_end]
    train_last_epoch = train_recs[-1].epoch if train_recs else 0

    idx_val_target_end = int(n * (train_ratio + val_ratio))
    val_max_gap_idx = idx_train_end + max(0, int((idx_val_target_end - idx_train_end) * 0.25))

    # Gap 1: skip records within gap_sec from train_last_epoch, capped to prevent val starvation
    val_start_idx = idx_train_end
    while val_start_idx < val_max_gap_idx and (sorted_recs[val_start_idx].epoch - train_last_epoch) < gap_sec:
        val_start_idx += 1

    gap1_dropped = val_start_idx - idx_train_end
    val_recs = sorted_recs[val_start_idx:idx_val_target_end]
    val_last_epoch = val_recs[-1].epoch if val_recs else (train_last_epoch + gap_sec)

    # Gap 2: skip records within gap_sec from val_last_epoch, capped to prevent test starvation
    test_target_start = idx_val_target_end
    test_max_gap_idx = test_target_start + max(0, int((n - test_target_start) * 0.25))
    test_start_idx = test_target_start
    while test_start_idx < test_max_gap_idx and (sorted_recs[test_start_idx].epoch - val_last_epoch) < gap_sec:
        test_start_idx += 1

    gap2_dropped = test_start_idx - test_target_start
    test_recs = sorted_recs[test_start_idx:]

    return ChronologicalSplitResult(
        train_records=train_recs,
        val_records=val_recs,
        test_records=test_recs,
        gap1_records_dropped=gap1_dropped,
        gap2_records_dropped=gap2_dropped,
        train_time_range=(train_recs[0].epoch, train_recs[-1].epoch) if train_recs else (0, 0),
        val_time_range=(val_recs[0].epoch, val_recs[-1].epoch) if val_recs else (0, 0),
        test_time_range=(test_recs[0].epoch, test_recs[-1].epoch) if test_recs else (0, 0),
    )


def create_rolling_origin_splits(
    records: List[BGLRecord],
    block_days: int = 30,
    gap_sec: int = 300,
    min_slices: int = 3,
) -> List[Dict[str, Any]]:
    """
    Creates monthly rolling-origin splits:
    Origin k: Train on first k months/blocks, test on month/block k+1.
    Measures performance drift decay across the timeline.
    """
    if not records:
        return []

    sorted_recs = sorted(records, key=lambda r: (r.epoch, r.line_idx))
    start_epoch = sorted_recs[0].epoch
    total_duration = max(1, sorted_recs[-1].epoch - start_epoch)
    
    sec_per_block = block_days * 86400
    if total_duration < sec_per_block * 2:
        # Dynamically size blocks if dataset slice is shorter than 2 full blocks
        sec_per_block = max(60, total_duration // (min_slices + 1))

    # Bucket records into blocks
    blocks: List[List[BGLRecord]] = []
    current_block_idx = 0
    current_block_recs: List[BGLRecord] = []

    for r in sorted_recs:
        b_idx = int((r.epoch - start_epoch) // sec_per_block)
        while b_idx > current_block_idx:
            if current_block_recs:
                blocks.append(current_block_recs)
                current_block_recs = []
            current_block_idx += 1
        current_block_recs.append(r)
    if current_block_recs:
        blocks.append(current_block_recs)

    rolling_splits = []
    n_blocks = len(blocks)

    for k in range(1, n_blocks):
        # Train on blocks 0..k-1
        train_pool = []
        for b in range(k):
            train_pool.extend(blocks[b])
        
        # Test on block k
        test_pool = blocks[k]

        if not train_pool or not test_pool:
            continue

        # Split train pool into train (90%) and val (10%) with gap
        split_res = create_chronological_splits(
            train_pool,
            train_ratio=0.90,
            val_ratio=0.10,
            test_ratio=0.00,
            gap_sec=gap_sec,
        )

        rolling_splits.append({
            "origin_k": k,
            "train_months": f"1..{k}",
            "test_month": f"{k+1}",
            "train_records": split_res.train_records,
            "val_records": split_res.val_records,
            "test_records": test_pool,
        })

    return rolling_splits


def filter_unseen_windows(
    train_windows: List[LogWindow],
    test_windows: List[LogWindow],
) -> Tuple[List[LogWindow], List[LogWindow], Dict[str, Any]]:
    """
    Partitions test windows into:
    - 'all_test': All test windows
    - 'unseen_only_test': Drops test windows whose exact event sequence appears in train.
    
    Also computes the drift diagnostic:
    - Share of test windows with templates unseen in train.
    """
    train_sequence_set: Set[Tuple[int, ...]] = {w.sequence_tuple for w in train_windows}

    unseen_only_test = [
        w for w in test_windows if w.sequence_tuple not in train_sequence_set
    ]

    total_test = len(test_windows)
    unseen_seq_count = len(unseen_only_test)
    unseen_tmpl_windows = sum(1 for w in test_windows if w.has_unseen_template)

    stats = {
        "total_test_windows": total_test,
        "unseen_sequence_windows": unseen_seq_count,
        "unseen_sequence_ratio": unseen_seq_count / max(1, total_test),
        "windows_with_unseen_templates": unseen_tmpl_windows,
        "share_test_windows_unseen_templates": unseen_tmpl_windows / max(1, total_test),
    }

    return test_windows, unseen_only_test, stats
