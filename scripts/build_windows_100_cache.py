"""
Pre-processes full 4.75M BGL stream into synchronized 100-message windows
with Plan 0 chronological split, message IDs, Drain template IDs, delta_t,
event-level labels, window labels, and category metadata.
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plan0_protocol.parser import mask_log_message


def main():
    print("=" * 70)
    print("BUILDING AUTHORITATIVE 100-MESSAGE WINDOWS CACHE (FULL BGL)")
    print("=" * 70)
    t0 = time.time()

    # Load msg_to_id from minilm_embeddings_cache.pt
    print("Loading MiniLM message vocabulary cache...")
    cache_emb = torch.load("data/cache/minilm_embeddings_cache.pt", map_location="cpu")
    msg_to_id = cache_emb["msg_to_id"]
    print(f"  Loaded vocabulary of {len(msg_to_id):,} unique masked messages.")

    # Load drain parsed data
    print("Loading Drain template cluster sequence...")
    with open("data/cache/bgl_parsed.pkl", "rb") as f:
        bgl_parsed = pickle.load(f)
    event_ids_drain = bgl_parsed["event_ids"]
    print(f"  Loaded {len(event_ids_drain):,} Drain event IDs.")

    win_msg_ids = []
    win_template_ids = []
    win_delta_ts = []
    win_event_labels = []
    win_labels = []
    win_categories = []
    win_first_alert_pos = []

    curr_msg_ids = []
    curr_delta_ts = []
    curr_labels = []
    curr_anom_tags = []
    prev_epoch = 0

    print("Streaming through 4,747,963 BGL log lines...")
    with open("data/BGL/BGL.log", "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            parts = line.split(maxsplit=9)
            if len(parts) < 9:
                continue
            tag = parts[0]
            try:
                epoch = int(parts[1])
            except ValueError:
                epoch = prev_epoch
            dt = max(0, epoch - prev_epoch) if prev_epoch > 0 else 0
            prev_epoch = epoch

            msg = parts[9].strip() if len(parts) > 9 else ""
            masked = mask_log_message(msg)
            mid = msg_to_id.get(masked, msg_to_id.get("", 0))

            is_anom = 1 if tag != "-" else 0
            curr_msg_ids.append(mid)
            curr_delta_ts.append(dt)
            curr_labels.append(is_anom)
            if is_anom:
                curr_anom_tags.append(tag)

            if len(curr_msg_ids) == 100:
                win_msg_ids.append(np.array(curr_msg_ids, dtype=np.int32))
                win_delta_ts.append(np.array(curr_delta_ts, dtype=np.float32))
                win_event_labels.append(np.array(curr_labels, dtype=np.int8))
                has_anom = 1 if len(curr_anom_tags) > 0 else 0
                win_labels.append(has_anom)
                win_categories.append(curr_anom_tags[0] if has_anom else "NORMAL")

                # First alert position within the window
                if has_anom:
                    first_pos = curr_labels.index(1)
                else:
                    first_pos = -1
                win_first_alert_pos.append(first_pos)

                # Slice Drain template IDs for this window
                start_idx = len(win_msg_ids) * 100 - 100
                end_idx = start_idx + 100
                win_template_ids.append(np.array(event_ids_drain[start_idx:end_idx], dtype=np.int32))

                curr_msg_ids = []
                curr_delta_ts = []
                curr_labels = []
                curr_anom_tags = []

    total_wins = len(win_labels)
    n_train = int(total_wins * 0.72)
    n_val = int(total_wins * 0.08)
    n_test = total_wins - n_train - n_val

    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

    windows_data = {
        "total_windows": total_wins,
        "win_msg_ids": np.stack(win_msg_ids),             # (47479, 100) int32
        "win_template_ids": np.stack(win_template_ids),   # (47479, 100) int32
        "win_delta_ts": np.stack(win_delta_ts),           # (47479, 100) float32
        "win_event_labels": np.stack(win_event_labels),   # (47479, 100) int8
        "win_labels": np.array(win_labels, dtype=np.int8), # (47479,) int8
        "win_categories": win_categories,                 # list of str
        "win_first_alert_pos": np.array(win_first_alert_pos, dtype=np.int16),
        "splits": splits,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }

    out_file = Path("data/cache/bgl_windows_100.pkl")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(windows_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"\n[DONE] Built and cached {total_wins:,} windows in {time.time() - t0:.2f}s!")
    print(f"  Train: {n_train:,} windows")
    print(f"  Val:   {n_val:,} windows")
    print(f"  Test:  {n_test:,} windows")
    print(f"  Cache file saved: {out_file} ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
