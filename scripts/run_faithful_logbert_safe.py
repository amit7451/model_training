"""
Master Safe Runner for Faithful Original LogBERT on Full BGL Dataset
Designed specifically for Laptop NVIDIA RTX 4060 GPU:
- VRAM ceiling at 65% to prevent Desktop Window Manager (dwm.exe) starvation
- Cooperative micro-yields (time.sleep & torch.cuda.synchronize) to eliminate screen glitching/tearing
- Automatic Mixed Precision (AMP FP16) for Tensor Core efficiency and cooler thermals
- Fully executes all 18 code blocks and populates the notebook for the evaluator
"""

import os
import sys
import time
import json
import random
import math
import gc
import pickle
import hashlib
import warnings
import platform
import base64
import io
import contextlib
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication plots
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# ─────────────────────────────────────────────────────────────────────────────
# Laptop RTX 4060 GPU Safety Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    # Reserve 35% of VRAM for Windows OS and Desktop Window Manager (dwm.exe)
    torch.cuda.set_per_process_memory_fraction(0.65, 0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[GPU Safety Active] Configured {gpu_name} with 65% VRAM cap & display sync.")
else:
    gpu_name = "CPU"

def safe_yield(interval=10, step=0):
    """Periodically synchronize and sleep 1ms so Windows DWM gets display refresh slots."""
    if torch.cuda.is_available() and step % interval == 0:
        torch.cuda.synchronize()
        time.sleep(0.001)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

def make_stream_output(text):
    lines = [l + '\n' for l in text.splitlines()]
    return {
        "name": "stdout",
        "output_type": "stream",
        "text": lines
    }

def make_image_output(img_b64):
    return {
        "data": {
            "image/png": img_b64,
            "text/plain": ["<Figure size ...>"]
        },
        "metadata": {},
        "output_type": "display_data"
    }

print("=" * 70)
print("STARTING SAFE FAITHFUL LOGBERT TRAINING & EVALUATION ON FULL BGL DATASET")
print("=" * 70)

def resolve_first(paths, default_val):
    for p in paths:
        candidate = Path(p)
        if candidate.exists():
            return candidate
    return Path(default_val)

OUTPUT_DIR = resolve_first(['results/logbert_faithful', '../results/logbert_faithful'], 'results/logbert_faithful')
MODEL_DIR = resolve_first(['models/logbert_faithful', '../models/logbert_faithful'], 'models/logbert_faithful')
RESULTS_DIR = OUTPUT_DIR
CACHE_DIR = resolve_first(['data/cache', '../data/cache'], 'data/cache')
for d in [OUTPUT_DIR, MODEL_DIR, RESULTS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load existing template notebook
nb_template_path = resolve_first([
    r"c:\Users\AMIT\model_training\notebooks\04_logbert_faithful_train_eval.ipynb",
    r"notebooks/04_logbert_faithful_train_eval.ipynb",
    r"../notebooks/04_logbert_faithful_train_eval.ipynb"
], r"notebooks/04_logbert_faithful_train_eval.ipynb")
with open(nb_template_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Global execution state (mirroring Jupyter kernel __main__ namespace)
exec_globals = sys.modules['__main__'].__dict__
exec_globals['__file__'] = str(nb_template_path)
exec_globals['safe_yield'] = safe_yield

code_cell_indices = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
print(f"Total cells: {len(nb['cells'])}, Code cells to execute: {len(code_cell_indices)}")

cell_exec_count = 1

for code_idx, cell_i in enumerate(code_cell_indices):
    cell = nb['cells'][cell_i]
    code_text = "".join(cell['source'])
    print(f"\n[{code_idx+1}/{len(code_cell_indices)}] Executing Notebook Cell {cell_i}...")
    t_start = time.time()
    
    cell['execution_count'] = cell_exec_count
    cell_exec_count += 1
    cell['outputs'] = []
    
    plt.close('all')
    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_buf
    
    error_occurred = None
    try:
        # Before Cell 20 (training) or Cell 24 (inference), ensure safe yields and weights_only=False
        exec(code_text, exec_globals)
    except Exception as e:
        error_occurred = e
    finally:
        sys.stdout = old_stdout
    
    out_text = stdout_buf.getvalue()
    if out_text:
        cell['outputs'].append(make_stream_output(out_text))
        summary_lines = out_text.strip().splitlines()
        if len(summary_lines) <= 8:
            print("\n".join(summary_lines))
        else:
            print("\n".join(summary_lines[:4]))
            print(f"  ... [{len(summary_lines) - 6} lines truncated for brevity] ...")
            print("\n".join(summary_lines[-2:]))
            
    # Check if a plot was created
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        img_b64 = fig_to_base64(fig)
        cell['outputs'].append(make_image_output(img_b64))
        print("  [Captured Matplotlib Plot]")
        
    if error_occurred:
        print(f"ERROR in Cell {cell_i}: {error_occurred}")
        import traceback
        traceback.print_exc()
        raise error_occurred
        
    print(f"Cell {cell_i} finished in {time.time()-t_start:.2f}s")

out_paths = [nb_template_path]

for p in out_paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"\n[Saved Populated Evaluator Notebook] -> {p.resolve()}")

print("\n" + "=" * 70)
print("ALL CELLS COMPLETED AND FULLY POPULATED!")
print("=" * 70)
