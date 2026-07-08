#!/usr/bin/env python3
"""
intersectional_eval.py — run separately on remote machine.
Computes intersectional Δ-AUC heatmap for each method × attribute pair.

Requirements: test.csv must have columns matching your sensitive attributes.
"""
import pickle, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = "/home/aym/Desktop/rhome/aym/FairMedFM"
TEST_CSV = f"{PROJECT_ROOT}/data/Harvard FairVLMed (Harvard-FairVLMed）/test.csv"

# Load test metadata
df = pd.read_csv(TEST_CSV)
print("Test CSV columns:", df.columns.tolist())
print(f"Test set size: {len(df)}")

# Age quartile boundaries (from training set)
AGE_BINS = [51.2, 63.9, 73.3]
df["Age_Group"] = pd.cut(df["Age"], bins=[-np.inf] + AGE_BINS + [np.inf],
                          labels=[0, 1, 2, 3]).astype(int)

# Attribute column mapping (adjust to match your CSV column names)
ATTR_COLS = {
    "Gender":   "Sex",          # adjust if column is named differently
    "Age":      "Age_Group",
    "Race":     "Race",
    "Language": "Language",
    "Ethnicity":"Ethnicity",
    "MaritalStatus": "MaritalStatus",
}

def intersect_delta_auc(probs, labels, a1_vals, a2_vals, min_n=10):
    cats1 = sorted(a1_vals.unique())
    cats2 = sorted(a2_vals.unique())
    cells = {}
    for c1 in cats1:
        for c2 in cats2:
            mask = (a1_vals == c1) & (a2_vals == c2)
            n = mask.sum()
            if n < min_n or labels[mask].nunique() < 2:
                cells[(c1, c2)] = (float("nan"), n)
                continue
            cells[(c1, c2)] = (roc_auc_score(labels[mask], probs[mask]), n)
    valid = [v for v, n in cells.values() if not np.isnan(v)]
    delta = max(valid) - min(valid) if len(valid) >= 2 else float("nan")
    return delta, cells

# Methods and their prediction pickle paths (adjust paths as needed)
# Predictions should be saved by your evaluate() function as .pkl files
# containing {"logits": array, "label": array}

PRED_PATHS = {
    "clip-zs":       f"{PROJECT_ROOT}/baseline/clip_zs/cls/clip-zs/template/ViT-B_32/erm/ep100_lr0.0001/FairVLMed10k/CLIP/All/seed0/predictions.pkl",
    "lora-text":     f"{PROJECT_ROOT}/output_seed0/cls/lora-text/template/ViT-B_32/erm/ep100_lr0.01/FairVLMed10k/CLIP/All/seed0/predictions.pkl",
    "full-ft-text":  f"{PROJECT_ROOT}/output_seed0/cls/full-ft-text/template/ViT-B_32/erm/ep100_lr0.005/FairVLMed10k/CLIP/All/seed0/predictions.pkl",
}

import torch.nn.functional as F
import torch

pairs = [("Gender", "Age"), ("Age", "Language"), ("Age", "Race")]

results = {}
for method, pkl_path in PRED_PATHS.items():
    print(f"\n[{method}]  {pkl_path}")
    try:
        with open(pkl_path, "rb") as f:
            pred_data = pickle.load(f)
        logits = pred_data.get("logits") or pred_data.get("predictions")
        labels = pred_data.get("label") or pred_data.get("labels")
        # Convert logits to probs
        import scipy.special
        probs = scipy.special.softmax(logits, axis=-1)[:, 1]
        labels = np.array(labels, dtype=int)
        print(f"  n={len(labels)}  pos_rate={labels.mean():.2f}")
        results[method] = {}
        for a1_name, a2_name in pairs:
            a1 = df[ATTR_COLS[a1_name]]
            a2 = df[ATTR_COLS[a2_name]]
            d, cells = intersect_delta_auc(probs, labels, a1, a2)
            results[method][f"{a1_name}×{a2_name}"] = d
            print(f"  {a1_name}×{a2_name}: Δ-AUC = {d:.4f}")
    except FileNotFoundError:
        print(f"  [MISSING] {pkl_path}")

import json
with open("intersectional_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n✓ Saved intersectional_results.json")
