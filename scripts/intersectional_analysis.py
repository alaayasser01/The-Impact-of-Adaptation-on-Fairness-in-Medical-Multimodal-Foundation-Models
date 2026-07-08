import os, pickle, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
import torch

# ============= ADAPT THESE TWO PATHS =============
# Path to a predictions.pkl (from an eval_only run for a specific attribute, e.g., Sex)
# PRED_PKL = "/home/aym/Desktop/rhome/aym/FairMedFM/lr_sweep/full-ft-zs_lr5e-3_seed0/cls/full-ft-zs/template/ViT-B_32/erm/ep0_lr0.005/FairVLMed10k/CLIP/All/seed0/full-ft-zs_eval_Sex/predictions.pkl"
PRED_PKL = "baseline/clip_zs/cls/clip-zs/template/ViT-B_32/erm/ep100_lr0.0001/FairVLMed10k/CLIP/All/seed0/clip_zs_final_Sex/predictions.pkl"
# Original test CSV (contains all attributes: gender, race, etc.)
TEST_CSV = "/home/aym/Desktop/rhome/aym/FairMedFM/data/Harvard FairVLMed (Harvard-FairVLMed）/test.csv"
# =================================================

# Load predictions
with open(PRED_PKL, "rb") as f:
    data = pickle.load(f)
logits = data["logits"]
labels = data["label"]
probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

# Load metadata
meta = pd.read_csv(TEST_CSV)
meta["prob"] = probs
meta["true_label"] = labels

# Identify all categorical demographic columns (exclude non-demographic ones)
# Adjust this list to match exactly your CSV column names
dem_cols = ["gender", "race", "language", "ethnicity", "maritalstatus", "age_group"]  # etc.

# Build intersection pairs: only use columns that exist and have <= 10 unique values (categorical)
valid_cols = [c for c in dem_cols if c in meta.columns and meta[c].nunique() <= 10]

# For each pair, compute AUC per combination and the gap
print("\n=== Intersectional AUC analysis ===")
for col1 in valid_cols:
    for col2 in valid_cols:
        if col1 >= col2:   # avoid duplicate pairs (keep order)
            continue
        groups = meta.groupby([col1, col2])
        aucs = {}
        for name, grp in groups:
            if grp.shape[0] < 5:   # skip too small groups
                continue
            auc = roc_auc_score(grp["true_label"], grp["prob"])
            aucs[name] = auc
        if len(aucs) < 2:
            continue
        gap = max(aucs.values()) - min(aucs.values())
        print(f"\n{col1} × {col2}   (gap={gap:.4f})")
        for name, auc_val in sorted(aucs.items(), key=lambda x: x[1], reverse=True):
            count = meta.groupby([col1, col2]).size()[name]
            print(f"  {str(name):30s}  n={count:4d}  AUC={auc_val:.4f}")