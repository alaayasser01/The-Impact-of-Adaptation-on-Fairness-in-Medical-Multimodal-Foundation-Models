# import os
# import re
# import glob
# import pandas as pd
# import numpy as np

# SWEEP_DIR = "/home/aym/Desktop/rhome/aym/FairMedFM/lr_sweep"
# OUT_DIR = "/home/aym/Desktop/rhome/aym/FairMedFM/poster_data"
# os.makedirs(OUT_DIR, exist_ok=True)

# USAGES = ["lp", "clip-adapt", "full-ft", "full-ft-zs", "lora-lp", "lora-zs"]
# SEEDS = [0, 1, 2]
# ATTRIBUTES = ["Sex", "Age", "Race", "Language", "Ethnicity", "MaritalStatus"]

# BEST_LRS = {
#     "lp": "1e-2",
#     "clip-adapt": "1e-2",
#     "full-ft": "1e-2",
#     "full-ft-zs": "5e-3",
#     "lora-lp": "1e-2",
#     "lora-zs": "1e-2",
# }

# def find_latest_log(folder):
#     logs = glob.glob(os.path.join(folder, "**", "history*.log"), recursive=True)
#     return max(logs, key=os.path.getmtime) if logs else None

# def extract_attr_metrics_from_log(log_text):
#     metrics = {}
#     for attr in ATTRIBUTES:
#         # Match overall-auc and eod blocks per attribute
#         m = re.search(rf"--- Evaluating {attr} ---.*?overall-auc:\s*([\d.]+).*?eod:\s*([\d.]+)", log_text, re.DOTALL)
#         if m:
#             metrics[attr] = {"AUC": float(m.group(1)), "EoD": float(m.group(2))}
#     return metrics

# # 1. Gather all data
# all_results = []

# for usage in USAGES:
#     lr = BEST_LRS[usage]
#     usage_seed_metrics = []
    
#     for seed in SEEDS:
#         exp_dir = os.path.join(SWEEP_DIR, f"{usage}_lr{lr}_seed{seed}")
#         log_file = find_latest_log(exp_dir)
        
#         if log_file:
#             with open(log_file, "r") as f:
#                 content = f.read()
#             metrics = extract_attr_metrics_from_log(content)
#             if metrics:
#                 # Calculate average AUC and EoD across all attributes for this seed
#                 avg_auc = np.mean([v["AUC"] for v in metrics.values()])
#                 avg_eod = np.mean([v["EoD"] for v in metrics.values()])
                
#                 usage_seed_metrics.append({
#                     "Seed": seed,
#                     "Avg_AUC": avg_auc,
#                     "Avg_EoD": avg_eod
#                 })
    
#     # 2. Compute Mean and Std
#     if usage_seed_metrics:
#         aucs = [x["Avg_AUC"] for x in usage_seed_metrics]
#         eods = [x["Avg_EoD"] for x in usage_seed_metrics]
        
#         mean_auc, std_auc = np.mean(aucs), np.std(aucs)
#         mean_eod, std_eod = np.mean(eods), np.std(eods)
        
#         # Format string as "Mean ± Std"
#         all_results.append({
#             "Usage": usage,
#             "Best LR": lr,
#             "Overall AUC (Mean ± Std)": f"{mean_auc:.4f} ± {std_auc:.4f}",
#             "Mean EoD (Mean ± Std)": f"{mean_eod:.4f} ± {std_eod:.4f}",
#             "Seeds Evaluated": len(usage_seed_metrics)
#         })

# # 3. Save to CSV
# df_merged = pd.DataFrame(all_results)
# merged_csv_path = os.path.join(OUT_DIR, "merged_seeds_summary.csv")
# df_merged.to_csv(merged_csv_path, index=False)

# print(f"✅ Successfully computed Mean and Std across {len(SEEDS)} seeds.")
# print(f"📄 Saved to: {merged_csv_path}")
# print(df_merged.to_string())

import os
import re
import glob
import pandas as pd
import numpy as np
import torch  # We will use torch to peek inside the checkpoint files!

SWEEP_DIR = "./lr_sweep"
OUT_DIR = "./poster_data"
os.makedirs(OUT_DIR, exist_ok=True)

USAGES = ["lp", "clip-adapt", "full-ft", "full-ft-zs", "lora-lp", "lora-zs"]
SEEDS = [0, 1, 2]
ATTRIBUTES = ["Sex", "Age", "Race", "Language", "Ethnicity", "MaritalStatus"]

BEST_LRS = {
    "lp": "1e-2",
    "clip-adapt": "1e-2",
    "full-ft": "1e-2",
    "full-ft-zs": "5e-3",
    "lora-lp": "1e-2",
    "lora-zs": "1e-2",
}

def find_latest_log(folder):
    logs = glob.glob(os.path.join(folder, "**", "history*.log"), recursive=True)
    return max(logs, key=os.path.getmtime) if logs else None

all_results = []

# --- 1. Process Trainable Usages ---
for usage in USAGES:
    lr = BEST_LRS[usage]
    
    seed_epochs = {0: "N/A", 1: "N/A", 2: "N/A"}
    seed_attr_metrics = {0: {}, 1: {}, 2: {}}
    overall_aucs = []
    overall_eods = []
    
    for seed in SEEDS:
        exp_dir = os.path.join(SWEEP_DIR, f"{usage}_lr{lr}_seed{seed}")
        
        # 🟢 BULLETPROOF EPOCH EXTRACTION: Open the .pth file directly!
        ckpt_path = os.path.join(exp_dir, "best_ckpt.pth")
        if os.path.exists(ckpt_path):
            try:
                # Load just the metadata to CPU to save memory
                ckpt = torch.load(ckpt_path, map_location="cpu")
                seed_epochs[seed] = ckpt.get("epoch", "N/A")
            except Exception as e:
                pass
        
        # 🟢 EXTRACT METRICS FROM LATEST LOG
        log_file = find_latest_log(exp_dir)
        if log_file:
            with open(log_file, "r") as f:
                content = f.read()
            
            metrics = {}
            for attr in ATTRIBUTES:
                m = re.search(rf"--- Evaluating {attr} ---.*?overall-auc:\s*([\d.]+).*?eod:\s*([\d.]+)", content, re.DOTALL)
                if m:
                    metrics[attr] = {"AUC": float(m.group(1)), "EoD": float(m.group(2))}
            
            if metrics:
                seed_attr_metrics[seed] = metrics
                overall_aucs.append(np.mean([v["AUC"] for v in metrics.values()]))
                overall_eods.append(np.mean([v["EoD"] for v in metrics.values()]))

    if overall_aucs:
        mean_auc, std_auc = np.mean(overall_aucs), np.std(overall_aucs)
        mean_eod, std_eod = np.mean(overall_eods), np.std(overall_eods)
        
        # 🟢 FIND THE WORST ATTRIBUTE (Lowest average EoD across all available seeds)
        worst_attr = None
        worst_mean_eod = float('inf')
        
        for attr in ATTRIBUTES:
            eods = [seed_attr_metrics[s][attr]["EoD"] for s in SEEDS if attr in seed_attr_metrics[s]]
            if eods:
                m_eod = np.mean(eods)
                if m_eod < worst_mean_eod:
                    worst_mean_eod = m_eod
                    worst_attr = attr

        # Get the exact EoD for that specific worst attribute per seed
        w_eod_s0 = f"{seed_attr_metrics[0][worst_attr]['EoD']:.4f}" if 0 in seed_attr_metrics and worst_attr in seed_attr_metrics[0] else "N/A"
        w_eod_s1 = f"{seed_attr_metrics[1][worst_attr]['EoD']:.4f}" if 1 in seed_attr_metrics and worst_attr in seed_attr_metrics[1] else "N/A"
        w_eod_s2 = f"{seed_attr_metrics[2][worst_attr]['EoD']:.4f}" if 2 in seed_attr_metrics and worst_attr in seed_attr_metrics[2] else "N/A"

        all_results.append({
            "Usage": usage,
            "LR": lr,
            "AUC (Mean ± Std)": f"{mean_auc:.4f} ± {std_auc:.4f}",
            "EoD (Mean ± Std)": f"{mean_eod:.4f} ± {std_eod:.4f}",
            "Worst Attr": worst_attr,
            "W_EoD S0": w_eod_s0,
            "W_EoD S1": w_eod_s1,
            "W_EoD S2": w_eod_s2,
            "Ep S0": seed_epochs[0],
            "Ep S1": seed_epochs[1],
            "Ep S2": seed_epochs[2]
        })

# --- 2. Process Baseline (BiomedCLIP Zero-Shot) ---
baseline_dir = "./baseline/biomedclip_zs"
baseline_log = find_latest_log(baseline_dir)

if baseline_log:
    with open(baseline_log, "r") as f:
        content = f.read()
    
    b_metrics = {}
    for attr in ATTRIBUTES:
        m = re.search(rf"--- Evaluating {attr} ---.*?overall-auc:\s*([\d.]+).*?eod:\s*([\d.]+)", content, re.DOTALL)
        if m:
            b_metrics[attr] = {"AUC": float(m.group(1)), "EoD": float(m.group(2))}
    
    if b_metrics:
        b_avg_auc = np.mean([v["AUC"] for v in b_metrics.values()])
        b_avg_eod = np.mean([v["EoD"] for v in b_metrics.values()])
        
        b_worst_attr = min(b_metrics.keys(), key=lambda k: b_metrics[k]["EoD"])
        b_worst_eod = f"{b_metrics[b_worst_attr]['EoD']:.4f}"
        
        all_results.insert(0, {
            "Usage": "biomedclip-zs",
            "LR": "N/A",
            "AUC (Mean ± Std)": f"{b_avg_auc:.4f} ± 0.0000",
            "EoD (Mean ± Std)": f"{b_avg_eod:.4f} ± 0.0000",
            "Worst Attr": b_worst_attr,
            "W_EoD S0": b_worst_eod,
            "W_EoD S1": "N/A",  # Zero-shot is deterministic, no variance across seeds
            "W_EoD S2": "N/A",
            "Ep S0": "N/A",
            "Ep S1": "N/A",
            "Ep S2": "N/A"
        })

# --- 3. Save to CSV ---
df_merged = pd.DataFrame(all_results)
merged_csv_path = os.path.join(OUT_DIR, "merged_seeds_summary_enhanced.csv")
df_merged.to_csv(merged_csv_path, index=False)

print("\n✅ Enhanced Merged Summary Generated!")
print(f"📄 Saved to: {merged_csv_path}\n")
print(df_merged.to_string())