import os
import glob
import re
import pandas as pd

# ---> EXACT PATH FOR FAIRVLMED10K CLIP RESULTS <---
RESULTS_DIR = "/home/alaa/papersubmission/The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-Foundation-Models/FairVLMed10k_clip_results"

usages = ["clip-zs", "lp", "clip-adapt", "lora-lp", "lora-zs", "full-ft", "full-ft-zs"]
SENSITIVE_ATTRS = ["Sex", "Age", "Race", "Language", "Ethnicity", "MaritalStatus"]

data = []

for usage in usages:
    row = {"Usage": usage, "Best_Epoch": "N/A"}
    
    log_files = glob.glob(os.path.join(RESULTS_DIR, f"{usage}_lr*", "**", "*.log"), recursive=True)
    if not log_files:
        # Fallback to general pattern if lr is not in folder name
        log_files = glob.glob(os.path.join(RESULTS_DIR, f"{usage}_*", "**", "*.log"), recursive=True)
    
    log_files = sorted(log_files, key=os.path.getmtime)
    
    for log_path in log_files:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # --- 1. Extract Test AUC (Overall) ---
        aucs = re.findall(r"overall-auc:\s*([\d\.]+)", content)
        if aucs:
            row["Test_AUC"] = round(float(aucs[-1]), 4)

        # --- 2. Extract AUC GAP for ALL Attributes ---
        chunks = re.split(r"--- Evaluating ", content)
        for chunk in chunks:
            for attr in SENSITIVE_ATTRS:
                attr_spaced = attr.replace("Status", " Status") if attr == "MaritalStatus" else attr
                
                if (chunk.startswith(attr) or 
                    chunk.startswith(attr_spaced) or 
                    (f"sensitive_name='{attr}'" in chunk and "overall-auc:" in chunk)):
                    
                    # EXTRACT AUC-GAP INSTEAD OF EOD
                    gaps = re.findall(r"auc-gap:\s*([\d\.]+)", chunk)
                    if gaps: 
                        row[f"{attr}_AUC_Gap"] = round(float(gaps[-1]), 4)

    data.append(row)

# --- 3. Build DataFrame and Compute Summary Metrics ---
df = pd.DataFrame(data)

gap_cols = [f"{attr}_AUC_Gap" for attr in SENSITIVE_ATTRS if f"{attr}_AUC_Gap" in df.columns]

if gap_cols:
    df["Mean_AUC_Gap"] = round(df[gap_cols].mean(axis=1), 4)
    df["Worst_AUC_Gap"] = df[gap_cols].max(axis=1) # MAX because higher gap is worse
    df["Worst_Attr"] = df[gap_cols].idxmax(axis=1).str.replace("_AUC_Gap", "")

base_cols = ["Usage", "Test_AUC"]
summary_cols = ["Mean_AUC_Gap", "Worst_AUC_Gap", "Worst_Attr"]
final_cols = base_cols + gap_cols + summary_cols

final_cols = [c for c in final_cols if c in df.columns]
df = df[final_cols]

# --- 4. Save and Print ---
out_file = os.path.join(RESULTS_DIR, "Final_AUC_Gap_Summary.csv")
df.to_csv(out_file, index=False)

print("\n" + "="*120)
print("MARGINAL AUC GAP SUMMARY (FAIRVLMED10K CLIP)")
print("="*120)
print(df.to_markdown(index=False))