# import os
# import glob
# import re
# import pandas as pd

# # ---> CHANGE THIS TO YOUR ACTUAL RESULTS DIRECTORY <---
# RESULTS_DIR = "/home/alaa/papersubmission/The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-Foundation-Models/FairVLMed10k_clip_results"

# usages = ["clip-zs", "lp", "clip-adapt", "full-ft", "lora-lp", "lora-zs", "full-ft-zs"]
# data = []

# for usage in usages:
#     row = {"Usage": usage, "Best_Epoch": "N/A"}
    
#     # Grab all log files for this specific usage
#     log_files = glob.glob(os.path.join(RESULTS_DIR, f"{usage}_*", "**", "*.log"), recursive=True)
    
#     for log_path in log_files:
#         with open(log_path, "r", encoding="utf-8") as f:
#             content = f.read()
        
#         # --- 1. Extract Best Epoch ---
#         if row["Best_Epoch"] == "N/A":
#             ep_match = re.search(r"(?:Loaded best checkpoint from epoch|best val AUC .* at epoch)\s+(\d+)", content, re.IGNORECASE)
#             if ep_match:
#                 row["Best_Epoch"] = ep_match.group(1)

#         # --- 2. Extract Test AUC (Overall) ---
#         # Since the test set is identical for Sex and Age, we just grab the last overall-auc recorded
#         aucs = re.findall(r"overall-auc:\s*([\d\.]+)", content)
#         if aucs:
#             row["Test_AUC"] = round(float(aucs[-1]), 4)

#         # --- 3. Extract EOD ---
#         chunks = re.split(r"--- Evaluating ", content)
#         for chunk in chunks:
#             attr = None
#             if chunk.startswith("Sex") or ("sensitive_name='Sex'" in chunk and "overall-auc:" in chunk):
#                 attr = "Sex"
#             elif chunk.startswith("Age") or ("sensitive_name='Age'" in chunk and "overall-auc:" in chunk):
#                 attr = "Age"
            
#             if attr:
#                 eods = re.findall(r"eod:\s*([\d\.]+)", chunk)
#                 if eods: 
#                     row[f"{attr}_EOD"] = round(float(eods[-1]), 4)

#     data.append(row)

# # --- 4. Build DataFrame and Compute Mean EOD ---
# df = pd.DataFrame(data)

# if "Sex_EOD" in df.columns and "Age_EOD" in df.columns:
#     df["Mean_EOD"] = round(df[["Sex_EOD", "Age_EOD"]].mean(axis=1), 4)

# # Organize columns beautifully
# cols = ["Usage", "Best_Epoch", "Test_AUC", "Sex_EOD", "Age_EOD", "Mean_EOD"]
# final_cols = [c for c in cols if c in df.columns]
# df = df[final_cols]

# # --- 5. Save and Print ---
# out_file = os.path.join(RESULTS_DIR, "Final_Metrics_Summary.csv")
# df.to_csv(out_file, index=False)

# print("\n" + "="*80)
# print("FINAL FAIRNESS METRICS SUMMARY")
# print("="*80)
# print(df.to_markdown(index=False))
# print("\n=> Successfully saved to:", out_file)

import os
import glob
import re
import pandas as pd

# ---> CHANGE THIS TO YOUR ACTUAL RESULTS DIRECTORY <---
RESULTS_DIR = "/home/alaa/papersubmission/The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-Foundation-Models/Fairmedvlm_biomedclip_results_BINARY"

usages = ["clip-zs", "lp", "clip-adapt", "full-ft", "lora-lp", "lora-zs", "full-ft-zs"]
SENSITIVE_ATTRS = ["Sex", "Age", "Race", "Language", "Ethnicity", "MaritalStatus"]

data = []

for usage in usages:
    row = {"Usage": usage, "Best_Epoch": "N/A"}
    
    # Grab all log files for this specific usage
    log_files = glob.glob(os.path.join(RESULTS_DIR, f"{usage}_*", "**", "*.log"), recursive=True)
    
    for log_path in log_files:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # --- 1. Extract Best Epoch ---
        if row["Best_Epoch"] == "N/A":
            ep_match = re.search(r"(?:Loaded best checkpoint from epoch|best val AUC .* at epoch)\s+(\d+)", content, re.IGNORECASE)
            if ep_match:
                row["Best_Epoch"] = ep_match.group(1)

        # --- 2. Extract Test AUC (Overall) ---
        aucs = re.findall(r"overall-auc:\s*([\d\.]+)", content)
        if aucs:
            row["Test_AUC"] = round(float(aucs[-1]), 4)

        # --- 3. Extract EOD for ALL Attributes ---
        chunks = re.split(r"--- Evaluating ", content)
        for chunk in chunks:
            for attr in SENSITIVE_ATTRS:
                # Handle potential space in "Marital Status" logging just in case
                attr_spaced = attr.replace("Status", " Status") if attr == "MaritalStatus" else attr
                
                # Check if the chunk belongs to the current attribute
                if (chunk.startswith(attr) or 
                    chunk.startswith(attr_spaced) or 
                    (f"sensitive_name='{attr}'" in chunk and "overall-auc:" in chunk)):
                    
                    eods = re.findall(r"eod:\s*([\d\.]+)", chunk)
                    if eods: 
                        row[f"{attr}_EOD"] = round(float(eods[-1]), 4)

    data.append(row)

# --- 4. Build DataFrame and Compute Summary Metrics ---
df = pd.DataFrame(data)

# Identify which EOD columns were successfully extracted
eod_cols = [f"{attr}_EOD" for attr in SENSITIVE_ATTRS if f"{attr}_EOD" in df.columns]

if eod_cols:
    # Calculate Mean EOD
    df["Mean_EOD"] = round(df[eod_cols].mean(axis=1), 4)
    
    # Calculate Worst (Minimum) EOD
    df["Worst_EOD"] = df[eod_cols].min(axis=1)
    
    # Identify the Worst Attribute (idxmin returns the column name, so we strip "_EOD" from it)
    df["Worst_Attr"] = df[eod_cols].idxmin(axis=1).str.replace("_EOD", "")

# Organize columns beautifully
base_cols = ["Usage", "Best_Epoch", "Test_AUC"]
summary_cols = ["Mean_EOD", "Worst_EOD", "Worst_Attr"]
final_cols = base_cols + eod_cols + summary_cols

# Filter to only keep columns that exist in the dataframe (prevents KeyError if an attribute is missing)
final_cols = [c for c in final_cols if c in df.columns]
df = df[final_cols]

# --- 5. Save and Print ---
out_file = os.path.join(RESULTS_DIR, "Final_Metrics_Summary.csv")
df.to_csv(out_file, index=False)

print("\n" + "="*120)
print("FINAL FAIRNESS METRICS SUMMARY")
print("="*120)
print(df.to_markdown(index=False))
print("\n=> Successfully saved to:", out_file)