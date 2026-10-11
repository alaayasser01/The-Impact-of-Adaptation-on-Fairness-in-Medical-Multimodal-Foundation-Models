import os
import glob
import re
import pandas as pd

# ---> Path to your FairVLMed10k + BiomedCLIP results <---
RESULTS_DIR = "/home/alaa/papersubmission/The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-Foundation-Models/biomedclip_results"

usages = ["clip-zs", "lp", "clip-adapt", "full-ft", "lora-lp", "lora-zs", "full-ft-zs"]
data = []

for usage in usages:
    row = {
        "Usage": usage, 
        "Best_Epoch": "N/A", 
        "Test_AUC": "N/A",
        "Mean_EOD": "N/A",
        "Worst_EOD": "N/A",
        "Worst_Attr": "N/A"
    }
    
    eod_dict = {}
    
    # Grab all log files for this specific usage
    log_files = glob.glob(os.path.join(RESULTS_DIR, f"{usage}_*", "**", "*.log"), recursive=True)
    
    # Sort by modification time so we process the most recent log last
    log_files = sorted(log_files, key=os.path.getmtime)
    
    for log_path in log_files:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # --- 1. Extract Best Epoch ---
        ep_match = re.search(r"(?:Loaded best checkpoint from epoch|best val AUC .* at epoch)\s+(\d+)", content, re.IGNORECASE)
        if ep_match:
            row["Best_Epoch"] = ep_match.group(1)

        # --- 2. Extract AUC and EOD per attribute ---
        # The logs use "--- Evaluating <Attribute> ---" as a separator
        chunks = content.split("--- Evaluating ")
        
        for chunk in chunks[1:]:  # Skip the first chunk (training logs)
            # Find the attribute name (e.g., Sex, Age, Race)
            attr_match = re.match(r"^([A-Za-z]+)\s*---", chunk)
            if not attr_match:
                continue
            
            attr = attr_match.group(1)
            
            # Grab the overall AUC (same across all attributes, but we update it)
            aucs = re.findall(r"overall-auc:\s*([\d\.]+)", chunk)
            if aucs:
                row["Test_AUC"] = round(float(aucs[-1]), 4)
                
            # Grab the EOD for this specific attribute
            eods = re.findall(r"eod:\s*([\d\.]+)", chunk)
            if eods:
                eod_dict[attr] = float(eods[-1])
    
    # --- 3. Compute Mean and Worst EOD ---
    if eod_dict:
        # Mean EOD across all evaluated attributes
        row["Mean_EOD"] = round(sum(eod_dict.values()) / len(eod_dict), 4)
        
        # Worst EOD (Higher EOD = larger disparity = worse fairness)
        worst_attr = min(eod_dict, key=eod_dict.get)
        row["Worst_EOD"] = round(eod_dict[worst_attr], 4)
        row["Worst_Attr"] = worst_attr

    data.append(row)

# --- 4. Build DataFrame and Print ---
df = pd.DataFrame(data)

out_file = os.path.join(RESULTS_DIR, "FairVLMed_BiomedCLIP_Summary.csv")
df.to_csv(out_file, index=False)

print("\n" + "="*85)
print("FINAL FAIRVLMED10K METRICS SUMMARY")
print("="*85)
print(df.to_markdown(index=False))
print("\n=> Successfully saved to:", out_file)