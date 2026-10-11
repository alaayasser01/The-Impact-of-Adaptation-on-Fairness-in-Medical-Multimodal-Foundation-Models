# import os
# import glob
# import re
# import numpy as np
# import pandas as pd

# # ---> UPDATE THIS TO YOUR ACTUAL BASE DIRECTORY <---
# BASE_DIR = "/home/alaa/papersubmission/The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-Foundation-Models/FairVLMed10k_clip_results"

# # Map the paper names to your actual folder/usage names!
# usage_map = {
#     "lp": "lp",
#     "clip-adapt": "clip-adapt",
#     "lora-lp": "lora-lp",
#     "lora-text": "lora-zs",        # Your file system uses lora-zs
#     "full-ft": "full-ft",
#     "full-ft-text": "full-ft-zs"   # Your file system uses full-ft-zs
# }

# results = []

# for display_name, folder_name in usage_map.items():
#     # Now it searches for "full-ft-zs_*" instead of "full-ft-text_*"
#     log_pattern = os.path.join(BASE_DIR, f"{folder_name}_*", "**", "Sex", "**", "history*.log")
#     log_files = glob.glob(log_pattern, recursive=True)
    
#     training_log_content = None
    
#     # Sort from newest to oldest
#     for log_file in sorted(log_files, key=os.path.getmtime, reverse=True):
#         with open(log_file, "r", encoding="utf-8") as f:
#             content = f.read()
        
#         # Strictly ensure this is a training run, not an evaluation run!
#         if "eval_only=False" in content and "epoch 1:" in content:
#             training_log_content = content
#             break
            
#     if not training_log_content:
#         print(f"Could not find a valid training log for {display_name} (searched for {folder_name})")
#         continue
        
#     # Cut off the final test set evaluation to only measure validation variance
#     train_only_content = re.split(r"Loaded best checkpoint", training_log_content)[0]
    
#     aucs = []
#     eods = []
    
#     chunks = re.split(r"----------------eva epoch \d+------------------", train_only_content)
    
#     for chunk in chunks[1:]: 
#         auc_match = re.search(r"overall-auc:\s*([\d\.]+)", chunk)
#         eod_match = re.search(r"eod:\s*([\d\.]+)", chunk)
        
#         if auc_match and eod_match:
#             aucs.append(float(auc_match.group(1)))
#             eods.append(float(eod_match.group(1)))

#     if not aucs:
#         continue

#     aucs = np.array(aucs)
#     eods = np.array(eods)
    
#     # Define "Convergence Plateau": Epochs where Val AUC is within 1% (0.01) of the max Val AUC
#     max_auc = aucs.max()
#     plateau_mask = aucs >= (max_auc - 0.01)
    
#     plateau_eods = eods[plateau_mask]
    
#     if len(plateau_eods) > 0:
#         std_eod = np.std(plateau_eods)
#         range_eod = np.max(plateau_eods) - np.min(plateau_eods)
        
#         results.append({
#             "Method": display_name,
#             "Max Val AUC": round(max_auc, 4),
#             "Plateau Epochs": len(plateau_eods),
#             "EoD Std Dev (\u03c3) \u2193": round(std_eod, 4),
#             "EoD Range \u2193": round(range_eod, 4)
#         })

# # Print Results
# df = pd.DataFrame(results)
# print("\n" + "="*80)
# print("TRAINING DYNAMICS: FAIRNESS STABILITY AT CONVERGENCE (SEX ATTRIBUTE)")
# print("="*80)
# print(df.to_markdown(index=False))
# print("="*80)

# plateau_epochs = np.where(plateau_mask)[0]

# print(display_name)
# print("plateau epochs:", plateau_epochs)
# print("AUC:", aucs[plateau_mask])
# print("EoD:", eods[plateau_mask])
# print("EoD std:", np.std(eods[plateau_mask]))

import os
import glob
import re
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/home/alaa/papersubmission/The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-Foundation-Models/FairVLMed10k_clip_results"

usage_map = {
    "lp": "lp",
    "clip-adapt": "clip-adapt",
    "lora-lp": "lora-lp",
    "lora-text": "lora-zs",
    "full-ft": "full-ft",
    "full-ft-text": "full-ft-zs"
}

# Absolute AUC tolerance defining the convergence region
AUC_TOLERANCE = 0.01


# ============================================================
# STORAGE
# ============================================================

summary_results = []
detailed_results = []


# ============================================================
# PROCESS EACH METHOD
# ============================================================

for display_name, folder_name in usage_map.items():

    print("\n" + "=" * 100)
    print(f"PROCESSING: {display_name}")
    print("=" * 100)

    log_pattern = os.path.join(
        BASE_DIR,
        f"{folder_name}_*",
        "**",
        "Sex",
        "**",
        "history*.log"
    )

    log_files = glob.glob(log_pattern, recursive=True)

    # --------------------------------------------------------
    # Find newest valid TRAINING log
    # --------------------------------------------------------

    training_log_content = None
    selected_log_file = None

    for log_file in sorted(log_files, key=os.path.getmtime, reverse=True):

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Make sure this is an actual training run
        if "eval_only=False" in content and "epoch 1:" in content:
            training_log_content = content
            selected_log_file = log_file
            break

    if training_log_content is None:
        print(
            f"Could not find a valid training log for {display_name} "
            f"(searched for {folder_name})"
        )
        continue

    print(f"Selected log: {selected_log_file}")

    # --------------------------------------------------------
    # Remove final test evaluation
    # --------------------------------------------------------

    train_only_content = re.split(
        r"Loaded best checkpoint",
        training_log_content,
        maxsplit=1
    )[0]

    # --------------------------------------------------------
    # Extract epoch, AUC and EoD
    # --------------------------------------------------------

    epochs = []
    aucs = []
    eods = []

    chunks = re.split(
        r"----------------eva epoch \d+------------------",
        train_only_content
    )

    for chunk in chunks[1:]:

        # Get the epoch number from the header if possible.
        # The header was removed by split, so recover it from
        # the preceding text using the "eva epoch" occurrences.
        #
        # We instead use sequential ordering below if necessary.

        auc_match = re.search(
            r"overall-auc:\s*([\d\.]+)",
            chunk
        )

        eod_match = re.search(
            r"eod:\s*([\d\.]+)",
            chunk
        )

        if auc_match and eod_match:
            aucs.append(float(auc_match.group(1)))
            eods.append(float(eod_match.group(1)))

    if not aucs:
        print("No validation AUC/EoD values found.")
        continue

    aucs = np.array(aucs)
    eods = np.array(eods)

    # Sequential epoch numbers corresponding to extracted evaluations.
    # This assumes the first extracted evaluation is epoch 0.
    epochs = np.arange(len(aucs))

    # --------------------------------------------------------
    # Find best validation AUC
    # --------------------------------------------------------

    max_auc = aucs.max()
    best_indices = np.where(aucs == max_auc)[0]
    best_epoch = epochs[best_indices[0]]

    # --------------------------------------------------------
    # Define convergence region
    # --------------------------------------------------------

    plateau_threshold = max_auc - AUC_TOLERANCE

    plateau_mask = aucs >= plateau_threshold

    plateau_epochs = epochs[plateau_mask]
    plateau_aucs = aucs[plateau_mask]
    plateau_eods = eods[plateau_mask]

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    n_plateau = len(plateau_eods)
    total_epochs = len(aucs)

    auc_mean = np.mean(plateau_aucs)
    auc_std = np.std(plateau_aucs)
    auc_range = np.max(plateau_aucs) - np.min(plateau_aucs)

    eod_mean = np.mean(plateau_eods)
    eod_std = np.std(plateau_eods)
    eod_range = np.max(plateau_eods) - np.min(plateau_eods)

    # Coefficient of variation (useful as an additional descriptive measure)
    eod_cv = eod_std / eod_mean if eod_mean != 0 else np.nan

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_results.append({
        "Method": display_name,
        "Best Epoch": best_epoch,
        "Max Val AUC": max_auc,
        "Plateau Epochs": n_plateau,
        "Plateau %": 100 * n_plateau / total_epochs,
        "AUC Mean": auc_mean,
        "AUC Std": auc_std,
        "AUC Range": auc_range,
        "EoD Mean": eod_mean,
        "EoD Std": eod_std,
        "EoD Range": eod_range,
        "EoD CV": eod_cv
    })

    # --------------------------------------------------------
    # Detailed epoch-level information
    # --------------------------------------------------------

    for epoch, auc, eod in zip(
        plateau_epochs,
        plateau_aucs,
        plateau_eods
    ):
        detailed_results.append({
            "Method": display_name,
            "Epoch": epoch,
            "Val AUC": auc,
            "Sex EoD": eod
        })

    # --------------------------------------------------------
    # PRINT EVERYTHING FOR THIS METHOD
    # --------------------------------------------------------

    print("\n--- Overall training ---")
    print(f"Total evaluated epochs : {total_epochs}")
    print(f"Maximum Val AUC        : {max_auc:.6f}")
    print(f"Best epoch             : {best_epoch}")

    print("\n--- Convergence definition ---")
    print(
        f"Plateau = Val AUC >= "
        f"{max_auc:.6f} - {AUC_TOLERANCE:.2f}"
    )
    print(f"Plateau threshold      : {plateau_threshold:.6f}")

    print("\n--- Plateau statistics ---")
    print(f"Plateau epochs         : {n_plateau}")
    print(f"Plateau fraction       : {n_plateau / total_epochs:.2%}")
    print(
        f"Epoch range            : "
        f"{plateau_epochs[0]} - {plateau_epochs[-1]}"
    )

    print("\nValidation AUC within plateau:")
    print(f"  Mean                 : {auc_mean:.6f}")
    print(f"  Std                  : {auc_std:.6f}")
    print(f"  Range                : {auc_range:.6f}")

    print("\nSex EoD within plateau:")
    print(f"  Mean                 : {eod_mean:.6f}")
    print(f"  Std                  : {eod_std:.6f}")
    print(f"  Range                : {eod_range:.6f}")
    print(f"  Coefficient variation: {eod_cv:.4%}")

    print("\n--- Plateau epoch-by-epoch values ---")
    detailed_df = pd.DataFrame({
        "Epoch": plateau_epochs,
        "Val_AUC": plateau_aucs,
        "Sex_EoD": plateau_eods
    })

    print(detailed_df.to_string(index=False))


# ============================================================
# FINAL SUMMARY TABLE
# ============================================================

summary_df = pd.DataFrame(summary_results)

print("\n\n")
print("=" * 120)
print("FINAL SUMMARY: FAIRNESS VARIABILITY NEAR CONVERGENCE")
print("=" * 120)

if not summary_df.empty:

    display_df = summary_df.copy()

    numeric_cols = [
        "Max Val AUC",
        "AUC Mean",
        "AUC Std",
        "AUC Range",
        "EoD Mean",
        "EoD Std",
        "EoD Range"
    ]

    for col in numeric_cols:
        display_df[col] = display_df[col].map(
            lambda x: f"{x:.4f}"
        )

    display_df["Plateau %"] = display_df["Plateau %"].map(
        lambda x: f"{x:.1f}%"
    )

    display_df["EoD CV"] = display_df["EoD CV"].map(
        lambda x: f"{x:.2%}"
    )

    print(display_df.to_string(index=False))

else:
    print("No results found.")


# ============================================================
# FINAL DETAILED TABLE
# ============================================================

detailed_df = pd.DataFrame(detailed_results)

print("\n\n")
print("=" * 80)
print("ALL PLATEAU EPOCHS")
print("=" * 80)

if not detailed_df.empty:
    print(
        detailed_df.to_string(
            index=False,
            formatters={
                "Val AUC": "{:.6f}".format,
                "Sex EoD": "{:.6f}".format
            }
        )
    )
else:
    print("No detailed results found.")