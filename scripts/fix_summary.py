import os, re, glob
import pandas as pd
import matplotlib.pyplot as plt

SWEEP_DIR = "./lr_sweep"
SEEDS = [1, 2]                     # adjust if needed
USAGES = ["lp", "clip-adapt", "full-ft", "full-ft-zs", "lora-lp", "lora-zs"]
LRS = ["1e-2", "5e-3"]            # only the best LRs we used for seeds 1/2
# If you ran all LRs for seeds 1/2, replace with: LRS = ["1e-5","5e-5","1e-4","5e-4","1e-3","5e-3","1e-2"]

for SEED in SEEDS:
    rows = []
    for usage in USAGES:
        for lr in LRS:
            base_dir = os.path.join(SWEEP_DIR, f"{usage}_lr{lr}_seed{SEED}")
            if not os.path.isdir(base_dir):
                rows.append([usage, lr, SEED, "MISSING_DIR", ""])
                continue

            log_candidates = glob.glob(os.path.join(base_dir, "**", "history*.log"),
                                       recursive=True)
            if not log_candidates:
                rows.append([usage, lr, SEED, "NO_LOG", ""])
                continue

            log_file = max(log_candidates, key=os.path.getmtime)

            with open(log_file, 'r') as f:
                content = f.read()

            # --- case‑insensitive match for "val AUC" or "Val AUC" ---
            auc_matches = re.findall(r"epoch (\d+): [Vv]al AUC = (0\.\d+)", content)
            if not auc_matches:
                # fallback: try "Loaded best checkpoint from epoch X with val AUC Y"
                fallback = re.findall(r"Loaded best checkpoint from epoch (\d+) with [Vv]al AUC (0\.\d+)", content)
                if fallback:
                    # take the last one (final best)
                    best_epoch, best_auc = fallback[-1]
                else:
                    rows.append([usage, lr, SEED, "NO_AUC", ""])
                    continue
            else:
                # take the epoch with the highest AUC
                best_epoch, best_auc = max(auc_matches, key=lambda x: float(x[1]))

            rows.append([usage, lr, SEED, best_auc, best_epoch])

    # Save CSV
    df = pd.DataFrame(rows, columns=["usage", "lr", "seed", "best_val_auc", "best_epoch"])
    csv_path = os.path.join(SWEEP_DIR, f"sweep_summary_seed{SEED}_corrected.csv")
    df.to_csv(csv_path, index=False)
    print(f"Seed {SEED} summary saved to {csv_path}")
    print(df.to_string())

# Optional: plot learning curves for each usage across seeds
# (can be added later)