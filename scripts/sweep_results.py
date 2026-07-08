import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import glob

SWEEP_DIR = "./lr_sweep"
OUTPUT_CSV = os.path.join(SWEEP_DIR, "sweep_summary_corrected.csv")
PLOT_DIR = os.path.join(SWEEP_DIR, "learning_curves")
os.makedirs(PLOT_DIR, exist_ok=True)

usages = ["lp", "clip-adapt", "full-ft", "full-ft-zs", "lora-lp", "lora-zs"]
lrs = ["1e-5", "5e-5", "1e-4", "5e-4", "1e-3", "5e-3", "1e-2"]
seed = 0

rows = []
all_curves = {}

for usage in usages:
    for lr in lrs:
        base_dir = os.path.join(SWEEP_DIR, f"{usage}_lr{lr}_seed{seed}")
        if not os.path.isdir(base_dir):
            rows.append([usage, lr, seed, "MISSING_DIR", ""])
            continue

        # Find any log file that starts with "history" and ends with ".log"
        log_candidates = glob.glob(os.path.join(base_dir, "**", "history*.log"),
                                   recursive=True)
        if not log_candidates:
            rows.append([usage, lr, seed, "NO_LOG", ""])
            continue

        log_file = max(log_candidates, key=os.path.getmtime)

        with open(log_file, 'r') as f:
            content = f.read()

        auc_matches = re.findall(r"epoch (\d+): val AUC = (0\.\d+)", content)
        if not auc_matches:
            rows.append([usage, lr, seed, "NO_AUC", ""])
            continue

        best_epoch, best_auc = max(auc_matches, key=lambda x: float(x[1]))
        rows.append([usage, lr, seed, best_auc, best_epoch])

        curve = [(int(ep), float(auc)) for ep, auc in auc_matches]
        all_curves[(usage, lr)] = curve

# Save CSV
df = pd.DataFrame(rows, columns=["usage", "lr", "seed", "best_val_auc", "best_epoch"])
df.to_csv(OUTPUT_CSV, index=False)
print("Corrected CSV saved to", OUTPUT_CSV)
print(df.to_string())

# Plot best LR per usage
valid_df = df[df["best_val_auc"].apply(lambda x: isinstance(x, str) and x.replace('.','',1).isdigit())].copy()
valid_df["best_val_auc"] = valid_df["best_val_auc"].astype(float)

best_per_usage = {}
for usage in usages:
    sub = valid_df[valid_df["usage"] == usage]
    if sub.empty:
        continue
    best_row = sub.loc[sub["best_val_auc"].idxmax()]
    best_lr = best_row["lr"]
    best_per_usage[usage] = best_lr

    curve = all_curves.get((usage, best_lr), [])
    if not curve:
        continue
    epochs, aucs = zip(*sorted(curve))
    plt.figure()
    plt.plot(epochs, aucs, marker='.', label=f"LR={best_lr}")
    plt.title(f"{usage} – Best LR = {best_lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation AUC")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, f"{usage}_best_lr.png"))
    plt.close()

print("\nBest learning rate per usage:")
for usage, lr in best_per_usage.items():
    print(f"  {usage}: {lr}")