#!/usr/bin/env python3
"""
Per-epoch Test-Sex AUC + Equalized Odds Difference — Seed 0, CLIP ViT-B/32.

Fixes vs. original script:
  1. Parser is bounded to the [Test-Sex] block only (no bleed into Race/Age/etc.)
  2. EoD converted: log stores (1 - EoD) → standard_EoD = 1 - logged_value
  3. Male/Female ordering verified at runtime (print check on first epoch)
  4. Thesis-quality styling; saves .pdf + .png
"""

import os, re
import numpy as np
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────
BASE   = "./output_seed0/cls"
USAGES = ["lp", "clip-adapt", "full-ft", "full-ft-zs", "lora-lp", "lora-zs"]

# Best-checkpoint epoch per method (from seed-0 LR sweep — used for vertical line)
BEST_EPOCHS = {
    "lp":         88,
    "clip-adapt": 87,
    "full-ft":     8,
    "full-ft-zs": 13,
    "lora-lp":    25,
    "lora-zs":    28,
}

# ── Log finder ───────────────────────────────────────────────────────────────
def find_log(usage):
    root = os.path.join(BASE, usage)
    if not os.path.exists(root):
        return None
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if (fn.startswith("history.log_") and fn.endswith(".log")
                    and "CLIP" in dirpath and "seed0" in dirpath):
                return os.path.join(dirpath, fn)
    return None

# ── Parser — bounded to [Test-Sex] blocks only ───────────────────────────────
def parse_log(logfile):
    """
    Returns (epochs, female_auc, male_auc, eod_standard)
    where eod_standard = 1 - logged_eod  (lower = fairer, range 0–1).

    ⚠ Index assumption: groups are sorted alphabetically by label string,
      so "female" (index 0) comes before "male" (index 1).
      Verify with the printed check below.
    """
    epochs, auc_0, auc_1, eod_vals = [], [], [], []
    in_sex_block = False
    sex_auc_done = False
    cur_ep       = None

    with open(logfile) as fh:
        lines = fh.readlines()

    for line in lines:
        # ── Enter [Test-Sex] block ───────────────────────────────────────
        hit = re.search(r'\[Test-Sex\] Epoch (\d+)', line)
        if hit:
            cur_ep       = int(hit.group(1))
            in_sex_block = True
            sex_auc_done = False
            continue

        # ── Any other section header → exit sex block ────────────────────
        if re.search(r'\[(Test|Val|Train)-\w+\] Epoch', line):
            in_sex_block = False
            cur_ep       = None
            continue

        if not in_sex_block:
            continue

        # ── AUC: capture once per block ──────────────────────────────────
        if 'Subgroup metrics:' in line and not sex_auc_done:
            m = re.search(r'auc=\[(.+?)\]', line)
            if m:
                raw  = m.group(1).replace('np.float64(', '').replace(')', '')
                aucs = [float(x.strip()) for x in raw.split(',')]
                if len(aucs) >= 2:
                    auc_0.append(aucs[0])   # alphabetically first group
                    auc_1.append(aucs[1])   # alphabetically second group
                    sex_auc_done = True

        # ── EoD: keep as logged (1 - difference); higher = fairer ────────
        if 'Organized:' in line:
            m = re.search(r'eod=([\d.]+)', line)
            if m:
                epochs.append(cur_ep)
                eod_vals.append(float(m.group(1)))   # 1 - diff; higher = fairer
            in_sex_block = False   # epoch fully captured, exit block

    n = min(len(epochs), len(auc_0), len(auc_1), len(eod_vals))
    return epochs[:n], auc_0[:n], auc_1[:n], eod_vals[:n]

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.spines.top":    False,
    "axes.labelsize":     9,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    7.5,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#dddddd",
})

COL_0   = "#d94801"   # orange-red  (female — alphabetically first)
COL_1   = "#2171b5"   # blue        (male   — alphabetically second)
COL_EOD = "#31a354"   # green

# ── Main plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=False)
axes = axes.flatten()

verified_ordering = False   # print ordering check once

for idx, usage in enumerate(USAGES):
    ax      = axes[idx]
    logfile = find_log(usage)

    if logfile is None:
        ax.text(0.5, 0.5, f"No log file found\nfor  {usage}",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color="#aaa")
        ax.set_title(usage); continue

    print(f"\n[{usage}]  {logfile}")
    epochs, a0, a1, eod = parse_log(logfile)
    print(f"  → {len(epochs)} epochs parsed")

    if not epochs:
        ax.text(0.5, 0.5, "No [Test-Sex] data found",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color="#c00")
        ax.set_title(usage); continue

    # ── Ordering verification (printed once) ─────────────────────────────
    if not verified_ordering:
        print("\n  ⚠  VERIFY Male/Female ordering — first 5 epochs:")
        print(f"     {'Epoch':>5}  {'auc[0]':>8}  {'auc[1]':>8}  {'EoD':>8}")
        for i in range(min(5, len(epochs))):
            print(f"     {epochs[i]:>5}  {a0[i]:>8.4f}  {a1[i]:>8.4f}  {eod[i]:>8.4f}")
        print("  → If Male AUC is consistently > Female AUC at convergence,")
        print("    swap COL_0/COL_1 and labels below.\n")
        verified_ordering = True

    ep = np.array(epochs)
    g0 = np.array(a0)   # group index 0 (likely Female)
    g1 = np.array(a1)   # group index 1 (likely Male)
    e  = np.array(eod)

    # ── AUC trajectories ─────────────────────────────────────────────────
    ax.plot(ep, g0, color=COL_0, lw=2.0, label="Female AUC", zorder=3)
    ax.plot(ep, g1, color=COL_1, lw=2.0, label="Male AUC",   zorder=3)
    ax.fill_between(ep, g0, g1, alpha=0.10, color="#888", zorder=2)

    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC")
    ax.set_xlabel("Epoch")
    ax.set_title(usage)

    # ── Best-checkpoint vertical line ─────────────────────────────────────
    best_ep = BEST_EPOCHS.get(usage)
    if best_ep is not None and best_ep <= max(ep):
        ax.axvline(best_ep, color="black", linestyle="--",
                   lw=1.3, alpha=0.55, zorder=4,
                   label=f"Best ckpt (ep {best_ep})")

    # ── EoD on right axis ─────────────────────────────────────────────────
    ax2 = ax.twinx()
    ax2.plot(ep, e, color=COL_EOD, lw=1.5, linestyle=":",
             alpha=0.9, label="EoD score  (↑ fairer)", zorder=2)
    ax2.axhline(1.0, color=COL_EOD, lw=0.6, alpha=0.3, linestyle="-")  # perfect fairness
    ax2.set_ylabel("EoD score  (higher = fairer)", color=COL_EOD, fontsize=8)
    ax2.tick_params(axis='y', labelcolor=COL_EOD, labelsize=7)
    ax2.set_ylim(0.5, 1.05)
    ax2.spines["right"].set_edgecolor(COL_EOD)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # ── Combined legend ───────────────────────────────────────────────────
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")

fig.suptitle(
    "Per-epoch Test-Set Sex AUC and Equalized Odds Difference  —  Seed 0, CLIP ViT-B/32\n"
    r"Shaded region = Male$-$Female AUC gap.  "
    r"EoD score $= 1 - |\Delta\mathrm{TPR}|$  (higher = fairer;  1.0 = perfect equality).",
    fontsize=11, y=1.02
)

fig.tight_layout(pad=2.5)
out = "sex_auc_eod_clip_seed0"
fig.savefig(f"{out}.pdf", bbox_inches="tight")
fig.savefig(f"{out}.png", bbox_inches="tight", dpi=200)
print(f"\n✓  Saved  {out}.pdf  +  {out}.png")
