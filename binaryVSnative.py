import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CLIP_NATIVE = "/home/alaa/papersubmission/The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-Foundation-Models/FairVLMed10k_clip_results/Final_Metrics_Summary.csv"
CLIP_BINARY = "/home/alaa/papersubmission/The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-Foundation-Models/FairVLMed10k_clip_results_BINARY/Final_Metrics_Summary.csv"

BIOMED_NATIVE = "/home/alaa/papersubmission/The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-Foundation-Models/biomedclip_results/Final_Metrics_Summary.csv"
BIOMED_BINARY = "/home/alaa/papersubmission/The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-Foundation-Models/Fairmedvlm_biomedclip_results_BINARY/Final_Metrics_Summary.csv"

OUTPUT_DIR = "/home/alaa/papersubmission/The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-Foundation-Models"

# Order of methods from lowest to highest capacity
ORDER = ["clip-zs", "lp", "clip-adapt", "lora-lp", "lora-zs", "full-ft", "full-ft-zs"]

NAME_MAP = {
    "clip-zs": "Zero-Shot",
    "lp": "Linear Probe",
    "clip-adapt": "CLIP-Adapt",
    "lora-lp": "LoRA-LP",
    "lora-zs": "LoRA-Text",
    "full-ft": "Full-FT",
    "full-ft-zs": "Full-FT-Text"
}

ATTRS = ["Sex", "Age", "Race", "Language", "Ethnicity", "MaritalStatus"]
ATTR_LABELS = ["Sex", "Age", "Race", "Lang", "Ethnic", "Marital"]

# ==============================================================================
# PLOTTING FUNCTION
# ==============================================================================
def plot_compact_protocol_comparison(native_csv, binary_csv, model_title, output_filename):
    if not os.path.exists(native_csv) or not os.path.exists(binary_csv):
        print(f"Skipping {output_filename} - CSVs missing.")
        return

    # Load and merge
    df_nat = pd.read_csv(native_csv)
    df_bin = pd.read_csv(binary_csv)
    df = pd.merge(df_nat, df_bin, on="Usage", suffixes=('_nat', '_bin'))

    # Sort
    df['Usage_Cat'] = pd.Categorical(df['Usage'], categories=ORDER, ordered=True)
    df = df.sort_values('Usage_Cat').reset_index(drop=True)
    df = df.dropna(subset=['Test_AUC_nat'])

    # Setup the 1x7 Grid
    n_methods = len(df)
    fig, axes = plt.subplots(1, n_methods, figsize=(20, 5), sharey=True)

    # 1. Add the clear, to-the-point model subtitle
    fig.suptitle(model_title, fontsize=20, fontweight='bold', y=1.12)

    # Colors
    color_native = '#d62728' # Red for Not Binarized
    color_binary = '#1f77b4' # Blue for Binarized

    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]
        
        # Plot the Dumbbells
        for j, attr in enumerate(ATTRS):
            col_nat = f"{attr}_EOD_nat"
            col_bin = f"{attr}_EOD_bin"
            
            if col_nat in row and col_bin in row:
                v_nat = row[col_nat]
                v_bin = row[col_bin]
                
                # Connect the dots
                ax.plot([j, j], [v_nat, v_bin], color='gray', linewidth=2.5, zorder=1, alpha=0.6)
                
                # Plot Native & Binary dots
                ax.scatter(j, v_nat, color=color_native, s=100, zorder=2, edgecolor='white',
                           label='Not Binarized' if i == 0 and j == 0 else "")
                ax.scatter(j, v_bin, color=color_binary, s=100, zorder=2, edgecolor='white',
                           label='Binarized' if i == 0 and j == 0 else "")

        # Formatting each panel
        auc_val = row['Test_AUC_nat']
        title_text = f"{NAME_MAP[row['Usage']]}\n(AUC: {auc_val:.2f})"
        
        ax.set_title(title_text, fontsize=15, fontweight='bold', pad=12)
        ax.set_xticks(range(len(ATTRS)))
        ax.set_xticklabels(ATTR_LABELS, rotation=45, ha='right', fontsize=13)
        ax.set_ylim(0.4, 1.05) 
        
        if i == 0:
            ax.set_ylabel("EoD", fontsize=16, fontweight='bold')
            ax.tick_params(axis='y', labelsize=13)
        
        ax.grid(axis='y', linestyle=':', alpha=0.7)
        sns.despine(ax=ax)

    # Global Legend pulled tight under the subtitle
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize=15, frameon=False, 
               handletextpad=0.4, columnspacing=2.0)
    
    # Adjust top margin so there is a clean space below the subtitle and legend before the plots start
    plt.subplots_adjust(wspace=0.1, top=0.78)
    
    out_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved compact panel plot to: {out_path}")
    plt.close()

# ==============================================================================
# EXECUTE
# ==============================================================================
plot_compact_protocol_comparison(CLIP_NATIVE, CLIP_BINARY, "CLIP", "fig_compact_protocol_clip.png")
plot_compact_protocol_comparison(BIOMED_NATIVE, BIOMED_BINARY, "BioMedCLIP", "fig_compact_protocol_biomedclip.png")