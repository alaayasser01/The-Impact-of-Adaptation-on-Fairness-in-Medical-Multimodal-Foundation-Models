
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Define CSV file paths (using curated results)
BASE_RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
clip_csv_path = os.path.join(BASE_RESULTS, "openai_clip_similarity_heatmaps.csv")
biomed_csv_path = os.path.join(BASE_RESULTS, "biomedclip_similarity_heatmaps.csv")

# 2. Reusable function to plot the row of heatmaps from a CSV file
def plot_model_heatmaps(csv_path, model_name):
    # Read the data directly from the CSV file
    df = pd.read_csv(csv_path)
    
    # Set a custom vmin based on the lowest value in this specific dataset
    # This bounds the linear scale, giving you that high-contrast look 
    # without needing a logarithmic scale.
    min_val = df['cosine_similarity'].min()
    custom_vmin = round(min_val - 0.02, 2)
    custom_vmax = 1.0 
    
    sources = ['template', 'gpt4_avg', 'clustered_summaries']
    titles = ['Template', 'GPT-4 Averaged', 'Clustered Summaries']
    
    # Create the 1x3 subplot figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Text Prototype Similarities — {model_name}', fontsize=16, fontweight='bold')
    
    for ax, source, title in zip(axes, sources, titles):
        # Filter and pivot the data for this specific heatmap
        subset = df[df['text_source'] == source]
        pivot_df = subset.pivot(index='row_label', columns='col_label', values='cosine_similarity')
        
        # Plotting the heatmap with the bounded linear scale
        sns.heatmap(pivot_df, 
                    ax=ax,
                    cmap="coolwarm", # or use "Reds" if you prefer your original color scheme
                    vmin=custom_vmin, 
                    vmax=custom_vmax, 
                    annot=True, 
                    fmt=".4f", 
                    linewidths=.5,
                    cbar_kws={'label': 'Cosine Sim'} if ax == axes[-1] else None)
        
        # Calculate max non-diagonal values for the title to match your style
        if source == 'clustered_summaries':
            global_max = pivot_df.max().max()
            mean_row_max = pivot_df.max(axis=1).mean()
            ax.set_title(f"{title}\nGlobal max={global_max:.3f}\nMean row max={mean_row_max:.3f}")
        else:
            # For 2x2 matrices, the cross-class similarity is the off-diagonal
            cross_sim = pivot_df.iloc[0, 1] 
            ax.set_title(f"{title}\nCos Sim = {cross_sim:.4f}")
            
        ax.set_ylabel("")
        ax.set_xlabel("")

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures", "paper")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"text_sources_heatmaps_{model_name}.png")
    plt.savefig(out_file, dpi=300)
    print(f"Saved: {out_file}")
    plt.close()

# 3. Generate both figures reading directly from the files
plot_model_heatmaps(clip_csv_path, "CLIP")
plot_model_heatmaps(biomed_csv_path, "BiomedCLIP")