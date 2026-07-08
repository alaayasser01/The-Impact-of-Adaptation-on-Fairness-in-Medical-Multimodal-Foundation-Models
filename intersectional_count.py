import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# 1. Load the data from your CSV file
# Replace this string with the actual path to your CSV file
file_path = '/home/aym/Desktop/rhome/aym/FairMedFM/thesis_data/intersectional/intersectional_master_table.csv' 
df = pd.read_csv(file_path)

# 2. Pivot the data to create a matrix for the heatmap
# Rows = Intersection, Columns = Model, Values = AUC Gap
pivot_df = df.pivot(index='Intersection', columns='Model', values='AUC Gap')

# 3. Explicitly define the row and column order to match the provided image
row_order = [
    'Age × Marital',
    'Race × Language',
    'Gender × Marital',
    'Language × Marital',
    'Age × Language',
    'Ethnicity × Marital',
    'Race × Marital',
    'Age × Ethnicity',
    'Race × Ethnicity',
    'Age × Race',
    'Language × Ethnicity',
    'Gender × Language',
    'Gender × Race',
    'Gender × Ethnicity',
    'Gender × Age'
]

ordered_models = [
    'clip-zs', 
    'lp', 
    'clip-adapt', 
    'lora-lp', 
    'lora-text', 
    'full-ft', 
    'full-ft-text'
]

# Apply the specific ordering to the DataFrame
pivot_df = pivot_df.reindex(index=row_order, columns=ordered_models)

# 4. Create a mask to identify the "winners" (the minimum AUC gap per row)
# True means it is NOT the winner (we will mask these out in the green overlay)
is_not_winner = pivot_df.values != pivot_df.min(axis=1).values[:, None]

# 5. Set up the visualization
plt.figure(figsize=(14, 10))
sns.set_theme(style="white")

# Plot 1: The Base Heatmap (Red gradient)
# fmt=".2f" ensures exactly 2 decimal places
sns.heatmap(pivot_df, 
            cmap="Reds", 
            annot=True, 
            fmt=".2f", 
            linewidths=.5,
            cbar_kws={'label': 'AUC Gap (Max - Min)'})

# Plot 2: The Green "Winner" Overlay
# We create a custom solid green colormap and overlay it, masking the losers
green_cmap = mcolors.ListedColormap(['#2ca02c'])
sns.heatmap(pivot_df, 
            mask=is_not_winner, 
            cmap=green_cmap, 
            annot=True, 
            fmt=".2f", 
            linewidths=.5,
            cbar=False) # Hide the colorbar for the overlay

# 6. Formatting for presentation quality
plt.title('Intersectional Fairness: AUC Gap by Adaptation Strategy\n(Green = Best Performing / Lowest Disparity)', 
          fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Adaptation Strategy', fontsize=14, labelpad=10)
plt.ylabel('Compounded Demographic Intersection', fontsize=14, labelpad=10)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

plt.tight_layout()

# Save the high-resolution image for your presentation slides
plt.savefig('intersectional_heatmap_winners_ordered.png', dpi=300)

plt.show()