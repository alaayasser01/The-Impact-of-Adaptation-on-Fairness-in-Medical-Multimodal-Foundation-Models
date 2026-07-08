import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# 1. Configuration and Paths
# ==========================================
BASE_DIR = "/home/aym/Desktop/rhome/aym/FairMedFM"
IMAGE_DIR = f"{BASE_DIR}/data/DDI/images"
CSV_PATH = f"{BASE_DIR}/data/DDI/test.csv"
OUTPUT_DIR = f"{BASE_DIR}/ddi_presentation_cards"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Using the exact names you requested, mapped to your previous paths
PRED_FILES = {
    "clip-zs": f"{BASE_DIR}/ddi_results/clip-zs/cls/clip-zs/template/ViT-B_32/erm/ep100_lr0.01/DDI/CLIP/fitzpatrick/seed0/clip_zs_final_fitzpatrick/predictions.pkl",
    "full-ft": f"{BASE_DIR}/ddi_results/full-ft/cls/full-ft/template/ViT-B_32/erm/ep100_lr0.01/DDI/CLIP/fitzpatrick/seed0/full-ft_final_fitzpatrick/predictions.pkl",
    "full-ft-text": f"{BASE_DIR}/ddi_results/full-ft-zs/cls/full-ft-zs/template/ViT-B_32/erm/ep100_lr0.005/DDI/CLIP/fitzpatrick/seed0/full-ft-zs_final_fitzpatrick/predictions.pkl",
    "lora-text": f"{BASE_DIR}/ddi_results/lora-zs/cls/lora-zs/template/ViT-B_32/erm/ep100_lr0.01/DDI/CLIP/fitzpatrick/seed0/lora-zs_final_fitzpatrick/predictions.pkl"
}

# ==========================================
# 2. Load Data and Merge Predictions
# ==========================================
df = pd.read_csv(CSV_PATH)

def load_predictions(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    logits = data["logits"]
    preds = np.argmax(logits, axis=1)
    return preds

# Map predictions
for method_name, pkl_path in PRED_FILES.items():
    if os.path.exists(pkl_path):
        preds = load_predictions(pkl_path)
        # 1 = malignant, 0 = benign
        df[method_name] = ["malignant" if p == 1 else "benign" for p in preds]
    else:
        print(f"Warning: Could not find {pkl_path}")
        df[method_name] = "N/A"

# ==========================================
# 3. Select 6 Unique Examples
# ==========================================
# 3 skin colors x 2 labels = 6 examples. All must have different diseases.
target_skin_types = ['I-II', 'III-IV', 'V-VI']
target_labels = ['benign', 'malignant']

selected_rows = []
used_diseases = set()

for skin in target_skin_types:
    for label in target_labels:
        # Filter for current combination and ensure disease isn't already used
        candidates = df[(df['fitzpatrick'] == skin) & 
                        (df['label_str'] == label) & 
                        (~df['disease'].isin(used_diseases))]
        
        if not candidates.empty:
            # Pick a random candidate from the valid options
            chosen = candidates.sample(1, random_state=42).iloc[0]
            selected_rows.append(chosen)
            used_diseases.add(chosen['disease'])
        else:
            print(f"Warning: Could not find unique disease for {skin} and {label}")

selected_df = pd.DataFrame(selected_rows)

# ==========================================
# 4. Generate the Presentation Cards
# ==========================================
def draw_card(row, include_predictions, output_filename):
    # Setup a 16:9 aspect ratio figure (great for presentations)
    fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [1, 1.2]})
    fig.patch.set_facecolor('#ffffff') # White background
    
    # --- LEFT SIDE: IMAGE ---
    img_path = os.path.join(IMAGE_DIR, row['DDI_file'])
    try:
        img = Image.open(img_path)
        ax_img.imshow(img)
    except FileNotFoundError:
        ax_img.text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=20)
    ax_img.axis('off')
    
    # --- RIGHT SIDE: TEXT ---
    ax_txt.axis('off')
    
    # Base styling
    y_pos = 0.85
    line_spacing = 0.08
    font_large = 28
    font_med = 24
    
    # 1. Ground Truth Info
    ax_txt.text(0.05, y_pos, "Patient Metadata:", fontsize=font_large, fontweight='bold', color='#1e293b')
    y_pos -= line_spacing * 1.5
    
    ax_txt.text(0.05, y_pos, f"Skin Color:", fontsize=font_med, fontweight='bold')
    ax_txt.text(0.35, y_pos, f"{row['fitzpatrick']} (Fitzpatrick)", fontsize=font_med)
    y_pos -= line_spacing
    
    ax_txt.text(0.05, y_pos, f"Disease:", fontsize=font_med, fontweight='bold')
    ax_txt.text(0.35, y_pos, f"{row['disease'].replace('-', ' ').title()}", fontsize=font_med)
    y_pos -= line_spacing
    
    gt_color = "#dc2626" if row['label_str'] == "malignant" else "#16a34a"
    ax_txt.text(0.05, y_pos, f"True Label:", fontsize=font_med, fontweight='bold')
    ax_txt.text(0.35, y_pos, f"{row['label_str'].upper()}", fontsize=font_med+4, fontweight='bold', color=gt_color)
    y_pos -= line_spacing * 2
    
    # 2. Predictions (Only if requested)
    if include_predictions:
        ax_txt.text(0.05, y_pos, "Model Predictions:", fontsize=font_large, fontweight='bold', color='#1e293b')
        y_pos -= line_spacing * 1.5
        
        for method in PRED_FILES.keys():
            if method in row:
                pred = row[method]
                
                # Determine match and color
                is_correct = (pred == row['label_str'])
                color = "#16a34a" if is_correct else "#dc2626" # Green if match, Red if incorrect
                icon = "✓ Match" if is_correct else "✗ Error"
                
                # Draw method name
                ax_txt.text(0.05, y_pos, f"{method}:", fontsize=font_med, fontweight='bold')
                
                # Draw Prediction and Match status in correct color
                result_text = f"{pred.upper()} ({icon})"
                ax_txt.text(0.40, y_pos, result_text, fontsize=font_med, fontweight='bold', color=color)
                
                y_pos -= line_spacing

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename), dpi=200, bbox_inches='tight')
    plt.close(fig)

# Generate the 12 photos
for idx, row in selected_df.iterrows():
    base_name = row['DDI_file'].split('.')[0] # e.g., '000119'
    
    # Card 1: Ground Truth Only
    draw_card(row, include_predictions=False, output_filename=f"{base_name}_1_ground_truth.png")
    
    # Card 2: Ground Truth + Predictions
    draw_card(row, include_predictions=True, output_filename=f"{base_name}_2_predictions.png")

print(f"Success! Generated 12 presentation cards inside the '{OUTPUT_DIR}' folder.")