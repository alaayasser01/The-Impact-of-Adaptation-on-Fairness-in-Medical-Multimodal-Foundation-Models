import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import textwrap

# ==========================================
# 1. Configuration and Paths
# ==========================================
BASE_DIR = "/home/aym/Desktop/rhome/aym/FairMedFM"
IMAGE_DIR = f"{BASE_DIR}/data/Harvard FairVLMed (Harvard-FairVLMed）/Dataset/Test"
CSV_PATH = f"{BASE_DIR}/data/Harvard FairVLMed (Harvard-FairVLMed）/test.csv"
OUTPUT_DIR = f"{BASE_DIR}/fairvlmed_presentation_cards"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map requested paths to their display names
PRED_FILES = {
    "clip-adapt": f"{BASE_DIR}/lr_sweep/clip-adapt_lr1e-2_seed0/cls/clip-adapt/template/ViT-B_32/erm/ep100_lr0.01/FairVLMed10k/CLIP/All/seed0/clip-adapt_eval_Age/predictions.pkl",
    "full-ft": f"{BASE_DIR}/lr_sweep/full-ft_lr1e-2_seed0/cls/full-ft/template/ViT-B_32/erm/ep100_lr0.01/FairVLMed10k/CLIP/All/seed0/full-ft_eval_Age/predictions.pkl",
    "full-ft-text": f"{BASE_DIR}/lr_sweep/full-ft-zs_lr5e-3_seed0/cls/full-ft-zs/template/ViT-B_32/erm/ep100_lr0.005/FairVLMed10k/CLIP/All/seed0/full-ft-zs_eval_Age/predictions.pkl",
    "lora-lp": f"{BASE_DIR}/lr_sweep/lora-lp_lr1e-2_seed0/cls/lora-lp/template/ViT-B_32/erm/ep100_lr0.01/FairVLMed10k/CLIP/All/seed0/lora-lp_eval_Age/predictions.pkl",
    "lora-text": f"{BASE_DIR}/lr_sweep/lora-zs_lr1e-2_seed0/cls/lora-zs/template/ViT-B_32/erm/ep100_lr0.01/FairVLMed10k/CLIP/All/seed0/lora-zs_eval_Age/predictions.pkl",
    "lp": f"{BASE_DIR}/lr_sweep/lp_lr1e-2_seed0/cls/lp/template/ViT-B_32/erm/ep100_lr0.01/FairVLMed10k/CLIP/All/seed0/lp_eval_Age/predictions.pkl"
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

# Map predictions: 1 -> Glaucoma, 0 -> Normal
for method_name, pkl_path in PRED_FILES.items():
    if os.path.exists(pkl_path):
        preds = load_predictions(pkl_path)
        df[method_name] = ["Glaucoma" if p == 1 else "Normal" for p in preds]
    else:
        print(f"Warning: Could not find {pkl_path}")
        df[method_name] = "N/A"

# Remap Ground Truth to "Glaucoma" / "Normal" for clean display
df['glaucoma_display'] = df['glaucoma'].apply(lambda x: "Glaucoma" if str(x).lower() == "yes" else "Normal")

# ==========================================
# 3. Intelligent Selection Based on Results
# ==========================================
profiles = [
    # 1. Age x Marital gap -> Older (Group 3), Widowed
    (df['age_group'] == 3) & (df['maritalstatus'] == 'widowed') & (df['glaucoma_display'] == 'Glaucoma'),
    # 2. Age x Language gap -> Non-English
    (df['language'] != 'english') & (df['glaucoma_display'] == 'Normal'),
    # 3. Race x Language gap -> Black patient
    (df['race'] == 'black') & (df['glaucoma_display'] == 'Glaucoma'),
    # 4. Age x Ethnicity gap -> Hispanic patient
    (df['ethnicity'] == 'hispanic') & (df['glaucoma_display'] == 'Normal'),
    # 5. The Majority Baseline (High performing) -> White, English, Married
    (df['race'] == 'white') & (df['language'] == 'english') & (df['maritalstatus'] == 'married') & (df['glaucoma_display'] == 'Glaucoma'),
    # 6. Another vulnerability -> Single/Divorced
    (df['maritalstatus'].isin(['single', 'divorced'])) & (df['glaucoma_display'] == 'Normal')
]

selected_rows = []
used_indices = set()

for condition in profiles:
    candidates = df[condition & (~df.index.isin(used_indices))]
    if not candidates.empty:
        chosen = candidates.sample(1, random_state=42)
        selected_rows.append(chosen.iloc[0])
        used_indices.add(chosen.index[0])
    else:
        # Fallback if specific combo isn't in the test set
        remaining = df[~df.index.isin(used_indices)].sample(1, random_state=42)
        selected_rows.append(remaining.iloc[0])
        used_indices.add(remaining.index[0])

selected_df = pd.DataFrame(selected_rows)

# ==========================================
# 4. Image Loading Helper (Grayscale)
# ==========================================
def load_npz_image_gray(filepath):
    try:
        data = np.load(filepath)
        img_arr = data[data.files[0]]
        
        # Ensure correct channel ordering (H, W, C)
        if img_arr.ndim == 3 and img_arr.shape[0] in [1, 3]:
            img_arr = np.transpose(img_arr, (1, 2, 0))
            
        # Convert RGB to Grayscale using luminosity method
        if img_arr.ndim == 3 and img_arr.shape[-1] >= 3:
            img_arr = np.dot(img_arr[..., :3], [0.2989, 0.5870, 0.1140])
            
        return img_arr
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# ==========================================
# 5. Generate the Presentation Cards
# ==========================================
def draw_card(row, include_predictions, output_filename, card_index):
    fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [1, 1.2]})
    fig.patch.set_facecolor('#ffffff')
    
    # --- LEFT SIDE: IMAGE (GRAYSCALE) ---
    img_path = os.path.join(IMAGE_DIR, row['filename'])
    img_arr = load_npz_image_gray(img_path)
    
    if img_arr is not None:
        ax_img.imshow(img_arr, cmap='gray')
    else:
        ax_img.text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=20)
    ax_img.axis('off')
    
    # --- RIGHT SIDE: TEXT ---
    ax_txt.axis('off')
    
    y_pos = 0.90
    line_spacing = 0.08
    font_large = 26
    font_med = 20
    
    title = f"Profile {card_index}: Metadata & Predictions" if include_predictions else f"Profile {card_index}: Metadata & Clinical Notes"
    ax_txt.text(0.02, y_pos, title, fontsize=font_large, fontweight='bold', color='#1e293b')
    y_pos -= line_spacing * 1.5
    
    # Map the integer age_group to your exact thesis quartiles
    thesis_age_ranges = {0: "(18-51)", 1: "(52-64)", 2: "(65-73)", 3: "(74-91)"}
    age_range_str = thesis_age_ranges.get(int(row['age_group']), "")

    # Protected Attributes
    attributes = [
        ("Age:", f"{row['age']} yrs {age_range_str}"),
        ("Gender:", f"{str(row['gender']).title()}"),
        ("Race:", f"{str(row['race']).title()}"),
        ("Ethnicity:", f"{str(row['ethnicity']).title()}"),
        ("Language:", f"{str(row['language']).title()}"),
        ("Marital Status:", f"{str(row['maritalstatus']).title()}")
    ]
    
    for label, val in attributes:
        ax_txt.text(0.02, y_pos, label, fontsize=font_med, fontweight='bold')
        ax_txt.text(0.35, y_pos, val, fontsize=font_med)
        y_pos -= line_spacing
        
    y_pos -= line_spacing * 0.2
    
    gt_val = row['glaucoma_display']
    ax_txt.text(0.02, y_pos, "Label:", fontsize=font_med+2, fontweight='bold')
    ax_txt.text(0.35, y_pos, gt_val, fontsize=font_med+2, fontweight='bold', color='#1e293b')
    y_pos -= line_spacing * 1.5

    # --- CONDITIONAL LOWER SECTION ---
    if not include_predictions:
        note_truncated = textwrap.shorten(str(row['note']), width=140, placeholder="...")
        gpt_truncated = textwrap.shorten(str(row['gpt4_summary']), width=140, placeholder="...")
        
        ax_txt.text(0.02, y_pos, "Clinical Note:", fontsize=font_med, fontweight='bold')
        y_pos -= line_spacing * 0.6
        ax_txt.text(0.02, y_pos, textwrap.fill(note_truncated, width=65), fontsize=font_med-2, va='top', linespacing=1.5)
        
        y_pos -= line_spacing * 2.0
        
        ax_txt.text(0.02, y_pos, "GPT-4 Summary:", fontsize=font_med, fontweight='bold')
        y_pos -= line_spacing * 0.6
        ax_txt.text(0.02, y_pos, textwrap.fill(gpt_truncated, width=65), fontsize=font_med-2, va='top', linespacing=1.5)
        
    else:
        ax_txt.text(0.02, y_pos, "Model Predictions:", fontsize=font_large, fontweight='bold', color='#1e293b')
        y_pos -= line_spacing * 1.2
        
        for method in PRED_FILES.keys():
            if method in row:
                pred = row[method]
                color = "#16a34a" if pred == gt_val else "#dc2626"
                
                ax_txt.text(0.02, y_pos, f"{method}:", fontsize=font_med, fontweight='bold')
                ax_txt.text(0.35, y_pos, pred, fontsize=font_med, fontweight='bold', color=color)
                
                y_pos -= line_spacing

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename), dpi=200, bbox_inches='tight')
    plt.close(fig)

# Loop and Generate the 12 photos
for idx, row in enumerate(selected_df.iterrows(), start=1):
    _, row_data = row
    base_name = row_data['filename'].replace('.npz', '') 
    
    draw_card(row_data, include_predictions=False, output_filename=f"profile_{idx}_{base_name}_1_notes.png", card_index=idx)
    draw_card(row_data, include_predictions=True, output_filename=f"profile_{idx}_{base_name}_2_predictions.png", card_index=idx)

print(f"Success! Generated 12 presentation cards inside the '{OUTPUT_DIR}' folder.")