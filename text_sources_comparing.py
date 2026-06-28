import json
import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import pandas as pd

sys.path.append('.')
from models.utils import get_model
from utils.text_sources import build_text_features

class Args:
    pass

# def run_similarity_analysis(model_name, backbone_name, output_img_name):
#     args = Args()
#     args.task = "cls"
#     args.model = model_name
#     args.backbone = backbone_name
#     args.proto_agg = "max"
#     args.dataset = "FairVLMed10k"
#     args.random_seed = 0
#     # Use GPU for massive speedup
#     args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     args.context_length = 77 if model_name == "CLIP" else 256

#     print(f"\n🚀 Initializing {model_name} ({backbone_name}) on {args.device}...")
    
#     with open("configs/datasets/FairVLMed10k.json") as f:
#         args.data_setting = json.load(f)
    
#     config_path = f"configs/models/{model_name}.json"
#     if not os.path.exists(config_path):
#         config_path = "configs/models/CLIP.json"
        
#     with open(config_path) as f:
#         args.model_setting = json.load(f)

#     model = get_model(args).to(args.device)
#     model.eval()

#     sources = ["template", "gpt4_avg", "clustered_summaries"]
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))

#     for ax, src in zip(axes, sources):
#         args.text_source = src
#         print(f"  -> Extracting features from text source: {src}")
        
#         try:
#             protos = build_text_features(args, model)
#         except Exception as e:
#             print(f"  ❌ Failed to extract source {src}: {e}")
#             continue

#         if isinstance(protos, list):
#             proto0, proto1 = protos[0].to(args.device), protos[1].to(args.device)
#             sim = F.cosine_similarity(proto0.mean(dim=0, keepdim=True), proto1.mean(dim=0, keepdim=True)).item()
#             mat = np.array([[1.0, sim], [sim, 1.0]])
#             sns.heatmap(mat, annot=True, fmt=".4f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
#                         xticklabels=["Normal", "Glaucoma"], yticklabels=["Normal", "Glaucoma"])
#             ax.set_title(f"{src}\n(Mean Cos Sim = {sim:.4f})")
            
#         elif protos.ndim == 2:
#             sim = F.cosine_similarity(protos[0], protos[1], dim=0).item()
#             mat = np.array([[1.0, sim], [sim, 1.0]])
#             sns.heatmap(mat, annot=True, fmt=".4f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
#                         xticklabels=["Normal", "Glaucoma"], yticklabels=["Normal", "Glaucoma"])
#             ax.set_title(f"{src}\nCos Sim = {sim:.4f}")
            
#         else:
#             C, K, D = protos.shape
#             # Ensure protos are on the correct device before math
#             protos = protos.to(args.device)
#             proto_norm = F.normalize(protos, dim=2)
            
#             # THE FIX: Standard Matrix Multiplication for (K, D) @ (D, K) -> (K, K)
#             sim_matrix = torch.matmul(proto_norm[0], proto_norm[1].T).cpu().numpy()
            
#             sns.heatmap(sim_matrix, annot=True, fmt=".4f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
#                         xticklabels=[f"G_{i}" for i in range(K)],
#                         yticklabels=[f"N_{i}" for i in range(K)])
#             ax.set_title(f"{src}\nMax Cross-Class Sim = {sim_matrix.max():.4f}")

#     plt.suptitle(f"Text Prototype Similarities — {model_name}", fontsize=16, fontweight='bold', y=1.05)
#     plt.tight_layout()
    
#     # Save to your poster_data directory
#     os.makedirs("./thesis_data", exist_ok=True)
#     out_path = os.path.join("./thesis_data", output_img_name)
#     plt.savefig(out_path, dpi=300, bbox_inches='tight')
    
#         # ---- Save CSV alongside the heatmap figure ----
#     csv_records = []
#     for src_idx, src in enumerate(sources):
#         # Re‑compute the matrix exactly as drawn (or capture it)
#         # (For simplicity we rebuild it here; you could store it earlier)
#         args.text_source = src
#         try:
#             protos = build_text_features(args, model)
#         except:
#             continue
#         if isinstance(protos, list):
#             proto0, proto1 = protos[0].to(args.device), protos[1].to(args.device)
#             sim = F.cosine_similarity(proto0.mean(dim=0, keepdim=True), proto1.mean(dim=0, keepdim=True)).item()
#             mat = [[1.0, sim], [sim, 1.0]]
#             row_labels = ["Normal", "Glaucoma"]
#             col_labels = ["Normal", "Glaucoma"]
#         elif protos.ndim == 2:
#             sim = F.cosine_similarity(protos[0], protos[1], dim=0).item()
#             mat = [[1.0, sim], [sim, 1.0]]
#             row_labels = ["Normal", "Glaucoma"]
#             col_labels = ["Normal", "Glaucoma"]
#         else:  # K prototypes per class
#             protos = protos.to(args.device)
#             proto_norm = F.normalize(protos, dim=2)
#             mat = torch.matmul(proto_norm[0], proto_norm[1].T).cpu().numpy()
#             K = mat.shape[0]
#             row_labels = [f"N_{i}" for i in range(K)]
#             col_labels = [f"G_{i}" for i in range(K)]
#         # Flatten to long format records
#         for i, rl in enumerate(row_labels):
#             for j, cl in enumerate(col_labels):
#                 csv_records.append({
#                     "text_source": src,
#                     "row_idx": i,
#                     "row_label": rl,
#                     "col_idx": j,
#                     "col_label": cl,
#                     "cosine_similarity": mat[i][j]
#                 })
#     # Write CSV
#     df_csv = pd.DataFrame(csv_records)
#     csv_path = os.path.join("./thesis_data", output_img_name.replace(".png", ".csv"))
#     df_csv.to_csv(csv_path, index=False)
#     print(f"📄 Similarity CSV saved to: {csv_path}")
    
#     plt.close()
#     print(f"✅ Success! Saved to: {out_path}")

def run_similarity_analysis(model_name, backbone_name, output_img_name):
    args = Args()
    args.task = "cls"
    args.model = model_name
    args.backbone = backbone_name
    args.proto_agg = "max"
    args.dataset = "FairVLMed10k"
    args.random_seed = 0
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.context_length = 77 if model_name == "CLIP" else 256

    print(f"\n🚀 Initializing {model_name} ({backbone_name}) on {args.device}...")

    with open("configs/datasets/FairVLMed10k.json") as f:
        args.data_setting = json.load(f)

    config_path = f"configs/models/{model_name}.json"
    if not os.path.exists(config_path):
        config_path = "configs/models/CLIP.json"

    with open(config_path) as f:
        args.model_setting = json.load(f)

    model = get_model(args).to(args.device)
    model.eval()

    sources = ["template", "gpt4_avg", "clustered_summaries"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    csv_records = []

    for ax, src in zip(axes, sources):

        args.text_source = src

        print(f"  -> Extracting features from {src}")

        try:
            protos = build_text_features(args, model)
        except Exception as e:
            print(f"Failed on {src}: {e}")
            continue

        # ======================================================
        # STANDARD 2-PROTOTYPE CASE
        # ======================================================
        # if protos.ndim == 2:

        #     sim = F.cosine_similarity(protos[0], protos[1], dim=0).item()

        #     mat = np.array([
        #         [1.0, sim],
        #         [sim, 1.0]
        #     ])

        #     sns.heatmap(
        #         mat,
        #         annot=True,
        #         fmt=".4f",
        #         cmap="coolwarm",
        #         vmin=-1,
        #         vmax=1,
        #         ax=ax,
        #         xticklabels=["Glaucoma", "Normal"],
        #         yticklabels=["Normal", "Glaucoma"]
        #     )

        #     ax.set_title(
        #         f"{src}\nCos Sim = {sim:.4f}"
        #     )

        #     row_labels = ["Normal", "Glaucoma"]
        #     col_labels = ["Glaucoma", "Normal"]

        if protos.ndim == 2:
            sim = F.cosine_similarity(protos[0], protos[1], dim=0).item()
            mat = np.array([[1.0, sim],
                            [sim, 1.0]])
            sns.heatmap(mat, annot=True, fmt=".4f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
                        xticklabels=["Normal", "Glaucoma"],
                        yticklabels=["Normal", "Glaucoma"])
            ax.set_title(f"{src}\nCos Sim = {sim:.4f}")
            row_labels = ["Normal", "Glaucoma"]
            col_labels = ["Normal", "Glaucoma"]

        # ======================================================
        # MULTI-PROTOTYPE CASE (K=5)
        # ======================================================
        else:

            C, K, D = protos.shape

            protos = protos.to(args.device)

            proto_norm = F.normalize(protos, dim=2)

            sim_matrix = (
                torch.matmul(proto_norm[0], proto_norm[1].T)
                .cpu()
                .numpy()
            )

            global_max = sim_matrix.max()

            row_max = sim_matrix.max(axis=1)

            mean_row_max = row_max.mean()

            sns.heatmap(
                sim_matrix,
                annot=True,
                fmt=".3f",
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                ax=ax,
                xticklabels=[f"G{i}" for i in range(K)],
                yticklabels=[f"N{i}" for i in range(K)]
            )

            ax.set_title(
                f"{src}\nGlobal max={global_max:.3f}\nMean row max={mean_row_max:.3f}"
            )

            row_labels = [f"N{i}" for i in range(K)]
            col_labels = [f"G{i}" for i in range(K)]

            mat = sim_matrix

        # ======================================================
        # SAVE CSV
        # ======================================================
        for i, rl in enumerate(row_labels):
            for j, cl in enumerate(col_labels):

                csv_records.append(
                    {
                        "text_source": src,
                        "row_idx": i,
                        "row_label": rl,
                        "col_idx": j,
                        "col_label": cl,
                        "cosine_similarity": float(mat[i][j])
                    }
                )

    plt.suptitle(
        f"Text Prototype Similarities — {model_name}",
        fontsize=16,
        fontweight="bold",
        y=1.03
    )

    plt.tight_layout()

    os.makedirs("./thesis_data", exist_ok=True)

    out_path = os.path.join("./thesis_data", output_img_name)

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    df_csv = pd.DataFrame(csv_records)

    csv_path = out_path.replace(".png", ".csv")

    df_csv.to_csv(csv_path, index=False)

    print(f"📄 CSV saved to {csv_path}")
    print(f"✅ Figure saved to {out_path}")

if __name__ == "__main__":
    # Run OpenAI CLIP
    run_similarity_analysis("CLIP", "ViT-B/32", "openai_clip_similarity_heatmaps.png")
    
    # Run BiomedCLIP
    run_similarity_analysis("BiomedCLIP", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", "biomedclip_similarity_heatmaps.png")