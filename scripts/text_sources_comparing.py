

# # import json, torch, sys
# # import matplotlib.pyplot as plt
# # from sklearn.decomposition import PCA
# # from sklearn.manifold import TSNE  # optional, if you prefer t-SNE

# # sys.path.append('.')
# # from models.utils import get_model
# # from utils.text_sources import build_text_features

# # # ---- Minimal args to load model and configs ----
# # class Args:
# #     pass

# # args = Args()
# # args.task = "cls"
# # args.model = "CLIP"
# # args.backbone = "ViT-B/32"
# # args.proto_agg = "max"
# # args.dataset = "FairVLMed10k"
# # args.random_seed = 0
# # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # args.context_length = 77

# # with open("configs/datasets/FairVLMed10k.json") as f:
# #     args.data_setting = json.load(f)
# # with open("configs/models/CLIP.json") as f:
# #     args.model_setting = json.load(f)

# # model = get_model(args)
# # model.to(args.device)

# # # ---- Collect prototypes for all sources ----
# # sources = ["template", "gpt4_avg", "clustered_summaries"]
# # all_protos = []      # list of (2D or 3D tensor)
# # source_names = []

# # for src in sources:
# #     args.text_source = src
# #     try:
# #         protos = build_text_features(args, model)   # may be tensor or list
# #     except Exception as e:
# #         print(f"Skipping {src}: {e}")
# #         continue

# #     # Convert to numpy, handle list (all_summaries) if present
# #     if isinstance(protos, list):
# #         # list of two tensors (N_class0, D) and (N_class1, D)
# #         # For PCA we need a 2D array: stack all vectors with labels
# #         proto0 = protos[0].cpu().numpy()
# #         proto1 = protos[1].cpu().numpy()
# #         all_vecs = np.vstack([proto0, proto1])
# #         labels = np.array([0]*proto0.shape[0] + [1]*proto1.shape[0])
# #     else:
# #         protos = protos.detach().cpu().numpy()   # shape (2, D) or (2, K, D)
# #         if protos.ndim == 2:
# #             all_vecs = protos
# #             labels = np.array([0, 1])
# #         else:   # (2, K, D)
# #             C, K, D = protos.shape
# #             all_vecs = protos.reshape(C*K, D)
# #             labels = np.repeat(np.arange(C), K)

# #     all_protos.append((all_vecs, labels))
# #     source_names.append(src)

# # # ---- Dimensionality reduction (PCA) ----
# # pca = PCA(n_components=2)
# # fig, axes = plt.subplots(1, len(source_names), figsize=(5*len(source_names), 4),
# #                          sharex=True, sharey=True)
# # if len(source_names) == 1:
# #     axes = [axes]

# # for ax, (vecs, labs), name in zip(axes, all_protos, source_names):
# #     pca_result = pca.fit_transform(vecs)
# #     ax.scatter(pca_result[labs==0, 0], pca_result[labs==0, 1],
# #                c='blue', label='Normal', alpha=0.7)
# #     ax.scatter(pca_result[labs==1, 0], pca_result[labs==1, 1],
# #                c='red', label='Glaucoma', alpha=0.7)
# #     ax.set_title(f"{name}\n(#protos: {vecs.shape[0]})")
# #     ax.legend()

# # plt.suptitle("2D PCA of Text Prototypes per Source")
# # plt.tight_layout()
# # plt.savefig("prototype_2d_comparison.png", dpi=150)
# # plt.show()
# import json, torch, sys, numpy as np
# sys.path.append('.')
# from models.utils import get_model
# from utils.text_sources import build_text_features
# import torch.nn.functional as F

# class Args:
#     pass

# args = Args()
# args.task = "cls"
# args.model = "CLIP"
# args.backbone = "ViT-B/32"
# args.proto_agg = "max"
# args.dataset = "FairVLMed10k"
# args.random_seed = 0
# args.device = torch.device("cpu")
# args.context_length = 77

# with open("configs/datasets/FairVLMed10k.json") as f:
#     args.data_setting = json.load(f)
# with open("configs/models/CLIP.json") as f:
#     args.model_setting = json.load(f)

# model = get_model(args).to(args.device)

# sources = ["gpt4_avg", "clustered_summaries"]
# for src in sources:
#     args.text_source = src
#     protos = build_text_features(args, model)   # tensor (2,D) or (2,K,D) or list

#     # Convert to a list of tensors: [class0_protos, class1_protos] where each is (N_i, D)
#     if isinstance(protos, list):
#         proto0 = protos[0]   # (N0, D)
#         proto1 = protos[1]   # (N1, D)
#     elif protos.ndim == 2:
#         proto0 = protos[0].unsqueeze(0)  # (1, D)
#         proto1 = protos[1].unsqueeze(0)
#     else:  # (2, K, D)
#         proto0 = protos[0]   # (K, D)
#         proto1 = protos[1]   # (K, D)

#     # Compute all pairwise cosine similarities
#     sims = []
#     for i in range(proto0.size(0)):
#         for j in range(proto1.size(0)):
#             sim = F.cosine_similarity(proto0[i].unsqueeze(0), proto1[j].unsqueeze(0)).item()
#             sims.append(sim)

#     if not sims:
#         print(f"{src:25s}  no prototypes")
#         continue

#     sims = np.array(sims)
#     print(f"{src:25s}  all‑vs‑all: max={sims.max():.4f}, median={np.median(sims):.4f}, min={sims.min():.4f}")

import json, torch, sys, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('.')
from models.utils import get_model
from utils.text_sources import build_text_features
import torch.nn.functional as F

# Minimal args – CPU only
class Args:
    pass

args = Args()
args.task = "cls"
args.model = "CLIP"
args.backbone = "ViT-B/32"
args.proto_agg = "max"
args.dataset = "FairVLMed10k"
args.random_seed = 0
args.device = torch.device("cpu")
args.context_length = 77

with open("configs/datasets/FairVLMed10k.json") as f:
    args.data_setting = json.load(f)
with open("configs/models/CLIP.json") as f:
    args.model_setting = json.load(f)

model = get_model(args).to(args.device)

sources = ["template", "clustered_summaries"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, src in zip(axes, sources):
    args.text_source = src
    protos = build_text_features(args, model)   # tensor (2, D) or (2, K, D)
    if protos.ndim == 2:
        # Single prototype per class → one value
        sim = F.cosine_similarity(protos[0], protos[1], dim=0).item()
        mat = np.array([[1, sim], [sim, 1]])  # symmetric 2x2
        sns.heatmap(mat, annot=True, fmt=".4f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
                    xticklabels=["Normal", "Glaucoma"], yticklabels=["Normal", "Glaucoma"])
    else:
        C, K, D = protos.shape
        proto_norm = F.normalize(protos, dim=2)
        # Cross‑class similarity matrix: normal (K) x glaucoma (K)
        sim_matrix = torch.einsum('ki,kj->ij', proto_norm[0], proto_norm[1]).numpy()
        sns.heatmap(sim_matrix, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
                    xticklabels=[f"G_{i}" for i in range(K)],
                    yticklabels=[f"N_{i}" for i in range(K)])
    ax.set_title(f"{src}\nmax cross‑class sim = {sim_matrix.max():.4f}" if src == "clustered_summaries"
                 else f"{src}\ncos sim = {sim:.4f}")

plt.tight_layout()
plt.savefig("prototype_similarity_heatmaps.png", dpi=150)
plt.show()