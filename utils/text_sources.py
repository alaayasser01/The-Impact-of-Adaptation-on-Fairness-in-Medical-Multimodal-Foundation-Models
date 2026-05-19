import torch
import pandas as pd
from utils.tokenizer import tokenize_text
import torch.nn.functional as F


def build_text_features(args, model):
    """
    Dispatcher: returns text features of shape (num_classes, feature_dim)
    based on args.text_source.
    """
    if args.text_source == "template":
        return build_template_features(args, model)
    elif args.text_source == "gpt4_avg":
        return build_gpt4_avg_features(args, model)
    elif args.text_source == "all_summaries":
        return build_all_summaries_features(args, model)
    elif args.text_source == "clustered_summaries":
        return build_clustered_features(args, model, K=args.model_setting.get("num_prototypes_per_class", 5))
    else:
        raise ValueError(f"Unknown text source: {args.text_source}")

def build_template_features(args, model):
    """
    Original zero‑shot: prompted templates from dataset config.
    """
    template = args.data_setting["text_template"]
    class_names = args.data_setting["class_names"]
    text_tokens = tokenize_text(args, class_names, template=template)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens.to(args.device)).detach()
    return text_features


def truncate_text_tail(text, max_chars=180):
    """Keep the last max_chars characters (roughly)."""
    if len(text) > max_chars:
        text = text[-max_chars:]
        # Optionally, cut at first space to avoid broken words
        # But not necessary; tokenizer can handle.
    return text


def build_gpt4_avg_features(args, model):
    train_meta = pd.read_csv(args.data_setting["train_meta_path"])
    train_meta = train_meta.dropna(subset=["gpt4_summary"])

    class_names = args.data_setting["class_names"]
    label_map = {"no": 0, "yes": 1}

    texts_by_class = {c: [] for c in class_names}
    for _, row in train_meta.iterrows():
        label_str = row["glaucoma"]
        if label_str in label_map:
            cls_idx = label_map[label_str]
            # Keep only the last part of the summary (where the conclusion usually is)
            summary = row["gpt4_summary"].strip()
            # Truncate from the end: keep last ~180 characters
            max_chars = 180
            if len(summary) > max_chars:
                summary = summary[-max_chars:]
            texts_by_class[class_names[cls_idx]].append(summary)

    all_features = []
    for class_name in class_names:
        texts = texts_by_class[class_name]
        print(f"\n--- {class_name} (first 3 examples) ---")
        for t in texts[:3]:
            print(t[:200])          # first 200 chars of truncated summary
        print(f"Total texts: {len(texts)}")
        if len(texts) == 0:
            avg_feat = torch.zeros(model.feat_dim)
        else:
            tokens = tokenize_text(args, texts, template=None)   # raw, will be head-truncated to 77 tokens
            # The tail of the text is now at the beginning because we kept the last 180 chars.
            # So head truncation is fine.
            with torch.no_grad():
                feats = model.encode_text(tokens.to(args.device))
            avg_feat = feats.mean(dim=0)
        all_features.append(avg_feat)

    text_features = torch.stack(all_features, dim=0)
    # debug prints
    print(f"[DEBUG] Prototype shape: {text_features.shape}")
    print(f"[DEBUG] Normal prototype L2 norm: {text_features[0].norm().item():.4f}")
    print(f"[DEBUG] Glaucoma prototype L2 norm: {text_features[1].norm().item():.4f}")
    cos_sim = F.cosine_similarity(text_features[0], text_features[1], dim=0).item()
    print(f"[DEBUG] Cosine similarity between prototypes: {cos_sim:.4f}")
    return text_features

def build_all_summaries_features(args, model):
    """
    Returns a list of two tensors: [normal_protos, glaucoma_protos]
    Each tensor has shape (N_class, 512), where N_class is the number of training summaries for that class.
    """
    train_meta = pd.read_csv(args.data_setting["train_meta_path"])
    train_meta = train_meta.dropna(subset=["gpt4_summary"])
    class_names = args.data_setting["class_names"]
    label_map = {"no": 0, "yes": 1}

    features_list = []   # will be [normal_protos, glaucoma_protos]
    for class_name in class_names:
        texts = train_meta[train_meta["glaucoma"].map(label_map) == class_names.index(class_name)]["gpt4_summary"].tolist()
        if len(texts) == 0:
            # fallback: one zero vector
            features_list.append(torch.zeros(1, model.feat_dim))
        else:
            tokens = tokenize_text(args, texts, template=None)   # raw summaries, truncated to 77 tokens
            with torch.no_grad():
                feats = model.encode_text(tokens.to(args.device))   # (N, 512)
            features_list.append(feats)
    return features_list

def build_clustered_features(args, model, K=5):
    train_meta = pd.read_csv(args.data_setting["train_meta_path"])
    train_meta = train_meta.dropna(subset=["gpt4_summary"])
    class_names = args.data_setting["class_names"]
    label_map = {"no": 0, "yes": 1}

    features_list = []   # list of two tensors, each (K, 512)
    for class_name in class_names:
        texts = train_meta[train_meta["glaucoma"].map(label_map) == class_names.index(class_name)]["gpt4_summary"].tolist()
        if len(texts) == 0:
            features_list.append(torch.zeros(K, model.feat_dim))
        else:
            tokens = tokenize_text(args, texts, template=None)
            with torch.no_grad():
                feats = model.encode_text(tokens.to(args.device)).cpu().numpy()
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=K, random_state=args.random_seed)
            kmeans.fit(feats)
            centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            features_list.append(centroids)
    return torch.stack(features_list, dim=0)   # shape (C, K, D)