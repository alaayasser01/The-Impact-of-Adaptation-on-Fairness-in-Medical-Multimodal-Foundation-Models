#!/usr/bin/env python3
"""
Compare text embeddings for contrastive prompts using different CLIP text encoders.
Usage:
    python analyse_text_embeddings.py --text_sources custom --custom_prompts_file prompts.json
    python analyse_text_embeddings.py --text_sources custom --custom_prompts_file prompts.json --text_encoder openai
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ------------------------ Model loading ------------------------
def load_biomedclip(device):
    from open_clip import create_model_from_pretrained
    model, _ = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        device=device
    )
    model.feat_dim = 512
    return model

def load_openai_clip(device):
    import clip
    model, _ = clip.load("ViT-B/32", device=device)
    model.feat_dim = 512
    return model

def tokenize_texts(texts, encoder_type, context_length=77):
    """Tokenize a list of strings without requiring an args object."""
    if encoder_type == "biomedclip":
        from open_clip import get_tokenizer
        tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        return tokenizer(texts, context_length=256)   # BiomedCLIP uses context length 256
    elif encoder_type == "openai":
        import clip
        return clip.tokenize(texts, context_length=context_length, truncate=True)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

def encode_texts(model, texts, encoder_type, device, context_length=77, batch_size=32):
    """Encode a list of texts in batches, return stacked features on CPU."""
    all_feats = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenize_texts(batch_texts, encoder_type, context_length)
        with torch.no_grad():
            feats = model.encode_text(tokens.to(device))
        all_feats.append(feats.cpu())
        torch.cuda.empty_cache()
    return torch.cat(all_feats, dim=0)

# ------------------------ Custom prompt loading ------------------------
def load_custom_prompts(prompts_file=None, inline_prompts=None):
    if prompts_file:
        with open(prompts_file) as f:
            data = json.load(f)
        return data   # expected keys: "normal", "glaucoma"
    elif inline_prompts and len(inline_prompts) == 2:
        return {"normal": [inline_prompts[0]], "glaucoma": [inline_prompts[1]]}
    else:
        raise ValueError("Provide --custom_prompts_file or --custom_templates 'p0' 'p1'")

# ------------------------ Feature builders for existing sources ------------------------
def build_template_features(args, model, encoder_type, device, context_length=77):
    template = args.data_setting["text_template"]
    class_names = args.data_setting["class_names"]
    texts = [template.format(name) for name in class_names]
    tokens = tokenize_texts(texts, encoder_type, context_length)
    with torch.no_grad():
        feats = model.encode_text(tokens.to(device))
    return feats.cpu()

def build_gpt4_avg_features(args, model, encoder_type, device, context_length=77,
                            text_col='gpt4_summary', truncate_tail=True, max_chars=180):
    train_meta = pd.read_csv(args.data_setting["train_meta_path"])
    train_meta = train_meta.dropna(subset=[text_col])
    class_names = args.data_setting["class_names"]
    label_map = {"no": 0, "yes": 1}
    texts_by_class = {c: [] for c in class_names}
    for _, row in train_meta.iterrows():
        label_str = row["glaucoma"]
        if label_str in label_map:
            cls_idx = label_map[label_str]
            text = row[text_col].strip()
            if truncate_tail and len(text) > max_chars:
                text = text[-max_chars:]
            texts_by_class[class_names[cls_idx]].append(text)
    all_features = []
    for class_name in class_names:
        texts = texts_by_class[class_name]
        if len(texts) == 0:
            avg_feat = torch.zeros(model.feat_dim)
        else:
            feats = encode_texts(model, texts, encoder_type, device, context_length, batch_size=32)
            avg_feat = feats.mean(dim=0)
        all_features.append(avg_feat)
    return torch.stack(all_features, dim=0)

def build_all_summaries_features(args, model, encoder_type, device, context_length=77, text_col='gpt4_summary'):
    train_meta = pd.read_csv(args.data_setting["train_meta_path"])
    train_meta = train_meta.dropna(subset=[text_col])
    class_names = args.data_setting["class_names"]
    label_map = {"no": 0, "yes": 1}
    features_list = []
    for class_name in class_names:
        rows = train_meta[train_meta["glaucoma"].map(label_map) == class_names.index(class_name)]
        texts = rows[text_col].tolist()
        if len(texts) == 0:
            features_list.append(torch.zeros(1, model.feat_dim))
        else:
            feats = encode_texts(model, texts, encoder_type, device, context_length, batch_size=32)
            features_list.append(feats)
    return features_list

def build_clustered_features(args, model, encoder_type, device, context_length=77,
                             K=5, text_col='gpt4_summary'):
    train_meta = pd.read_csv(args.data_setting["train_meta_path"])
    train_meta = train_meta.dropna(subset=[text_col])
    class_names = args.data_setting["class_names"]
    label_map = {"no": 0, "yes": 1}
    from sklearn.cluster import KMeans
    all_features = []
    for class_name in class_names:
        rows = train_meta[train_meta["glaucoma"].map(label_map) == class_names.index(class_name)]
        texts = rows[text_col].tolist()
        if len(texts) == 0:
            all_features.append(torch.zeros(K, model.feat_dim))
        else:
            feats = encode_texts(model, texts, encoder_type, device, context_length, batch_size=32).cpu().numpy()
            kmeans = KMeans(n_clusters=K, random_state=args.random_seed).fit(feats)
            centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            all_features.append(centroids)
    return torch.stack(all_features, dim=0)

# ------------------------ Main text feature builder ------------------------
def build_text_features(args, model, encoder_type, device, source,
                        truncate_gpt4=True, text_col='gpt4_summary',
                        custom_prompts=None, context_length=77):
    if source == "template":
        return build_template_features(args, model, encoder_type, device, context_length)
    elif source == "gpt4_avg":
        return build_gpt4_avg_features(args, model, encoder_type, device, context_length,
                                       text_col=text_col, truncate_tail=truncate_gpt4)
    elif source == "all_summaries":
        return build_all_summaries_features(args, model, encoder_type, device, context_length, text_col=text_col)
    elif source == "clustered_summaries":
        K = args.model_setting.get("num_prototypes_per_class", 5)
        return build_clustered_features(args, model, encoder_type, device, context_length,
                                        K=K, text_col=text_col)
    elif source == "custom":
        # custom_prompts is a dict: {"normal": [...], "glaucoma": [...]}
        class_names = list(custom_prompts.keys())  # ["normal", "glaucoma"]
        all_class_feats = []
        for c in class_names:
            texts = custom_prompts[c]
            feats = encode_texts(model, texts, encoder_type, device, context_length, batch_size=32)
            all_class_feats.append(feats)
        return all_class_feats
    else:
        raise ValueError(f"Unknown text source: {source}")

# ------------------------ Metrics ------------------------
def compute_metrics(features):
    if isinstance(features, torch.Tensor):
        if features.dim() == 2:
            norm = features.norm(dim=-1)
            cos_sim = torch.nn.functional.cosine_similarity(features[0], features[1], dim=0).item()
            l2 = torch.norm(features[0] - features[1]).item()
            return {"cosine_sim": cos_sim, "l2": l2, "norm0": norm[0].item(), "norm1": norm[1].item()}
        elif features.dim() == 3:
            avg0 = features[0].mean(dim=0)
            avg1 = features[1].mean(dim=0)
            cos_sim = torch.nn.functional.cosine_similarity(avg0, avg1, dim=0).item()
            l2 = torch.norm(avg0 - avg1).item()
            intra_var0 = features[0].var(dim=0).sum().item()
            intra_var1 = features[1].var(dim=0).sum().item()
            return {"cosine_sim_centroid": cos_sim, "l2_centroid": l2, "intra_var0": intra_var0, "intra_var1": intra_var1}
    elif isinstance(features, list):
        feats0, feats1 = features[0], features[1]
        if feats0.shape[0] > 1 and feats1.shape[0] > 1:
            avg0 = feats0.mean(dim=0)
            avg1 = feats1.mean(dim=0)
            cos_sim = torch.nn.functional.cosine_similarity(avg0, avg1, dim=0).item()
            l2 = torch.norm(avg0 - avg1).item()
            intra_var0 = feats0.var(dim=0).sum().item()
            intra_var1 = feats1.var(dim=0).sum().item()
            return {"cosine_sim_centroid": cos_sim, "l2_centroid": l2, "intra_var0": intra_var0, "intra_var1": intra_var1}
    return {}

# ------------------------ Main ------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyse text embeddings with different CLIP encoders")
    parser.add_argument('--data_config', default='configs/datasets/FairVLMed10k.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--text_encoder', choices=['biomedclip', 'openai'], default='biomedclip')
    parser.add_argument('--text_sources', nargs='+',
                        default=['template', 'gpt4_avg', 'all_summaries', 'clustered_summaries'])
    parser.add_argument('--text_column', default='gpt4_summary')
    parser.add_argument('--no_truncate_gpt4', action='store_true')
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--custom_prompts_file', default=None)
    parser.add_argument('--custom_templates', nargs=2, default=None)
    args = parser.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU.")
        args.device = 'cpu'
    print(f"Using device: {args.device}")

    with open(args.data_config) as f:
        data_setting = json.load(f)

    print(f"Loading {args.text_encoder} text encoder...")
    if args.text_encoder == 'biomedclip':
        model = load_biomedclip(args.device)
    else:
        model = load_openai_clip(args.device)
    model.eval()
    print("Model loaded.\n")

    # Load custom prompts if needed
    custom_prompts = None
    if 'custom' in args.text_sources:
        custom_prompts = load_custom_prompts(args.custom_prompts_file, args.custom_templates)

    class TArgs:
        pass
    targs = TArgs()
    targs.data_setting = data_setting
    targs.model_setting = {'num_prototypes_per_class': 5}
    targs.random_seed = 0

    for source in args.text_sources:
        print(f"\n{'='*50}")
        print(f"Text source: {source}")
        print(f"{'='*50}")
        truncate = True
        if source == 'gpt4_avg' and args.no_truncate_gpt4:
            truncate = False
            print(f">>> Using full text from '{args.text_column}' (first 256 tokens)")
        elif source == 'gpt4_avg':
            print(f">>> Using tail‑truncated text from '{args.text_column}' (last 180 chars)")

        if source == 'custom':
            n_normal = len(custom_prompts.get('normal', []))
            n_glaucoma = len(custom_prompts.get('glaucoma', []))
            print(f">>> Custom prompts: {n_normal} normal, {n_glaucoma} glaucoma")

        features = build_text_features(targs, model, args.text_encoder, args.device,
                                       source=source,
                                       truncate_gpt4=truncate,
                                       text_col=args.text_column,
                                       custom_prompts=custom_prompts,
                                       context_length=args.context_length)
        torch.cuda.empty_cache()
        metrics = compute_metrics(features)
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")

        if source == 'gpt4_avg':
            train_meta = pd.read_csv(data_setting["train_meta_path"])
            train_meta = train_meta.dropna(subset=[args.text_column])
            n_normal = (train_meta["glaucoma"] == "no").sum()
            n_glaucoma = (train_meta["glaucoma"] == "yes").sum()
            print(f"  Num texts - Normal: {n_normal}, Glaucoma: {n_glaucoma}")

if __name__ == "__main__":
    main()