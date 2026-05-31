import os, json, torch, numpy as np, pandas as pd
from ptflops import get_model_complexity_info
from models.utils import get_model
from wrappers.utils import get_warpped_model

# ---- Configuration ----
BATCH_SIZE = 128
IMG_SIZE = (3, 224, 224)                     # as used in transforms
TRAIN_CSV = "data/Harvard FairVLMed (Harvard-FairVLMed）/train.csv"   # adjust if needed
OUTPUT_DIR = "./poster_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USAGES = ["lp", "clip-adapt", "full-ft", "full-ft-zs", "lora-lp", "lora-zs"]
BEST_LR = {
    "lp": 0.01,
    "clip-adapt": 0.01,
    "full-ft": 0.01,
    "full-ft-zs": 0.005,
    "lora-lp": 0.01,
    "lora-zs": 0.01,
}
BEST_EPOCH = {
    "lp": 88,
    "clip-adapt": 87,
    "full-ft": 8,
    "full-ft-zs": 13,
    "lora-lp": 25,
    "lora-zs": 28,
}

# ------------------------------------------------------------------
# 1. Minimal args to build the model & wrapper
# ------------------------------------------------------------------
class DummyArgs:
    task = "cls"
    model = "CLIP"
    backbone = "ViT-B/32"
    text_source = "template"
    proto_agg = "max"
    sensitive_name = "Sex"
    method = "erm"
    total_epochs = 100
    blr = 0.01
    weight_decay = 1e-4
    warmup_epochs = 5
    fixed_lr = False
    min_lr = 1e-5
    early_stopping = False
    cls_balance = False
    is_3d = False
    batch_size = 128
    context_length = 77
    exp_path = "./tmp"
    pretrained_path = ""
    device = torch.device("cpu")

    def __init__(self):
        # Load dataset & model configs
        with open("configs/datasets/FairVLMed10k.json") as f:
            self.data_setting = json.load(f)
        with open("configs/models/CLIP.json") as f:
            self.model_setting = json.load(f)

# Count training samples
def count_train_samples(csv_path):
    return len(pd.read_csv(csv_path))

# ------------------------------------------------------------------
# 2. Compute MACs for a single forward pass
# ------------------------------------------------------------------
def compute_macs(usage, lr):
    args = DummyArgs()
    args.usage = usage
    args.blr = lr
    args.lr = lr

    model = get_model(args).cpu()
    model.eval()
    wrapped = get_warpped_model(args, model).cpu()
    wrapped.eval()

    # ptflops expects (batch, C, H, W)
    macs, params = get_model_complexity_info(wrapped, IMG_SIZE, as_strings=False, print_per_layer_stat=False)
    return macs, params

# ------------------------------------------------------------------
# 3. Main
# ------------------------------------------------------------------
num_train = count_train_samples(TRAIN_CSV)
print(f"Training samples: {num_train}")

total_flops = {}
for usage in USAGES:
    lr = BEST_LR[usage]
    macs, _ = compute_macs(usage, lr)            # MACs per image
    flops_per_image = macs * 2                   # 1 MAC ≈ 2 FLOPs
    total_flops_val = flops_per_image * num_train * BEST_EPOCH[usage]
    total_flops[usage] = total_flops_val
    print(f"{usage:12s} → MACs: {macs:.2e}, total FLOPs: {total_flops_val:.2e}")

# ------------------------------------------------------------------
# 4. Write a CSV with FLOPs per usage
# ------------------------------------------------------------------
df_flops = pd.DataFrame([
    {"Usage": u, "Total FLOPs": total_flops[u]} for u in USAGES
])
df_flops.to_csv(os.path.join(OUTPUT_DIR, "flops_per_usage.csv"), index=False)
print(f"\nFLOPs data saved to {OUTPUT_DIR}/flops_per_usage.csv")