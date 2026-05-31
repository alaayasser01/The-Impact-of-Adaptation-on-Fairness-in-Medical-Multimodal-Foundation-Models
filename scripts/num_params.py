import json, torch, sys
from models.utils import get_model
from wrappers.utils import get_warpped_model

usages = ["lp", "clip-adapt", "full-ft", "full-ft-zs", "lora-lp", "lora-zs"]

# Minimal args that will be filled properly
class Args:
    pass

args = Args()
args.task = "cls"
args.model = "CLIP"
args.backbone = "ViT-B/32"
args.text_source = "template"
args.proto_agg = "max"
args.sensitive_name = "Sex"
args.method = "erm"
args.total_epochs = 100
args.blr = 0.001
args.weight_decay = 1e-4
args.warmup_epochs = 5
args.fixed_lr = False
args.min_lr = 1e-5
args.early_stopping = False
args.cls_balance = False
args.is_3d = False
args.batch_size = 128
args.context_length = 77
args.exp_path = "./tmp"
args.pretrained_path = ""
args.device = torch.device("cpu")

# Load dataset config (needed for text template, class names)
with open("configs/datasets/FairVLMed10k.json") as f:
    args.data_setting = json.load(f)

# Load model config (needed for lora_targets)
with open("configs/models/CLIP.json") as f:
    args.model_setting = json.load(f)

for usage in usages:
    args.usage = usage
    model = get_model(args)
    wrapped = get_warpped_model(args, model)
    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    print(f"{usage}: {trainable:,} trainable parameters")