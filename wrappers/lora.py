
import torch
import torch.nn as nn
import torch.nn.functional as F
from wrappers.base import BaseWrapper
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# 1. LoRA + Linear Probing (Vision Only)
# ==========================================
class LoRALPWrapper(BaseWrapper):
    def __init__(self, model, lora_targets=["in_proj", "out_proj", "c_fc", "c_proj"], *args, **kwargs):
        super().__init__(model, *args, **kwargs)

        feat_dim = getattr(model, 'feat_dim', getattr(model, 'output_dim', 512))

        config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,               # r=8 is generally more competitive for medical imaging than r=16
            lora_alpha=16,      # alpha usually matches r for standard scaling
            target_modules=lora_targets,
            lora_dropout=0.1,
            bias="none"
        )

        self.model = get_peft_model(model, config)
        self.head = torch.nn.Linear(feat_dim, 2)

    def forward(self, x):
        # 1. Extract visual features using the LoRA-adapted backbone
        features = self.model(x) if not hasattr(self.model, 'encode_image') else self.model.encode_image(x)
        # 2. Pass directly to the new linear head (ignoring text entirely)
        return self.head(features)


# ==========================================
# 2. LoRA + Zero-Shot (Vision aligned to Text)
# ==========================================
class LoRAZSWrapper(BaseWrapper):
    def __init__(self, model, base_text_features, lora_targets=["in_proj", "out_proj", "c_fc", "c_proj"], proto_agg="max", *args, **kwargs):
        super().__init__(model, *args, **kwargs)

        # 1. Handle prototype shapes
        if isinstance(base_text_features, torch.Tensor):
            if base_text_features.dim() == 2:
                # Single prototype per class → add K=1 dimension
                base_text_features = base_text_features.unsqueeze(1)   # (C, D) -> (C, 1, D)
            elif base_text_features.dim() != 3:
                raise ValueError("Prototypes tensor must be 2D or 3D")
        elif isinstance(base_text_features, list):
            # Convert list of tensors to padded 3D tensor
            max_len = max(f.size(0) for f in base_text_features)
            proto_list = []
            for f in base_text_features:
                if f.size(0) < max_len:
                    pad = torch.zeros(max_len - f.size(0), f.size(1), device=f.device)
                    f = torch.cat([f, pad], dim=0)
                proto_list.append(f)
            base_text_features = torch.stack(proto_list, dim=0)   # (C, max_len, D)
        else:
            raise TypeError("Prototypes must be a tensor or list of tensors")

        # 2. Register as non‑trainable buffer (frozen)
        self.register_buffer('base_text_features', base_text_features.clone().detach())

        # 3. Store aggregation method
        self.agg = proto_agg

        # 4. Apply LoRA
        config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=16,
            target_modules=lora_targets,
            lora_dropout=0.1,
            bias="none"
        )
        self.model = get_peft_model(model, config)

    def forward(self, x):
        if hasattr(self.model, 'encode_image'):
            image_features = self.model.encode_image(x)
        else:
            image_features = self.model(x)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(self.base_text_features, dim=-1)   # (C, K, D)
        sim = 100.0 * torch.einsum('bd,ckd->bck', image_features, text_features)
        if sim.size(2) == 1:
            logits = sim.squeeze(2)
        else:
            if self.agg == 'max':
                logits, _ = sim.max(dim=2)
            else:
                logits = sim.mean(dim=2)
        return logits