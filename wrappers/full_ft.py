import torch
import torch.nn as nn
import torch.nn.functional as F
from wrappers.base import BaseWrapper

class FullFTWrapper(BaseWrapper):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        
        # Grab the feature dimension (usually 512 for ViT-B/32)
        feat_dim = getattr(model, 'feat_dim', getattr(model, 'output_dim', 512))
        
        # 1. UNFREEZE THE ENTIRE VISION BACKBONE
        # We explicitly ensure every parameter can calculate gradients
        for param in self.model.parameters():
            param.requires_grad = True
            
        # 2. Add the trainable classification head
        self.head = torch.nn.Linear(feat_dim, 2)

    def forward(self, x):
        # Extract visual features dynamically using the fully unfrozen backbone
        if hasattr(self.model, 'encode_image'):
            features = self.model.encode_image(x)
        else:
            features = self.model(x)
            
        # Pass features through our trainable classification head
        return self.head(features)
    
class FullFTZSWrapper(BaseWrapper):
    def __init__(self, model, base_text_features, proto_agg="max", *args, **kwargs):
        super().__init__(model, *args, **kwargs)

        # ---------- Handle prototype shapes ----------
        if isinstance(base_text_features, torch.Tensor):
            if base_text_features.dim() == 2:
                base_text_features = base_text_features.unsqueeze(1)   # (C, D) -> (C, 1, D)
            elif base_text_features.dim() != 3:
                raise ValueError("Prototypes tensor must be 2D or 3D")
        elif isinstance(base_text_features, list):
            max_len = max(f.size(0) for f in base_text_features)
            proto_list = []
            for f in base_text_features:
                if f.size(0) < max_len:
                    pad = torch.zeros(max_len - f.size(0), f.size(1), device=f.device)
                    f = torch.cat([f, pad], dim=0)
                proto_list.append(f)
            base_text_features = torch.stack(proto_list, dim=0)   # (C, max_len, D)
        else:
            raise TypeError("base_text_features must be a tensor or list of tensors")

        self.register_buffer('base_text_features', base_text_features.clone().detach())
        self.agg = proto_agg

        for param in self.model.parameters():
            param.requires_grad = True

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