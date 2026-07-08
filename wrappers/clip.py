import torch
import torch.nn as nn
import torch.nn.functional as F

from wrappers.base import BaseWrapper

class CLIPWrapper(BaseWrapper):
    def __init__(self, model, base_text_features, proto_agg="max", *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.model = model
        if isinstance(base_text_features, torch.Tensor):
            if base_text_features.dim() == 2:
                base_text_features = base_text_features.unsqueeze(1)   # C,D -> C,1,D
            self.prototypes = nn.Parameter(base_text_features.clone())
        elif isinstance(base_text_features, list):
            # convert list of tensors to a single 3D tensor (C, max_len, D)
            max_len = max(f.size(0) for f in base_text_features)
            proto_list = []
            for f in base_text_features:
                if f.size(0) < max_len:
                    pad = torch.zeros(max_len - f.size(0), f.size(1), device=f.device)
                    f = torch.cat([f, pad], dim=0)
                proto_list.append(f)
            self.prototypes = nn.Parameter(torch.stack(proto_list, dim=0))
        else:
            raise TypeError("base_text_features must be a tensor or list of tensors")
        self.agg = proto_agg
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        if hasattr(self.model, 'encode_image'):
            image_features = self.model.encode_image(x)
        else:
            image_features = self.model(x)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(self.prototypes, dim=-1)   # (C, K, D) or (C, 1, D)
        sim = 100.0 * torch.einsum('bd,ckd->bck', image_features, text_features)
        if sim.size(2) == 1:
            logits = sim.squeeze(2)
        else:
            if self.agg == 'max':
                logits, _ = sim.max(dim=2)
            else:
                logits = sim.mean(dim=2)
        return logits