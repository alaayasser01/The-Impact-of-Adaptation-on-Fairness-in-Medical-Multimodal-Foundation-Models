import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CLIP(nn.Module):
    def __init__(self, backbone="ViT-B/32", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model, _ = clip.load(backbone, device="cpu")
        self.feat_dim = 512

    def forward_clip(self, images, text_features, *args, **kwargs):
        image_features = self.model.encode_image(images)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logits = 100.0 * image_features @ text_features.T

        return logits

    def encode_text(self, text):
        return self.model.encode_text(text.to(next(self.model.parameters()).device))

    def forward(self, *args, **kwargs):
        images = None

        if len(args) > 0:
            images = args[0]

        if images is None:
            images = kwargs.get("images", kwargs.get("pixel_values", None))

        if images is None and len(kwargs) > 0:
            images = list(kwargs.values())[0]

        if images is None or not isinstance(images, torch.Tensor):
            raise ValueError("CLIP.forward: Could not find image tensor in inputs.")

        return self.model.visual(images)

    def from_pretrained(self, path, *args, **kwargs):
        checkpoint = torch.load(path, map_location="cpu")

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith("module."):
                name = name.replace("module.", "")
            if name.startswith("model."):
                name = name.replace("model.", "")
            new_state_dict[name] = v

        msg = self.model.load_state_dict(new_state_dict, strict=False)
        print("\n" + "=" * 40)
        print("SUCCESS: Loaded weights from checkpoint")
        print(f"Matched Keys: {len(new_state_dict) - len(msg.unexpected_keys)}")
        print(f"Missing Keys: {len(msg.missing_keys)}")
        print("=" * 40 + "\n")