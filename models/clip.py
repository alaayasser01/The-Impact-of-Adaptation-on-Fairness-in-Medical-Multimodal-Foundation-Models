# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import clip


# class CLIP(nn.Module):
#     def __init__(self, backbone="ViT-B/32", *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         self.model, _ = clip.load(backbone, device="cpu")
#         self.feat_dim = 512

#     def forward_clip(self, images, text_features):
#         image_features = self.model.encode_image(images)

#         image_features = F.normalize(image_features, dim=-1)
#         text_features = F.normalize(text_features, dim=-1)

#         logits = 100.0 * image_features @ text_features.T

#         return logits

#     def encode_text(self, text):
#         return self.model.encode_text(text.to(next(self.model.parameters()).device))

#     def forward(self, images):
#         return self.model.visual(images)

#     def from_pretrained(self, path):
#         pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CLIP(nn.Module):
    def __init__(self, backbone="ViT-B/16", *args, **kwargs) -> None:
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
        
        # 1. Check if it's the first positional argument
        if len(args) > 0:
            images = args[0]
            
        # 2. If not, check if it's in kwargs under common names
        if images is None:
            images = kwargs.get("images", kwargs.get("pixel_values", None))
            
        # 3. If it's still None, it might be the only item in kwargs
        if images is None and len(kwargs) > 0:
            images = list(kwargs.values())[0]

        # Final check
        if images is None or not isinstance(images, torch.Tensor):
            # This will help us see exactly what IS being passed
            print(f"DEBUG: args={args}")
            print(f"DEBUG: kwargs keys={kwargs.keys()}")
            raise ValueError(f"CLIP.forward: Could not find image tensor in inputs.")

        # Pass only the tensor to the visual encoder
        return self.model.visual(images)

#### I added this function to load weights from a fine-tuned checkpoint into the internal CLIP model. This is necessary for the "clip-zs" usage where we want to evaluate a fine-tuned CLIP model in a zero-shot manner. The function handles different checkpoint formats and prints a report of matched, missing, and unexpected keys after loading. ####
    def from_pretrained(self, path, *args, **kwargs):
        import torch
        checkpoint = torch.load(path, map_location="cpu")
        
        # 1. Target the correct key you found in debug
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Fallback for other formats
            state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))

        # 2. Clean the prefixes
        # Checkpoints from FairCLIP often have 'module.' or 'visual.' prefixes
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            # Remove common prefixes that OpenAI CLIP doesn't use
            if name.startswith("module."):
                name = name.replace("module.", "")
            if name.startswith("model."):
                name = name.replace("model.", "")
            
            new_state_dict[name] = v

        # 3. Load weights
        msg = self.model.load_state_dict(new_state_dict, strict=False)
        
        print("\n" + "="*40)
        print(f"SUCCESS: Loaded weights from model_state_dict")
        print(f"Matched Keys: {len(new_state_dict) - len(msg.unexpected_keys)}")
        print(f"Missing Keys: {len(msg.missing_keys)}")
        print("="*40 + "\n")
        
        # FINAL DEBUG: If Matched Keys is still 0, we need to see one name
        if (len(new_state_dict) - len(msg.unexpected_keys)) == 0:
            print(f"CRITICAL: Still 0 matches. Example key in file: {list(new_state_dict.keys())[0]}")
            print(f"Example key expected by model: {list(self.model.state_dict().keys())[0]}")