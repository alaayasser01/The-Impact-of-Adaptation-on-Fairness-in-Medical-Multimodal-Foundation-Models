# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from wrappers.base import BaseWrapper
# from peft import LoraConfig, get_peft_model


# class LoRAWrapper(BaseWrapper):
#     def __init__(self, model, lora_targets, *args, **kwargs) -> None:
#         super().__init__(model, *args, **kwargs)

#         self.model = get_peft_model(model, LoraConfig(lora_targets))

#         self.head = torch.nn.Linear(self.encoder.feat_dim, 1)

#         for name, param in self.model.named_parameters():
#             if "lora" not in name:
#                 param.requires_grad = False

#     def forward(self, x):
#         return self.head(self.encoder(x))



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from wrappers.base import BaseWrapper
# from peft import LoraConfig, get_peft_model


# class LoRAWrapper(BaseWrapper):
#     def __init__(self, model, lora_targets, *args, **kwargs) -> None:
#         super().__init__(model, *args, **kwargs)

#         self.model = get_peft_model(model, LoraConfig(target_modules=lora_targets))

#         self.head = torch.nn.Linear(self.encoder.feat_dim, 1)

#         for name, param in self.model.named_parameters():
#             if "lora" not in name:
#                 param.requires_grad = False

#     def forward(self, x):
#         return self.head(self.encoder(x))

import torch
import torch.nn as nn
from wrappers.base import BaseWrapper
from peft import LoraConfig, get_peft_model, TaskType

# class LoRAWrapper(BaseWrapper):
#     def __init__(self, model, lora_targets, *args, **kwargs) -> None:
#         # 1. Initialize the BaseWrapper
#         super().__init__(model, *args, **kwargs)

#         # 2. Define LoRA Config
#         # Using TaskType.FEATURE_EXTRACTION is best for CLIP-based classification
#         config = LoraConfig(
#             task_type=TaskType.FEATURE_EXTRACTION,
#             r=8, 
#             lora_alpha=16, 
#             target_modules=lora_targets, 
#             lora_dropout=0.1,
#             bias="none"
#         )

#         # 3. Wrap the model with LoRA
#         # Note: 'model' here is the CLIP object which has the 'feat_dim' attribute
#         self.model = get_peft_model(model, config)

#         # 4. Define the Head
#         # Use model.feat_dim (usually 512) and set output to 2 for CrossEntropy
#         self.head = torch.nn.Linear(model.feat_dim, 2)

#     def forward(self, x):
#         # Pass images through the LoRA-adapted CLIP
#         # This returns the pooled visual features
#         features = self.model(x)
        
#         # Pass features through our trainable classification head
#         return self.head(features)


import torch
import torch.nn as nn
from wrappers.base import BaseWrapper
from peft import LoraConfig, get_peft_model, TaskType

class LoRAWrapper(BaseWrapper):
    def __init__(self, model, lora_targets, *args, **kwargs) -> None:
        # 1. Initialize the BaseWrapper
        super().__init__(model, *args, **kwargs)

        # Grab the feature dimension before wrapping the model (usually 512 for ViT-B/32)
        # We check a few common places FairMedFM stores it to avoid AttributeErrors
        feat_dim = getattr(model, 'feat_dim', getattr(model, 'output_dim', 512))

        # 2. Define LoRA Config
        config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8, 
            lora_alpha=16, 
            target_modules=lora_targets, 
            lora_dropout=0.1,
            bias="none"
        )

        # 3. Wrap the model with LoRA
        self.model = get_peft_model(model, config)

        # 4. Define the Head for 2-class CrossEntropy
        self.head = torch.nn.Linear(feat_dim, 2)

    def forward(self, x):
        # SAFELY extract visual features depending on the CLIP architecture
        if hasattr(self.model, 'encode_image'):
            features = self.model.encode_image(x)
        else:
            features = self.model(x)
        
        # Pass features through our trainable classification head
        return self.head(features)