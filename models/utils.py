import json
import inspect
import torch
from icecream import ic
# from segment_anything.build_sam import sam_model_registry

import models
# from models.sam_builder.build_sammed2d import sam_model_registry1
# from models.sam_builder.build_tinysam import sam_model_registry2


# def get_model(args):
#     if args.task == "cls":
#         model_setting = args.model_setting

#         model_name = getattr(models, args.model)

#         model = model_name()

#         if model_setting is not None and "pretrained_path" in model_setting.keys():
#             model.from_pretrained(model_setting["pretrained_path"])

#     elif args.task == "seg":
#         sam_checkpoint = args.sam_ckpt_path
#         if sam_checkpoint == None:
#             raise ValueError(
#                 "SAM checkpoint path is required for segmentation task!")
#         # ic(sam_checkpoint)
#         if args.model in ["SAM", "MedSAM", "MedSAMAdaptor"]:
#             model = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
#         elif args.model in ["SAMMed2D", "FT-SAM"]:
#             model = sam_model_registry1['vit_b'](args)
#         elif args.model in ["TinySAM", "MobileSAM"]:
#             model = sam_model_registry2['vit_t'](checkpoint=sam_checkpoint)
#         else:
#             raise ValueError("Invalid model name!")

#         print(f"Weights of model: #{args.model}# is loaded!")

#     else:
#         raise NotImplementedError("Task type not supported!")

#     return model


def get_model(args):
    if args.task == "cls":
        model_setting = args.model_setting or {}
        model_class = getattr(models, args.model)

        # 1. Determine Backbone (Priority: JSON > CLI > Default)
        backbone = model_setting.get("backbone", getattr(args, "backbone", "ViT-B/32"))

        # 2. Intelligent Initialization
        # We check if the model's __init__ actually accepts a 'backbone' argument
        signature = inspect.signature(model_class.__init__)
        if "backbone" in signature.parameters:
            model = model_class(backbone=backbone)
        else:
            model = model_class()

        # 3. Determine Pretrained Path (Priority: CLI > JSON)
        pretrained_path = getattr(args, "pretrained_path", "")
        if not pretrained_path:
            pretrained_path = model_setting.get("pretrained_path", "")

        # 4. Universal Weight Loading
        if pretrained_path:
            if hasattr(model, "from_pretrained"):
                print(f">>> Loading weights into {args.model} from: {pretrained_path}")
                model.from_pretrained(pretrained_path)
            else:
                # Fallback for standard PyTorch models without our custom method
                print(f">>> Standard load for {args.model} from: {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)

    elif args.task == "seg":
#          sam_checkpoint = args.sam_ckpt_path
#         if sam_checkpoint == None:
#             raise ValueError(
#                 "SAM checkpoint path is required for segmentation task!")
#         # ic(sam_checkpoint)
#         if args.model in ["SAM", "MedSAM", "MedSAMAdaptor"]:
#             model = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
#         elif args.model in ["SAMMed2D", "FT-SAM"]:
#             model = sam_model_registry1['vit_b'](args)
#         elif args.model in ["TinySAM", "MobileSAM"]:
#             model = sam_model_registry2['vit_t'](checkpoint=sam_checkpoint)
#         else:
#             raise ValueError("Invalid model name!")

#         print(f"Weights of model: #{args.model}# is loaded!")

#     else:
#         raise NotImplementedError("Task type not supported!")
        pass

    return model