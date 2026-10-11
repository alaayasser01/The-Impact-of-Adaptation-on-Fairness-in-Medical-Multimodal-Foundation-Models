import inspect
import json
import torch

import models


def get_model(args):
    if args.task != "cls":
        raise NotImplementedError(f"Task '{args.task}' is not implemented.")

    model_setting = args.model_setting or {}
    model_class = getattr(models, args.model)

    # 1. Determine Backbone (Priority: JSON > CLI > Default)
    backbone = model_setting.get("backbone", getattr(args, "backbone", "ViT-B/32"))

    # 2. Initialization with backbone if accepted
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
            print(f">>> Standard load for {args.model} from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)

    return model