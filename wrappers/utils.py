

import json
import torch

import models
from utils.static import CLIP_MODELS
from utils.text_sources import build_text_features
import pandas as pd
import clip

def get_warpped_model(args, model, data_engine=None):

    if args.task != "cls":
        raise NotImplementedError(
            f"Task '{args.task}' is not implemented"
        )

    # -------------------------------------------------
    # PURE VISION METHODS
    # -------------------------------------------------

    if args.usage == "lp":
        from wrappers import LinearProbeWrapper
        return LinearProbeWrapper(model)

    elif args.usage == "full-ft":
        from wrappers.full_ft import FullFTWrapper
        return FullFTWrapper(model)

    elif args.usage == "lora-lp":
        from wrappers.lora import LoRALPWrapper

        targets = args.model_setting.get(
            "lora_targets",
            ["in_proj", "out_proj", "c_fc", "c_proj"]
        ) if args.model_setting else [
            "in_proj", "out_proj", "c_fc", "c_proj"
        ]

        return LoRALPWrapper(
            model,
            lora_targets=targets
        )

    # -------------------------------------------------
    # TEXT-GUIDED METHODS
    # -------------------------------------------------

    elif args.usage in [
        "clip-zs",
        "clip-adapt",
        "lora-zs",
        "full-ft-zs",
    ]:

        assert args.model in CLIP_MODELS, (
            f"{args.usage} is not applicable for {args.model}"
        )

        text_features = build_text_features(args, model)

        # -----------------------------
        # CLIP prototype tuning
        # -----------------------------
        if args.usage in [
            "clip-zs",
            "clip-adapt",
        ]:

            from wrappers import CLIPWrapper

            return CLIPWrapper(
                model,
                text_features,
                proto_agg=args.proto_agg
            )

        # -----------------------------
        # LoRA zero-shot alignment
        # -----------------------------

        elif args.usage == "lora-zs":

            from wrappers.lora import LoRAZSWrapper

            targets = args.model_setting.get(
                "lora_targets",
                ["in_proj", "out_proj", "c_fc", "c_proj"]
            ) if args.model_setting else [
                "in_proj", "out_proj", "c_fc", "c_proj"
            ]

            return LoRAZSWrapper(
                model,
                text_features,
                lora_targets=targets,
                proto_agg=args.proto_agg
            )

        # -----------------------------
        # Full fine-tuning with text anchors
        # -----------------------------

        elif args.usage == "full-ft-zs":

            from wrappers.full_ft import FullFTZSWrapper

            return FullFTZSWrapper(
                model,
                text_features,
                proto_agg=args.proto_agg
            )

    else:
        raise NotImplementedError(
            f"Usage '{args.usage}' is not implemented"
        )
        