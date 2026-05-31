# import json
# import os
# import random

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from icecream import ic

# import parse_args
# from datasets.utils import get_dataset
# from models.utils import get_model
# from trainers.utils import get_trainer
# from utils import basics
# from wrappers.utils import get_warpped_model

# os.environ["WANDB_DISABLED"] = "true"


# def create_exerpiment_setting(args):
#     # get hash
#     args.device = torch.device("cuda" if args.cuda else "cpu")
#     args.lr = args.blr

#     args.save_folder = os.path.join(
#         args.exp_path,
#         args.task,
#         args.usage,
#         args.method,
#         args.dataset,
#         args.model,
#         args.sensitive_name,
#         f"seed{args.random_seed}",
#     )

#     args.resume_path = args.save_folder
#     basics.creat_folder(args.save_folder)

#     try:
#         with open(f"configs/datasets/{args.dataset}.json", "r") as f:
#             data_setting = json.load(f)
#             data_setting["augment"] = False
#             data_setting["test_meta_path"] = data_setting[
#                 f"test_{str.lower(args.sensitive_name)}_meta_path"]
#             args.data_setting = data_setting

#             if args.pos_class is not None:
#                 args.data_setting["pos_class"] = args.pos_class
#     except:
#         args.data_setting = None

#     try:
#         with open(f"configs/models/{args.model}.json", "r") as f:
#             args.model_setting = json.load(f)
#     except:
#         args.model_setting = None

#     return args


# if __name__ == "__main__":
#     args = parse_args.collect_args()
#     args = create_exerpiment_setting(args)

#     logger = basics.setup_logger(
#         "train", args.save_folder, "history.log", screen=True, tofile=True)
#     logger.info("Using following arguments for training.")
#     logger.info(args)

#     torch.manual_seed(args.random_seed)
#     np.random.seed(args.random_seed)
#     random.seed(args.random_seed)
#     if args.cuda:
#         torch.cuda.manual_seed(args.random_seed)
#         torch.cuda.manual_seed_all(args.random_seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

#     train_data, train_dataloader, train_meta = get_dataset(args, split="train")
#     test_data, test_dataloader, test_meta = get_dataset(args, split="test")
#     model = get_model(args).to(args.device)

#     # ic(train_data, train_dataloader, train_meta)
#     # ic(test_data, test_dataloader, test_meta)
#     # ic(model)

#     if args.task == "cls":
#         model = get_warpped_model(args, model).to(args.device)
#     elif args.task == "seg":
#         model = get_warpped_model(args, model, test_data).to(
#             args.device)  # SAMLearner

#     trainer = get_trainer(args, model, logger, test_dataloader)

#     if args.usage == "clip-zs":
#         logger.info("CLIP Zero-shot performance:")
#         trainer.evaluate(test_dataloader, save_path=os.path.join(
#             args.save_folder, "clip_zs_final"))
#         exit(0)

#     elif args.usage == "clip-adapt":
#         logger.info("CLIP-Adaptor performance:")
#         trainer.init_optimizers()
#         trainer.train(train_dataloader)
#         trainer.evaluate(test_dataloader, save_path=os.path.join(
#             args.save_folder, "clip_adaptor_final"))
#         exit(0)

#     elif args.usage == "lp":
#         logger.info("Linear probing performance:")
#         trainer.init_optimizers()
#         trainer.train(train_dataloader)
#         trainer.evaluate(test_dataloader, save_path=os.path.join(
#             args.save_folder, "lp_final"))
#         exit(0)

#     elif args.usage == "seg2d":
#         logger.info(f"2D SegFM using {args.prompt} prompt performance:")
#         trainer.evaluate(test_dataloader, save_path=os.path.join(
#             args.save_folder, args.prompt))
#         exit(0)

#     # elif args.usage == "seg2d-rands":
#     #     logger.info("2D SegFM using 5 random points prompt performance:")
#     #     trainer.evaluate(test_dataloader, save_path=os.path.join(
#     #         args.save_folders, "rands"))
#     #     exit(0)

#     # elif args.usage == "seg2d-bbox":
#     #     logger.info("2D SegFM using 1 bounding box prompt performance:")
#     #     trainer.evaluate(test_dataloader, save_path=os.path.join(
#     #         args.save_folder, "bbox"))
#     #     exit(0)

#     elif args.usage == "seg3d-center":
#         # TODO
#         logger.info("3D SegFM using 1 center point prompt performance:")
#         trainer.evaluate(test_dataloader, save_path=os.path.join(
#             args.save_folder, "center"))
#         exit(0)

#     else:
#         raise NotImplementedError


import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

import parse_args
from datasets.utils import get_dataset
from models.utils import get_model
from trainers.utils import get_trainer
from utils import basics
from wrappers.utils import get_warpped_model

os.environ["WANDB_DISABLED"] = "true"


def create_exerpiment_setting(args):
    # get hash
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.lr = args.blr
    
    # ####I added this####
    # usage_path_name = args.usage
    # if args.usage == 'clip-zs' and getattr(args, 'pretrained_path', None):
    #     usage_path_name = "full-ft-eval"
    # ################

    safe_backbone = str(args.backbone).replace("/", "_") if hasattr(args, 'backbone') else "NoBackbone"
    
    # 2. Format the hyperparameters for the folder name
    # e.g., "100ep_0.025lr"
    # hyperparams = f"{args.total_epochs}ep_{args.lr}lr"
    # hyperparams = f"ep{args.total_epochs}_lr{args.lr}"
    # # 3. Build the highly detailed, self-documenting path!
    # args.save_folder = os.path.join(
    #     args.exp_path,            # ./output
    #     args.task,                # cls
    #     args.usage,               # clip-adapt
    #     safe_backbone,            # ViT-B_32
    #     args.method,              # erm
    #     hyperparams,              # 100ep_0.025lr
    #     args.dataset,             # FairVLMed10k
    #     args.model,               # CLIP
    #     args.sensitive_name,      # All
    #     f"seed{args.random_seed}",# seed0
    # )

    hyperparams = f"ep{args.total_epochs}_lr{args.lr}"

    text_source = getattr(args, "text_source", "template")
    args.save_folder = os.path.join(
        args.exp_path,
        args.task,
        args.usage,
        text_source,                 # <-- ADD THIS
        safe_backbone,
        args.method,
        hyperparams,
        args.dataset,
        args.model,
        args.sensitive_name,
        f"seed{args.random_seed}",
    )

    # args.save_folder = os.path.join(
    #     args.exp_path,
    #     args.task,
    #     args.usage,
    #     args.method,
    #     args.dataset,
    #     args.model,
    #     args.sensitive_name,
    #     f"seed{args.random_seed}",
    # )

    args.resume_path = args.save_folder
    basics.creat_folder(args.save_folder)

    # try:
    #     with open(f"configs/datasets/{args.dataset}.json", "r") as f:
    #         data_setting = json.load(f)
    #         data_setting["augment"] = False
    #         data_setting["test_meta_path"] = data_setting[
    #             f"test_{str.lower(args.sensitive_name)}_meta_path"]
    #         args.data_setting = data_setting

    #         if args.pos_class is not None:
    #             args.data_setting["pos_class"] = args.pos_class
    # except:
    #     args.data_setting = None
    try:
        with open(f"configs/datasets/{args.dataset}.json", "r") as f:
            data_setting = json.load(f)
            data_setting["augment"] = False
            
            if args.sensitive_name != "All":
                specific_test_key = f"test_{str.lower(args.sensitive_name)}_meta_path"
                if specific_test_key in data_setting:
                    data_setting["test_meta_path"] = data_setting[specific_test_key]
            else:
                # NEW: Force it to use the master test file when "All" is selected!
                # We just borrow the path from sex, since it already points to 'test.csv'
                data_setting["test_meta_path"] = data_setting.get("test_sex_meta_path")
                
            args.data_setting = data_setting

            if args.pos_class is not None:
                args.data_setting["pos_class"] = args.pos_class
                
    except Exception as e:
        # Stop hiding the errors! Print exactly why it failed.
        print(f"\n❌ CRITICAL ERROR loading dataset config for {args.dataset}: {e}\n")
        args.data_setting = None

    try:
        with open(f"configs/models/{args.model}.json", "r") as f:
            args.model_setting = json.load(f)
    except:
        args.model_setting = None

    return args

# def evaluate_all_attributes(args, trainer, attributes_to_test, save_prefix):
#     for attr in attributes_to_test:
#         logger.info(f"\n--- Evaluating {attr} ---")

#         args.sensitive_name = attr

#         _, specific_test_loader, _ = get_dataset(args, split="test")

#         trainer.evaluate(
#             specific_test_loader,
#             save_path=os.path.join(
#                 args.save_folder,
#                 f"{save_prefix}_{attr}"
#             )
#         )

def evaluate_all_attributes(args, trainer, logger, attributes_to_test, save_prefix):
    for attr in attributes_to_test:
        logger.info(f"\n--- Evaluating {attr} ---")
        args.sensitive_name = attr
        _, specific_test_loader, _ = get_dataset(args, split="test")
        trainer.evaluate(
            specific_test_loader,
            save_path=os.path.join(args.save_folder, f"{save_prefix}_{attr}")
        )        
if __name__ == "__main__":
    args = parse_args.collect_args()
    args = create_exerpiment_setting(args)

    logger = basics.setup_logger(
        "train", args.save_folder, "history.log", screen=True, tofile=True)
    logger.info("Using following arguments for training.")
    logger.info(args)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_data, train_dataloader, train_meta = get_dataset(args, split="train")
    test_data, test_dataloader, test_meta = get_dataset(args, split="test")
    val_loader = None
    if args.usage not in ["clip-zs"]:   # only for trainable usages
        if "validation_meta_path" in args.data_setting:
            args.data_setting["image_val_path"] = args.data_setting["image_validation_path"]
            _, val_loader, _ = get_dataset(args, split="validation")
    model = get_model(args).to(args.device)
 
  ##I added ##
    # if getattr(args, 'pretrained_path', None) and args.pretrained_path != "":
    #     # This calls the function we just wrote in clip.py
    #     model.from_pretrained(args.pretrained_path)
    #     ####
        
    # ic(train_data, train_dataloader, train_meta)
    # ic(test_data, test_dataloader, test_meta)
    # ic(model)

    if args.task == "cls":
        model = get_warpped_model(args, model).to(args.device)
    elif args.task == "seg":
        model = get_warpped_model(args, model, test_data).to(
            args.device)  # SAMLearner

    # trainer = get_trainer(args, model, logger, test_dataloader)
    trainer = get_trainer(args, model, logger, test_dataloader, val_loader)

# ==========================================================
    # Determine which attributes to evaluate based on the argument
    # ==========================================================
    if args.sensitive_name == "All":
        attributes_to_test = ["Sex","Age", "Race", "Language", "Ethnicity", "MaritalStatus"]
        logger.info("====== INITIATING MULTI-ATTRIBUTE EVALUATION ======")
    else:
        attributes_to_test = [args.sensitive_name]

    if getattr(args, 'eval_only', False):
        import glob as globlib
        search_dir = args.exp_path
        matches = globlib.glob(os.path.join(search_dir, "**", "best_ckpt.pth"), recursive=True)
        if not matches:
            raise FileNotFoundError(f"No best_ckpt.pth found inside {search_dir}")
        ckpt_path = matches[0]   # there will be exactly one
        logger.info(f"Found checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        logger.info(f"Loaded checkpoint from {ckpt_path}")
        evaluate_all_attributes(args, trainer, logger, attributes_to_test,
                                save_prefix=f"{args.usage}_eval")
        exit(0)
    # ==========================================
    # ZERO-SHOT ONLY
    # ==========================================
    if args.usage == "clip-zs":

        logger.info("Evaluating CLIP Zero-Shot...")

        evaluate_all_attributes(
            args,
            trainer,
            logger,
            attributes_to_test,
            save_prefix="clip_zs_final"
        )
        exit(0)

    # ==========================================
    # TRAINING METHODS
    # ==========================================
    elif args.usage in [
        "clip-adapt",
        "lp",
        "full-ft",
        "full-ft-zs",
        "lora-lp",
        "lora-zs"
    ]:

        logger.info(f"{args.usage} training...")

        trainer.init_optimizers()

        trainer.train(train_dataloader, val_dataloader=val_loader)

        evaluate_all_attributes(
            args,
            trainer,
            logger,
            attributes_to_test,
            save_prefix=f"{args.usage}_final"
        )
        exit(0)

    # ==========================================
    # SEGMENTATION
    # ==========================================
    elif args.usage == "seg2d":

        trainer.evaluate(
            test_dataloader,
            save_path=os.path.join(args.save_folder, args.prompt)
        )

    elif args.usage == "seg3d-center":

        trainer.evaluate(
            test_dataloader,
            save_path=os.path.join(args.save_folder, "center")
        )

    else:
        raise NotImplementedError
        # elif args.usage == "lora":
        #     logger.info("LoRA Finetuning performance:")
        #     trainer.init_optimizers()
            
        #     # Train ONCE
        #     trainer.train(train_dataloader)
            
        #     # Evaluate multiple times
        #     for attr in attributes_to_test:
        #         logger.info(f"\n--- Evaluating {attr} ---")
        #         args.sensitive_name = attr
        #         _, specific_test_loader, _ = get_dataset(args, split="test")
                
        #         trainer.evaluate(specific_test_loader, save_path=os.path.join(
        #             args.save_folder, f"lora_final_{attr}"))
        #     exit(0)

        # elif args.usage == "seg2d":
        #     logger.info(f"2D SegFM using {args.prompt} prompt performance:")
        #     trainer.evaluate(test_dataloader, save_path=os.path.join(
        #         args.save_folder, args.prompt))
        #     exit(0)

        # elif args.usage == "seg3d-center":
        #     logger.info("3D SegFM using 1 center point prompt performance:")
        #     trainer.evaluate(test_dataloader, save_path=os.path.join(
        #         args.save_folder, "center"))
        #     exit(0)

        # else:
        #     raise NotImplementedError

#### I edited this to include the full-ft evaluation when pretrained_path is provided for clip-zs usage.####
#     if args.usage == "clip-zs":
#         logger.info(f"Evaluating CLIP model...")
#         trainer.evaluate(test_dataloader, save_path=os.path.join(
#             args.save_folder, "eval_results"))
#         exit(0)
#         # if getattr(args, 'pretrained_path', None):
#         #     logger.info("="*30)
#         #     logger.info(f"FULL FINE-TUNED EVALUATION")
#         #     logger.info(f"Checkpoint: {args.pretrained_path}")
#         #     logger.info("="*30)
#         #     save_name = "full_ft_eval_final"
#         # else:
#         #     logger.info("CLIP Zero-shot performance (Standard Weights):")
#         #     save_name = "clip_zs_final"

#         # trainer.evaluate(test_dataloader, save_path=os.path.join(
#         #     args.save_folder, save_name))
#         # exit(0)
# #################
#     elif args.usage == "clip-adapt":
#         logger.info("CLIP-Adaptor performance:")
#         trainer.init_optimizers()
#         trainer.train(train_dataloader)
#         trainer.evaluate(test_dataloader, save_path=os.path.join(
#             args.save_folder, "clip_adaptor_final"))
#         exit(0)

#     elif args.usage == "lp":
#         logger.info("Linear probing performance:")
#         trainer.init_optimizers()
#         trainer.train(train_dataloader)
#         trainer.evaluate(test_dataloader, save_path=os.path.join(
#             args.save_folder, "lp_final"))
#         exit(0)

#     elif args.usage == "lora":
#             logger.info("LoRA Finetuning performance:")
#             trainer.init_optimizers()
#             trainer.train(train_dataloader)
#             # Final official evaluation
#             trainer.evaluate(test_dataloader, save_path=os.path.join(
#                 args.save_folder, "lora_final"))
#             exit(0)
            
    # elif args.usage == "seg2d":
    #     logger.info(f"2D SegFM using {args.prompt} prompt performance:")
    #     trainer.evaluate(test_dataloader, save_path=os.path.join(
    #         args.save_folder, args.prompt))
    #     exit(0)

    # # elif args.usage == "seg2d-rands":
    # #     logger.info("2D SegFM using 5 random points prompt performance:")
    # #     trainer.evaluate(test_dataloader, save_path=os.path.join(
    # #         args.save_folders, "rands"))
    # #     exit(0)

    # # elif args.usage == "seg2d-bbox":
    # #     logger.info("2D SegFM using 1 bounding box prompt performance:")
    # #     trainer.evaluate(test_dataloader, save_path=os.path.join(
    # #         args.save_folder, "bbox"))
    # #     exit(0)

    # elif args.usage == "seg3d-center":
    #     # TODO
    #     logger.info("3D SegFM using 1 center point prompt performance:")
    #     trainer.evaluate(test_dataloader, save_path=os.path.join(
    #         args.save_folder, "center"))
    #     exit(0)

    # else:
    #     raise NotImplementedError