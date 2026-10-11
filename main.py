import glob as globlib
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import parse_args
from datasets.utils import get_dataset
from models.utils import get_model
from trainers.utils import get_trainer
from utils import basics
from wrappers.utils import get_warpped_model

os.environ["WANDB_DISABLED"] = "true"


def create_exerpiment_setting(args):
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.lr = args.blr

    safe_backbone = str(args.backbone).replace("/", "_") if hasattr(args, "backbone") else "NoBackbone"
    hyperparams = f"ep{args.total_epochs}_lr{args.lr}"
    text_source = getattr(args, "text_source", "template")

    args.save_folder = os.path.join(
        args.exp_path,
        args.task,
        args.usage,
        text_source,
        safe_backbone,
        args.method,
        hyperparams,
        args.dataset,
        args.model,
        args.sensitive_name,
        f"seed{args.random_seed}",
    )

    args.resume_path = args.save_folder
    basics.creat_folder(args.save_folder)

    try:
        with open(f"configs/datasets/{args.dataset}.json", "r") as f:
            data_setting = json.load(f)
            data_setting["augment"] = False

            # Automatically resolve paths if folder is located at ./<name> or has alternate naming
            first_meta = data_setting.get("train_meta_path", "")
            if not os.path.exists(first_meta):
                for k, v in list(data_setting.items()):
                    if isinstance(v, str) and v.startswith("./data/"):
                        alt_path = v.replace("./data/", "./")
                        if os.path.exists(alt_path):
                            data_setting[k] = alt_path

                # If still not found, search for matching folders (e.g. 'Harvard FairVLMed (Harvard-FairVLMed）')
                if not os.path.exists(data_setting.get("train_meta_path", "")):
                    candidates = []
                    for search_root in [".", "./data"]:
                        if os.path.exists(search_root):
                            for d in os.listdir(search_root):
                                p = os.path.join(search_root, d)
                                if os.path.isdir(p):
                                    if "fairvlmed" in d.lower() or "chexpert" in d.lower() or "ddi" in d.lower():
                                        candidates.append(p)

                    for cand in candidates:
                        if os.path.exists(os.path.join(cand, "train.csv")) or os.path.exists(os.path.join(cand, "train_processed.csv")):
                            for k, v in list(data_setting.items()):
                                if isinstance(v, str):
                                    if "Dataset" in v:
                                        data_setting[k] = os.path.join(cand, "Dataset")
                                    elif v.endswith(".csv"):
                                        data_setting[k] = os.path.join(cand, os.path.basename(v))
                                    elif "image" in k:
                                        data_setting[k] = cand
                            break


            if getattr(args, "train_meta_path", None):
                data_setting["train_meta_path"] = args.train_meta_path

            if args.sensitive_name != "All":
                specific_test_key = f"test_{str.lower(args.sensitive_name)}_meta_path"
                if specific_test_key in data_setting:
                    data_setting["test_meta_path"] = data_setting[specific_test_key]
            else:
                data_setting["test_meta_path"] = data_setting.get("test_sex_meta_path")


            args.data_setting = data_setting

            if args.pos_class is not None:
                args.data_setting["pos_class"] = args.pos_class

    except Exception as e:
        print(f"\n❌ Error loading dataset config for {args.dataset}: {e}\n")
        args.data_setting = None

    try:
        with open(f"configs/models/{args.model}.json", "r") as f:
            args.model_setting = json.load(f)
    except:
        args.model_setting = None

    return args


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
        "train", args.save_folder, "history.log", screen=True, tofile=True
    )
    logger.info("Using following arguments for training:")
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
    if args.usage not in ["clip-zs"]:
        if args.data_setting and "validation_meta_path" in args.data_setting:
            args.data_setting["image_val_path"] = args.data_setting.get("image_validation_path")
            _, val_loader, _ = get_dataset(args, split="validation")

    model = get_model(args).to(args.device)
    model = get_warpped_model(args, model).to(args.device)

    trainer = get_trainer(args, model, logger, test_dataloader, val_loader)

    # Multi-attribute evaluation setup
    if args.sensitive_name == "All":
        if args.dataset == "FairVLMed10k":
            attributes_to_test = ["Sex", "Age", "Race", "Language", "Ethnicity", "MaritalStatus"]
        elif args.dataset == "CXP":
            attributes_to_test = ["Sex", "Age"]
        elif args.dataset == "DDI":
            attributes_to_test = ["fitzpatrick"]
        else:
            attributes_to_test = ["Sex"]
        logger.info("====== INITIATING MULTI-ATTRIBUTE EVALUATION ======")
    else:
        attributes_to_test = [args.sensitive_name]

    if getattr(args, "eval_only", False):
        search_dir = args.exp_path
        matches = globlib.glob(os.path.join(search_dir, "**", "best_ckpt.pth"), recursive=True)
        if not matches:
            raise FileNotFoundError(f"No best_ckpt.pth found inside {search_dir}")
        ckpt_path = matches[0]
        logger.info(f"Found checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        logger.info(f"Loaded checkpoint from {ckpt_path}")
        evaluate_all_attributes(
            args, trainer, logger, attributes_to_test, save_prefix=f"{args.usage}_eval"
        )
        exit(0)

    # Zero-shot evaluation
    if args.usage == "clip-zs":
        logger.info("Evaluating CLIP Zero-Shot...")
        evaluate_all_attributes(
            args,
            trainer,
            logger,
            attributes_to_test,
            save_prefix="clip_zs_final",
        )
        exit(0)

    # Trainable adaptation protocols
    elif args.usage in [
        "clip-adapt",
        "lp",
        "lora-lp",
        "lora-text",
        "lora-zs",
        "full-ft",
        "full-ft-text",
        "full-ft-zs",
    ]:
        logger.info(f"{args.usage} training...")
        trainer.init_optimizers()
        trainer.train(train_dataloader, val_dataloader=val_loader)
        evaluate_all_attributes(
            args,
            trainer,
            logger,
            attributes_to_test,
            save_prefix=f"{args.usage}_final",
        )
        exit(0)

    else:
        raise NotImplementedError(f"Unsupported usage protocol: {args.usage}")