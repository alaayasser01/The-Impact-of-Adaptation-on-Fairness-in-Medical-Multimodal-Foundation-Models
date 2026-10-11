import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

import datasets


class ToTensor(object):
    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, img):
        img = (img - img.min()) / (img.max() - img.min())
        img = img.transpose(0, 3, 1, 2)
        return torch.FloatTensor(img)


def get_transform(args, split, augment=False):
    if args.dataset in ["CXP"]:
        mean = [0.5056, 0.5056, 0.5056]
        std = [0.252, 0.252, 0.252]
    else:
        # Standard ImageNet / CLIP normalization (used for FairVLMed10k and DDI)
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

    normalize = transforms.Normalize(mean=mean, std=std)

    if split == "train":
        if augment:
            transform = transforms.Compose(
                [
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation((-15, 15)),
                    transforms.RandomCrop((224, 224)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transform


def get_dataset(args, split):
    data_setting = args.data_setting

    transform = get_transform(args, split, augment=getattr(args, "augment", False))

    g = torch.Generator()
    g.manual_seed(args.random_seed)

    def seed_worker(worker_id):
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    meta_key = f"{split}_meta_path"
    if meta_key not in data_setting:
        raise KeyError(f"Metadata path '{meta_key}' not found in dataset config.")

    meta = pd.read_csv(data_setting[meta_key])

    image_path = data_setting.get(f"image_{split}_path", data_setting.get("image_train_path"))
    binarize = getattr(args, "binarize_attributes", False)
    try:
        data = dataset_name(meta, args.sensitive_name, transform, path_to_images=image_path, binarize=binarize)
    except TypeError:
        data = dataset_name(meta, args.sensitive_name, transform, path_to_images=image_path)

    if split == "train":
        data_loader = torch.utils.data.DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=(args.method != "resampling"),
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
        )

    if getattr(args, "cls_balance", False) and hasattr(data, "class_weights_sa"):
        data_loader.class_weights_sa = data.class_weights_sa
    if getattr(args, "cls_balance", False) and hasattr(data, "class_weights_y"):
        data_loader.class_weights_y = data.class_weights_y

    return data, data_loader, meta