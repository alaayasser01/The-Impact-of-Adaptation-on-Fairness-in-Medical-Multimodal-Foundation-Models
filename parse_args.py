import argparse


def collect_args():
    parser = argparse.ArgumentParser(
        description="The Impact of Adaptation on Fairness in Medical Multimodal Foundation Models"
    )

    # Experiment settings
    parser.add_argument("--task", default="cls", choices=["cls"])
    parser.add_argument(
        "--usage",
        type=str,
        default="clip-zs",
        choices=[
            "clip-zs",
            "lp",
            "clip-adapt",
            "lora-lp",
            "lora-text",
            "lora-zs",
            "full-ft",
            "full-ft-text",
            "full-ft-zs",
        ],
        help="Adaptation protocol to evaluate or train",
    )
    parser.add_argument(
        "--dataset",
        default="FairVLMed10k",
        choices=["FairVLMed10k", "CXP", "DDI"],
        help="Dataset to use for experiment",
    )
    parser.add_argument(
        "--sensitive_name",
        default="All",
        choices=[
            "Sex",
            "Age",
            "Race",
            "Language",
            "Ethnicity",
            "MaritalStatus",
            "All",
            "fitzpatrick",
        ],
        help="Sensitive demographic attribute to evaluate fairness across",
    )
    parser.add_argument(
        "--binarize_attributes",
        action="store_true",
        help="Binarize demographic attributes (ablation study). Defaults to False (native cardinality as in paper).",
    )
    parser.add_argument(
        "--model",
        default="CLIP",
        choices=["CLIP", "BiomedCLIP", "CLIPftFairVLMed"],
        help="Vision-language foundation model",
    )
    parser.add_argument(
        "--backbone",
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16"],
        help="Vision encoder backbone architecture",
    )
    parser.add_argument(
        "--method",
        default="erm",
        choices=["erm"],
        help="Training algorithm (Empirical Risk Minimization)",
    )

    # Text prototypes & text sources for text-guided adaptation
    parser.add_argument(
        "--text_source",
        type=str,
        default="template",
        choices=["template", "gpt4_avg", "all_summaries", "clustered_summaries"],
        help="Source of text prototypes for CLIP-based methods",
    )
    parser.add_argument(
        "--proto_agg",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="Aggregation method for multi-prototype text sources",
    )

    # Paths and evaluation controls
    parser.add_argument("--train_meta_path", type=str, default="", help="Custom train metadata path")
    parser.add_argument("--pretrained_path", type=str, default="", help="Path to fine-tuned .pth file")
    parser.add_argument("--eval_only", action="store_true", help="Skip training, evaluate checkpoint only")
    parser.add_argument(
        "--save_test_per_epoch",
        action="store_true",
        help="Evaluate on test set after every epoch and save results",
    )
    parser.add_argument("--exp_path", type=str, default="./output", help="Directory for experiment outputs")
    parser.add_argument("--resume_path", type=str, default="", help="Checkpoint path to resume")

    # Optimization & training hyperparameters
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--optimizer", default="adamw", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--blr", type=float, default=1e-2, help="Base learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--fixed_lr", action="store_true", help="Keep learning rate fixed")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="Decay rate of learning rate")
    parser.add_argument("--lr_decay_period", type=float, default=10, help="Decay period of learning rate")
    parser.add_argument("--total_epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--early_stopping", type=int, default=1, help="Early stopping flag")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--no_cuda", dest="cuda", action="store_false", help="Disable CUDA")
    parser.add_argument("--no_cls_balance", dest="cls_balance", action="store_false")
    parser.add_argument("--context_length", default=77, type=int, help="CLIP context length")
    parser.add_argument(
        "--val_strategy",
        type=str,
        default="loss",
        choices=["loss", "worst_auc"],
        help="Strategy for selecting best validation model",
    )
    parser.add_argument("--log_freq", type=int, default=50, help="Logging frequency in steps")
    parser.add_argument("--pos_class", type=int, default=None)

    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args
