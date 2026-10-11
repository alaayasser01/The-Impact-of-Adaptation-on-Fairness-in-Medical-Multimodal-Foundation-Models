#!/usr/bin/env python3
"""
Remote Verification & Reproduction Benchmark Script
===================================================
Run this script on your remote GPU server to verify that:
1. The hardware, CUDA, and Python environment are properly configured.
2. Dataset directories and preprocessed CSV splits are accessible.
3. Fast Zero-Shot evaluation reproduces the published Paper & Thesis numbers.
4. (Optional) Checkpoints or training dry-runs execute without errors.

Usage:
------
# 1. Full Pre-flight & Zero-Shot Verification (Recommended):
python scripts/verify_reproduction.py

# 2. Check environment and datasets only (No model run):
python scripts/verify_reproduction.py --mode check_env

# 3. Verify on DDI dataset:
python scripts/verify_reproduction.py --dataset DDI

# 4. Verify on CheXpert:
python scripts/verify_reproduction.py --dataset CXP

# 5. Evaluate an existing trained checkpoint:
python scripts/verify_reproduction.py --mode eval_ckpt --ckpt_path ./output/fairvlmed_clip_seed0/lora-text_lr1e-2_seed0/best_ckpt.pth --usage lora-text
"""

import argparse
import glob
import json
import os
import pickle
import subprocess
import sys

# ==============================================================================
# TARGET BENCHMARKS FROM PAPER & THESIS
# ==============================================================================
PAPER_TARGETS = {
    "FairVLMed10k": {
        "clip-zs": {
            "overall_auc": 0.6028,
            "mean_eod": 0.9990,
            "Sex_eod": 0.9989,
            "Age_eod": 0.9974,
            "Race_eod": 0.9993,
            "Language_eod": 0.9995,
            "Ethnicity_eod": 0.9995,
            "MaritalStatus_eod": 0.9991,
        },
        "lp": {
            "overall_auc": 0.7247,
            "mean_eod": 0.8627,
            "Age_eod": 0.7090,
            "Sex_eod": 0.9673,
        },
        "clip-adapt": {
            "overall_auc": 0.7292,
            "mean_eod": 0.8607,
            "Age_eod": 0.6940,
            "Sex_eod": 0.9645,
        },
        "lora-lp": {
            "overall_auc": 0.8043,
            "mean_eod": 0.8320,
            "Age_eod": 0.6852,
            "Sex_eod": 0.9886,
        },
        "lora-text": {
            "overall_auc": 0.7996,
            "mean_eod": 0.8165,
            "Age_eod": 0.6703,
            "Language_eod": 0.6583,
        },
        "full-ft": {
            "overall_auc": 0.7954,
            "mean_eod": 0.8137,
            "Age_eod": 0.6456,
            "Sex_eod": 0.9409,
        },
        "full-ft-text": {
            "overall_auc": 0.8006,
            "mean_eod": 0.8609,
            "Age_eod": 0.7542,
            "Language_eod": 0.7355,
        },
    },
    "DDI": {
        "clip-zs": {
            "overall_auc": 0.5060,
            "mean_eod": 0.9017,
            "fitzpatrick_eod": 0.9017,
        },
        "lora-text": {
            "overall_auc": 0.7216,
            "mean_eod": 0.8973,
            "fitzpatrick_eod": 0.8973,
        },
        "full-ft": {
            "overall_auc": 0.6751,
            "mean_eod": 0.9344,
            "fitzpatrick_eod": 0.9344,
        },
    },
    "CXP": {
        "clip-zs": {
            "overall_auc": 0.5120,
            "mean_eod": 0.9750,
        }
    },
}


def print_header(title):
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)


def check_environment():
    """Verify GPU, CUDA, PyTorch, and required packages."""
    print_header("PHASE 1: Hardware & Environment Pre-flight Check")
    all_ok = True

    # 1. Python version
    py_ver = sys.version.split()[0]
    print(f"[*] Python Version: {py_ver}")

    # 2. PyTorch & CUDA
    try:
        import torch

        print(f"[*] PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"[*] CUDA Available: {cuda_avail}")
        if cuda_avail:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"    - GPU Count: {device_count}")
            print(f"    - Primary Device: {device_name} ({vram_gb:.1f} GB VRAM)")
        else:
            print("    [!] WARNING: CUDA is not available. Runs will default to CPU (very slow).")
            all_ok = False
    except ImportError:
        print("[!] ERROR: PyTorch is not installed. Run: pip install torch torchvision")
        all_ok = False

    # 3. Key Dependencies
    packages = [
        ("torchvision", "torchvision"),
        ("clip (OpenAI)", "clip"),
        ("open_clip_torch", "open_clip"),
        ("peft (HuggingFace)", "peft"),
        ("timm", "timm"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("numpy", "numpy"),
    ]

    print("\n[*] Checking Required Packages:")
    for label, pkg in packages:
        try:
            __import__(pkg)
            print(f"    [OK] {label:<22} is installed.")
        except ImportError:
            print(f"    [FAIL] {label:<22} is MISSING. Install via requirements.txt")
            all_ok = False

    return all_ok


def check_dataset_paths(dataset_name):
    """Verify that the dataset configuration and files exist."""
    print_header(f"PHASE 2: Dataset Configuration Check ({dataset_name})")
    config_file = f"configs/datasets/{dataset_name}.json"
    if not os.path.exists(config_file):
        print(f"[!] ERROR: Config file {config_file} not found!")
        return False

    with open(config_file, "r") as f:
        cfg = json.load(f)

    print(f"[*] Dataset Config: {config_file}")
    img_dir = cfg.get("image_train_path") or cfg.get("image_test_path")
    print(f"[*] Image Root: {img_dir}")
    if img_dir and not os.path.exists(img_dir):
        print(f"    [!] Note: Directory '{img_dir}' does not exist locally.")
        print("        Ensure your datasets are downloaded or symlinked to ./data/")

    # Check metadata CSVs
    meta_keys = [k for k in cfg.keys() if "meta_path" in k]
    found_any = False
    for k in meta_keys:
        p = cfg[k]
        exists = os.path.exists(p)
        status = "[OK] Found" if exists else "[MISSING]"
        print(f"    - {k:<28}: {status} -> {p}")
        if exists:
            found_any = True

    if not found_any:
        print(f"\n[!] WARNING: No metadata CSV files were found for {dataset_name}.")
        print("    Did you run the pre-processing script, e.g.:")
        print(f"    jupyter nbconvert --to notebook --execute pre-processing/classification/{dataset_name}.ipynb")
        return False

    return True


def run_zero_shot_verification(dataset="FairVLMed10k", model="CLIP", batch_size=128):
    """Execute Zero-Shot evaluation and compare results to published paper targets."""
    print_header(f"PHASE 3: Zero-Shot Reproduction Verification ({dataset} - {model})")
    exp_dir = f"./output/verify_reproduction_{dataset.lower()}_{model.lower()}"
    os.makedirs(exp_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "main.py",
        "--task",
        "cls",
        "--usage",
        "clip-zs",
        "--dataset",
        dataset,
        "--model",
        model,
        "--backbone",
        "ViT-B/32",
        "--sensitive_name",
        "All",
        "--batch_size",
        str(batch_size),
        "--text_source",
        "template",
        "--exp_path",
        exp_dir,
        "--random_seed",
        "0",
    ]

    print(f"[*] Executing Command:\n    {' '.join(cmd)}\n")
    ret = subprocess.run(cmd)

    if ret.returncode != 0:
        print("\n[!] ERROR: Zero-shot evaluation failed with non-zero exit code.")
        return False

    print("\n[*] Zero-shot run completed. Gathering results from output folder...")
    return parse_and_compare_results(exp_dir, dataset=dataset, protocol="clip-zs")


def parse_and_compare_results(exp_dir, dataset="FairVLMed10k", protocol="clip-zs"):
    """Parse metrics.pkl files from output directory and print comparison table."""
    targets = PAPER_TARGETS.get(dataset, {}).get(protocol, {})
    if not targets:
        print(f"[!] No reference paper targets configured for {dataset} / {protocol}.")
        return True

    # Search for metrics.pkl files
    metric_files = glob.glob(os.path.join(exp_dir, "**", "metrics.pkl"), recursive=True)
    if not metric_files:
        print(f"[!] No metrics.pkl files found under {exp_dir}!")
        return False

    extracted = {}
    eod_values = []

    for fpath in metric_files:
        attr_dir = os.path.basename(os.path.dirname(fpath))
        # e.g. clip_zs_final_Sex or clip_zs_final_Age
        attr = attr_dir.split("_")[-1]
        try:
            with open(fpath, "rb") as f:
                data = pickle.load(f)
                overall = data.get("overall", {})
                subgroup = data.get("subgroup", {})

                if "overall_auc" not in extracted and "auc" in overall:
                    extracted["overall_auc"] = overall["auc"]

                if "eod" in subgroup:
                    extracted[f"{attr}_eod"] = subgroup["eod"]
                    eod_values.append(subgroup["eod"])
        except Exception as e:
            print(f"    [!] Warning: Could not read {fpath}: {e}")

    if eod_values and "mean_eod" not in extracted:
        extracted["mean_eod"] = sum(eod_values) / len(eod_values)

    # Print comparison table
    print("\n" + "=" * 78)
    print(f" VERIFICATION RESULTS vs. PAPER TARGETS: {dataset} ({protocol})")
    print("=" * 78)
    print(f"{'Metric':<22} | {'Your Run':<12} | {'Paper Target':<14} | {'Diff':<10} | {'Status'}")
    print("-" * 78)

    all_passed = True
    tolerance = 0.015  # 1.5% margin for cross-platform / PyTorch version floating point differences

    for metric, paper_val in targets.items():
        if metric in extracted:
            run_val = extracted[metric]
            diff = run_val - paper_val
            match = abs(diff) <= tolerance
            status_str = "[PASS MATCH]" if match else "[DISCREPANCY]"
            if not match:
                all_passed = False
            print(
                f"{metric:<22} | {run_val:<12.4f} | {paper_val:<14.4f} | {diff:<+10.4f} | {status_str}"
            )
        else:
            print(f"{metric:<22} | {'N/A':<12} | {paper_val:<14.4f} | {'N/A':<10} | [NOT EVALUATED]")

    print("=" * 78)
    if all_passed:
        print("\n>>> SUCCESS: All verified metrics closely match the published Paper & Thesis! <<<")
    else:
        print("\n>>> NOTE: Some metrics differed beyond tolerance. Verify dataset splits/transforms. <<<")

    return all_passed


def evaluate_checkpoint(ckpt_path, dataset="FairVLMed10k", usage="lora-text", batch_size=128):
    """Evaluate an existing best_ckpt.pth across sensitive groups."""
    print_header(f"Evaluating Checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"[!] ERROR: Checkpoint file {ckpt_path} does not exist!")
        return False

    exp_dir = os.path.dirname(ckpt_path)
    cmd = [
        sys.executable,
        "main.py",
        "--task",
        "cls",
        "--usage",
        usage,
        "--dataset",
        dataset,
        "--model",
        "CLIP",
        "--backbone",
        "ViT-B/32",
        "--sensitive_name",
        "All",
        "--batch_size",
        str(batch_size),
        "--text_source",
        "template",
        "--exp_path",
        exp_dir,
        "--eval_only",
    ]

    print(f"[*] Executing Command:\n    {' '.join(cmd)}\n")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print("[!] Evaluation failed.")
        return False

    return parse_and_compare_results(exp_dir, dataset=dataset, protocol=usage)


def main():
    parser = argparse.ArgumentParser(
        description="Verify Reproduction of Paper/Thesis Results on Remote Machine"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "check_env", "zeroshot", "eval_ckpt"],
        default="all",
        help="Verification mode: all (env + zeroshot), check_env, zeroshot, eval_ckpt",
    )
    parser.add_argument(
        "--dataset",
        choices=["FairVLMed10k", "DDI", "CXP"],
        default="FairVLMed10k",
        help="Dataset to test",
    )
    parser.add_argument(
        "--model",
        choices=["CLIP", "BiomedCLIP"],
        default="CLIP",
        help="Foundation model",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to best_ckpt.pth when mode=eval_ckpt"
    )
    parser.add_argument(
        "--usage",
        type=str,
        default="lora-text",
        help="Protocol name when evaluating checkpoint (e.g. lp, lora-text, full-ft)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("      MEDICAL MULTIMODAL FOUNDATION MODELS - REPRODUCTION VERIFIER")
    print("=" * 78)

    # 1. Environment Check
    env_ok = check_environment()
    data_ok = check_dataset_paths(args.dataset)

    if args.mode == "check_env":
        print("\nPre-flight check completed.")
        return

    if not data_ok:
        print("\n[!] Please verify that preprocessed CSV files exist before running evaluation.")
        print("    Exiting.")
        return

    # 2. Execution Modes
    if args.mode in ["all", "zeroshot"]:
        run_zero_shot_verification(
            dataset=args.dataset, model=args.model, batch_size=args.batch_size
        )

    elif args.mode == "eval_ckpt":
        if not args.ckpt_path:
            print("[!] Please provide --ckpt_path when using mode=eval_ckpt")
            return
        evaluate_checkpoint(
            ckpt_path=args.ckpt_path,
            dataset=args.dataset,
            usage=args.usage,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
