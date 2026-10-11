# The Impact of Adaptation on Fairness in Medical Multimodal Foundation Models

This repository contains the official PyTorch implementation, reproduction scripts, and fairness evaluation benchmarks for the research study:

> **The Impact of Adaptation on Fairness in Medical Multimodal Foundation Models**  
> *Master's Thesis & Research Paper*

---

## 📌 Abstract

Vision-language foundation models (e.g., OpenAI CLIP and BiomedCLIP) offer substantial diagnostic promise in medical imaging. However, safely integrating these models into clinical workflows demands rigorous evaluation of their fairness across diverse patient populations.

In this work, we present a comprehensive empirical investigation into how downstream **adaptation protocols**—ranging from zero-shot inference to parameter-efficient tuning and full model fine-tuning—influence demographic performance parity. We evaluate **7 adaptation paradigms** across **6 sensitive demographic attributes** (Sex, Age, Race, Language, Ethnicity, and Marital Status) on **FairVLMed10k** (ophthalmology/glaucoma), and demonstrate generalization on **CheXpert** (chest radiography) and the **Diverse Dermatology Images (DDI)** dataset.

### Key Insights
1. **Capacity vs. Fairness Trade-off**: Increasing adaptation capacity (e.g., transitioning from Linear Probing or LoRA to Full Fine-Tuning) consistently improves average diagnostic performance (AUC), but frequently amplifies fairness gaps across vulnerable subgroups.
2. **Text-Anchored Regularization**: Text-guided adaptation protocols (`lora-text`, `full-ft-text`) preserve multimodal semantic alignment better than purely visual heads, serving as an implicit regularizer against catastrophic group disparity.
3. **Severe Intersectional Vulnerabilities**: Marginal (single-attribute) fairness evaluations mask compounded performance degradation when evaluating multiple intersecting demographic axes (e.g., Non-English speaking, senior female patients).
4. **Domain Pretraining vs. Generic Pretraining**: BiomedCLIP exhibits different baseline inductive biases than OpenAI CLIP, affecting how disparate error rates manifest post-adaptation.

---

## 🔬 Adaptation Protocols & Architectural Taxonomy

Downstream adaptation of multimodal foundation models can be structured along two fundamental axes: **how predictions are formed** (unimodal classifier vs. vision-language cosine alignment) and **which components are updated** (frozen, parameter-efficient LoRA, or full fine-tuning).

<p align="center">
  <img src="./readme-figs/main.png" alt="Taxonomy of Adaptation Protocols" width="92%">
  <br>
  <em><b>Figure 1: Taxonomy of Adaptation Protocols in Medical Multimodal Foundation Models.</b><br>
  (Left, Blue): Unimodal adaptation discarding the text encoder and training a linear classification head (<code>lp</code>, <code>lora-lp</code>, <code>full-ft</code>).<br>
  (Right, Green): Multimodal adaptation retaining vision-language cosine similarity against prompt anchors or learned prototypes (<code>clip-adapt</code>, <code>lora-text</code>, <code>full-ft-text</code>).</em>
</p>

We evaluate 7 adaptation protocols ranked by their trainable capacity and multimodal integration:

| Protocol | `--usage` Flag | Trainable Parameters | Prediction Paradigm | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Zero-Shot** | `clip-zs` | None (0%) | Cosine Similarity | Zero-shot inference against fixed prompt text prototypes |
| **Linear Probing** | `lp` | Classifier Head only | Linear Classifier | Freeze vision backbone; train linear classification head |
| **CLIP-Adapt** | `clip-adapt` | Residual Bottleneck Head | Cosine Similarity | Two-layer residual adapter on frozen vision embeddings |
| **LoRA (Linear Probe)** | `lora-lp` | LoRA weights + Head | Linear Classifier | Low-Rank Adaptation on attention layers with linear head |
| **LoRA (Text-Guided)** | `lora-text` | LoRA weights only | Cosine Similarity | LoRA on vision backbone aligned with fixed text embeddings |
| **Full Fine-Tuning** | `full-ft` | Entire Backbone + Head | Linear Classifier | End-to-end retraining of vision backbone with linear head |
| **Full-FT (Text-Guided)** | `full-ft-text` | Entire Vision Backbone | Cosine Similarity | End-to-end retraining aligned with text anchor embeddings |

*(Note: `lora-zs` and `full-ft-zs` are retained as aliases for backward compatibility).*

---

## 📊 Datasets & Benchmarks

| Dataset | Modality | Target Task | Sensitive Attributes Evaluated |
| :--- | :--- | :--- | :--- |
| **FairVLMed10k** (Primary) | SLO Fundus Imaging | Glaucoma Detection | Sex, Age, Race, Language, Ethnicity, Marital Status |
| **CheXpert** (Generalization) | Chest X-Ray | Diagnostic Abnormality | Sex, Age |
| **DDI** (Generalization) | Dermoscopy | Skin Lesion Malignancy | Fitzpatrick Skin Tone (I–VI) |

---

## 📁 Repository Structure

```
.
├── configs/
│   ├── datasets/               # Dataset paths and label template configurations
│   │   ├── FairVLMed10k.json
│   │   ├── CXP.json
│   │   └── DDI.json
│   └── models/                 # Model parameters (e.g. LoRA target projection layers)
│       ├── CLIP.json
│       └── BiomedCLIP.json
├── datasets/                   # Modular dataset loaders with demographic subgroup splits
│   ├── base.py
│   ├── FairVLMed.py
│   ├── CXP.py
│   ├── DDI.py
│   └── utils.py
├── models/                     # Foundation model implementations
│   ├── clip.py                 # OpenAI CLIP ViT-B/32
│   ├── biomed_clip.py          # Microsoft BiomedCLIP
│   └── utils.py
├── wrappers/                   # 7 adaptation protocol wrappers
│   ├── clip.py                 # Zero-shot & CLIP-Adapt
│   ├── linear_probe.py         # Linear probing
│   ├── lora.py                 # LoRA-LP and LoRA-Text
│   ├── full_ft.py              # Full fine-tuning & text-guided fine-tuning
│   └── utils.py
├── trainers/                   # Training loop with validation & early stopping
│   ├── cls.py                  # Classification trainer with multi-attribute logging
│   └── utils.py
├── scripts/                    # Reproduction bash scripts for each experiment
│   ├── reproduce_fairvlmed.sh  # Main benchmark on FairVLMed10k
│   ├── reproduce_biomedclip.sh # BiomedCLIP experiments
│   ├── reproduce_chexpert.sh   # CheXpert generalization
│   ├── reproduce_ddi.sh        # DDI generalization
│   └── eval_only.sh            # Checkpoint evaluation script
├── analysis/                   # Paper figure and metric reproduction scripts
│   ├── intersectional_analysis.py  # Generates intersectional LaTeX summary table
│   ├── training_dynamics.py        # Computes convergence plateaus & fairness stability
│   └── text_source_ablation.py     # Generates text prototype similarity heatmaps
├── figures/                    # Publication figures
│   ├── paper/                  # Figures appearing in paper / supplementary
│   └── thesis/                 # Extended demographic and ablation figures
├── results/                    # Curated experiment summary CSVs & evaluation metrics
├── pre-processing/             # Dataset preprocessing notebooks
│   └── classification/
│       ├── FairVLMed-10k.ipynb # Generates official train/val/test splits
│       ├── CXP.ipynb
│       └── DDI.ipynb
├── readme-figs/                # README visual assets
│   └── main.png
├── main.py                     # Unified entry point for training and evaluation
├── parse_args.py               # Argument definitions and parser
└── prompts.json                # Standard and augmented clinical prompt templates
```

---

## ⚙️ Environment Setup

We recommend creating a fresh Conda environment with Python 3.10:

```bash
# 1. Create and activate environment
conda create -n fairmed python=3.10 -y
conda activate fairmed

# 2. Install PyTorch (select the appropriate CUDA version for your hardware)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install core dependencies
pip install open_clip_torch ftfy regex scikit-learn pandas matplotlib seaborn
```

---

## 🗂️ Data Preparation

1. Download the respective datasets:
   - **FairVLMed10k**: [Harvard FairVLMed Dataset](https://ophai.hms.harvard.edu/datasets/harvard-fairvlmed10k/)
   - **CheXpert**: [CheXpert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert/)
   - **DDI**: [Diverse Dermatology Images](https://ddi-dataset.github.io/)
2. Run the corresponding preprocessing notebook in `pre-processing/classification/` to format metadata and demographic groups.
3. Update the dataset paths in `configs/datasets/`:
   - `configs/datasets/FairVLMed10k.json`
   - `configs/datasets/CXP.json`
   - `configs/datasets/DDI.json`

---

## 🚀 Quick Start & Reproduction

### Reproducing Benchmark Experiments
To run all 7 adaptation protocols on FairVLMed10k with CLIP:
```bash
bash scripts/reproduce_fairvlmed.sh
```

To run BiomedCLIP experiments:
```bash
bash scripts/reproduce_biomedclip.sh
```

To run CheXpert and DDI generalization experiments:
```bash
bash scripts/reproduce_chexpert.sh
bash scripts/reproduce_ddi.sh
```

### Running an Individual Experiment
You can execute any single model, protocol, and dataset directly via `main.py`:

```bash
# Example: Train LoRA with text anchors (lora-text) on FairVLMed10k
python main.py \
    --task cls \
    --usage lora-text \
    --dataset FairVLMed10k \
    --model CLIP \
    --backbone "ViT-B/32" \
    --sensitive_name All \
    --method erm \
    --batch_size 128 \
    --total_epochs 100 \
    --blr 1e-2 \
    --random_seed 0 \
    --exp_path "./output/my_experiment"
```

```bash
# Example: Zero-shot evaluation with OpenAI CLIP
python main.py \
    --task cls \
    --usage clip-zs \
    --dataset FairVLMed10k \
    --model CLIP \
    --backbone "ViT-B/32" \
    --sensitive_name All \
    --batch_size 128 \
    --exp_path "./output/zero_shot_eval"
```

### Evaluating Existing Checkpoints
```bash
python main.py \
    --task cls \
    --usage full-ft-text \
    --dataset FairVLMed10k \
    --model CLIP \
    --backbone "ViT-B/32" \
    --sensitive_name All \
    --exp_path "./output/my_experiment" \
    --eval_only
```

---

## 📈 Downstream Analysis Scripts

Scripts in `analysis/` reproduce figures and summary tables from the paper:

```bash
# 1. Generate intersectional LaTeX table from evaluation outputs
python analysis/intersectional_analysis.py

# 2. Evaluate training dynamics and fairness stability near convergence
python analysis/training_dynamics.py

# 3. Generate clinical text prototype cosine similarity heatmaps
python analysis/text_source_ablation.py
```

Generated outputs are stored in `results/` and `figures/paper/`.

---

## 📑 Citation

If you find this research or codebase useful in your work, please cite:

```bibtex
@article{yasser2026impact,
  title={The Impact of Adaptation on Fairness in Medical Multimodal Foundation Models},
  author={Yasser, Alaa},
  year={2026}
}
```

---

## 🙏 Acknowledgements

This research builds upon the architectural framework of the open-source benchmark [FairMedFM](https://github.com/FairMedFM/FairMedFM). We express our appreciation to the authors and contributors of FairMedFM.

---

## 📄 License

This project is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License - see the [LICENSE](./LICENSE) file for details.