#!/bin/bash
set +e

# ==============================================================================
# Evaluate Pretrained Checkpoints Across Sensitive Groups
# ==============================================================================

DATASET="FairVLMed10k"
MODEL="CLIP"
BACKBONE="ViT-B/32"
BATCH_SIZE=128
TEXT_SOURCE="template"
OUTPUT_DIR="./output/fairvlmed_clip_seed0"

USAGES=("lp" "clip-adapt" "lora-lp" "lora-text" "full-ft" "full-ft-text")
SEEDS=(0)

for SEED in "${SEEDS[@]}"; do
    for usage in "${USAGES[@]}"; do
        if [ "$usage" == "full-ft-text" ] || [ "$usage" == "full-ft" ] || [ "$usage" == "full-ft-zs" ]; then
            lr="5e-3"
        else
            lr="1e-2"
        fi

        SAVE_DIR="$OUTPUT_DIR/${usage}_lr${lr}_seed${SEED}"

        echo "========================================================="
        echo "Evaluating | Usage: $usage | Seed: $SEED"
        echo "========================================================="

        python main.py \
            --task cls \
            --usage "$usage" \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --backbone "$BACKBONE" \
            --sensitive_name All \
            --method erm \
            --batch_size $BATCH_SIZE \
            --text_source "$TEXT_SOURCE" \
            --blr "$lr" \
            --random_seed "$SEED" \
            --exp_path "$SAVE_DIR" \
            --eval_only
    done
done

echo "All evaluations completed!"
