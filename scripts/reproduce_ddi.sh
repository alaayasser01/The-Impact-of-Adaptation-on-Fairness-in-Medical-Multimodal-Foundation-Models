#!/bin/bash
set +e

# ==============================================================================
# Reproduce DDI Generalization Experiments (Dermatology, Fitzpatrick skin tone)
# ==============================================================================

DATASET="DDI"
MODEL="CLIP"
BACKBONE="ViT-B/32"
EPOCHS=100
BATCH_SIZE=32
TEXT_SOURCE="template"
SEED=0
OUTPUT_DIR="./output/ddi_clip_seed${SEED}"

mkdir -p "$OUTPUT_DIR"

# Protocols evaluated on DDI
USAGES=("clip-zs" "full-ft" "lora-text" "full-ft-text")

declare -A BEST_LR
BEST_LR[clip-zs]="1e-2"
BEST_LR[full-ft]="1e-2"
BEST_LR[lora-text]="1e-2"
BEST_LR[full-ft-text]="5e-3"

for usage in "${USAGES[@]}"; do
    lr="${BEST_LR[$usage]}"
    SAVE_DIR="$OUTPUT_DIR/${usage}"

    echo "========================================================="
    echo "Running DDI Experiment | Protocol: $usage | LR=$lr | Seed=$SEED"
    echo "========================================================="

    if [ "$usage" == "clip-zs" ]; then
        python main.py \
            --task cls \
            --usage "$usage" \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --backbone "$BACKBONE" \
            --sensitive_name fitzpatrick \
            --method erm \
            --batch_size $BATCH_SIZE \
            --text_source "$TEXT_SOURCE" \
            --exp_path "$SAVE_DIR" \
            --random_seed "$SEED"
    else
        python main.py \
            --task cls \
            --usage "$usage" \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --backbone "$BACKBONE" \
            --sensitive_name fitzpatrick \
            --method erm \
            --batch_size $BATCH_SIZE \
            --text_source "$TEXT_SOURCE" \
            --total_epochs $EPOCHS \
            --blr "$lr" \
            --random_seed "$SEED" \
            --exp_path "$SAVE_DIR"
    fi
done

echo "DDI experiments completed. Results saved to $OUTPUT_DIR"
