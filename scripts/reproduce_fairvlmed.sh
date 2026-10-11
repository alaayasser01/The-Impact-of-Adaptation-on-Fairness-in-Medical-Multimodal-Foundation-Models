#!/bin/bash
set +e

# ==============================================================================
# Reproduce FairVLMed10k Experiments (CLIP ViT-B/32, 7 Adaptation Protocols)
# ==============================================================================

DATASET="FairVLMed10k"
MODEL="CLIP"
BACKBONE="ViT-B/32"
EPOCHS=100
BATCH_SIZE=128
SEED=0
TEXT_SOURCE="template"
OUTPUT_DIR="./output/fairvlmed_clip_seed${SEED}"

# All 7 adaptation protocols evaluated in the paper
USAGES=("clip-zs" "lp" "clip-adapt" "lora-lp" "lora-text" "full-ft" "full-ft-text")

mkdir -p "$OUTPUT_DIR"
SUMMARY_FILE="$OUTPUT_DIR/summary.csv"

if [ ! -f "$SUMMARY_FILE" ]; then
    echo "usage,lr,seed,best_val_auc,best_epoch" > "$SUMMARY_FILE"
fi

for usage in "${USAGES[@]}"; do
    # Dynamic learning rate selected per protocol:
    # Full fine-tuning uses 5e-3, while parameter-efficient and linear probe use 1e-2
    if [ "$usage" == "full-ft-text" ] || [ "$usage" == "full-ft" ] || [ "$usage" == "full-ft-zs" ]; then
        lr="5e-3"
    else
        lr="1e-2"
    fi

    echo "========================================================="
    echo "Running $usage | Model: $MODEL | LR=$lr | Seed=$SEED"
    echo "========================================================="

    SAVE_DIR="$OUTPUT_DIR/${usage}_lr${lr}_seed${SEED}"

    # Skip if already completed
    if [ -f "$SAVE_DIR/best_ckpt.pth" ] || [ "$usage" == "clip-zs" -a -f "$SAVE_DIR/history_*.log" ]; then
        echo "  => Already completed, skipping."
        continue
    fi

    if [ "$usage" == "clip-zs" ]; then
        python main.py \
            --task cls \
            --usage "$usage" \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --backbone "$BACKBONE" \
            --sensitive_name All \
            --batch_size $BATCH_SIZE \
            --text_source "$TEXT_SOURCE" \
            --exp_path "$OUTPUT_DIR" \
            --random_seed "$SEED"
    else
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
            --total_epochs $EPOCHS \
            --blr "$lr" \
            --random_seed "$SEED" \
            --exp_path "$OUTPUT_DIR"
    fi
done

echo "Experiments finished. Results logged to $OUTPUT_DIR"
