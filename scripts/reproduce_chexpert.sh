#!/bin/bash
set +e

# ==============================================================================
# Reproduce CheXpert Generalization Experiments
# ==============================================================================

DATASET="CXP"
MODEL="CLIP"
BACKBONE="ViT-B/32"
EPOCHS=100
BATCH_SIZE=128
SEED=0
TEXT_SOURCE="template"
OUTPUT_DIR="./output/chexpert_clip_seed${SEED}"

USAGES=("clip-zs" "lp" "clip-adapt" "lora-lp" "lora-text" "full-ft" "full-ft-text")

mkdir -p "$OUTPUT_DIR"
SUMMARY_FILE="$OUTPUT_DIR/chexpert_summary.csv"

if [ ! -f "$SUMMARY_FILE" ]; then
    echo "usage,lr,seed,best_val_auc,best_epoch" > "$SUMMARY_FILE"
fi

for usage in "${USAGES[@]}"; do
    if [ "$usage" == "full-ft-text" ] || [ "$usage" == "full-ft" ] || [ "$usage" == "full-ft-zs" ]; then
        lr="5e-3"
    else
        lr="1e-2"
    fi

    echo "========================================================="
    echo "Running $usage | Dataset: $DATASET | LR=$lr | Seed=$SEED"
    echo "========================================================="

    SAVE_DIR="$OUTPUT_DIR/${usage}_lr${lr}_seed${SEED}"

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

echo "CheXpert experiments completed. Results logged to $OUTPUT_DIR"
