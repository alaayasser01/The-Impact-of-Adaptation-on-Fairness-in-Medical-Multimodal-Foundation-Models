#!/bin/bash
set +e

# ================= Configuration =================
DATASET="FairVLMed10k"                               # Set to "CXP" (or "CheXpert" if registered under that name)
MODEL="CLIP"                               # Set to "BiomedCLIP" (or "CLIP" if registered under that name)
BACKBONE="ViT-B/32"
EPOCHS=100                                   # 20 epochs is standard for CheXpert's ~156k train set
BATCH_SIZE=128
OUTPUT_DIR="/home/alaa/papersubmission/The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-Foundation-Models/FairVLMed10k_clip_results"
SEED=1
TEXT_SOURCE="template"

# 1. Include all 7 adaptation usages
USAGES=("clip-zs" "lp" "clip-adapt" "full-ft" "lora-lp" "lora-zs" "full-ft-zs")
# USAGES=( "lp" "clip-adapt" "full-ft" "lora-lp" "lora-zs" "full-ft-zs")

# USAGES=("clip-zs" "full-ft" "lora-lp" "lora-zs" "full-ft-zs")

mkdir -p "$OUTPUT_DIR"
SUMMARY_FILE="$OUTPUT_DIR/summary.csv"

# Write header only if file doesn't exist
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "usage,lr,seed,best_val_auc,best_epoch" > "$SUMMARY_FILE"
fi

for usage in "${USAGES[@]}"; do
    # 2. Dynamic LR handling based on adaptation type
    if [ "$usage" == "full-ft-zs" ] || [ "$usage" == "full-ft" ]; then
        lr="5e-3"
    else
        lr="1e-2"
    fi

    echo "========================================================="
    echo "Checking $usage | LR=$lr | Seed=$SEED"
    SAVE_DIR="$OUTPUT_DIR/${usage}_lr${lr}_seed${SEED}"

    # ---------- Skip if already completed ----------
    if [ -f "$SAVE_DIR/best_ckpt.pth" ] || [ "$usage" == "clip-zs" -a -f "$SAVE_DIR/history_*.log" ]; then
        LOG=$(ls -t "$SAVE_DIR"/history_*.log 2>/dev/null | head -1)
        if [ -n "$LOG" ]; then
            BEST_LINE=$(grep -i "best val AUC" "$LOG" | tail -1)
            if [ -z "$BEST_LINE" ]; then
                BEST_LINE=$(grep -i "Loaded best checkpoint from epoch" "$LOG" | tail -1)
            fi
            if [ -z "$BEST_LINE" ]; then
                BEST_LINE=$(grep -i "Test AUC" "$LOG" | tail -1)
            fi
            if [ -n "$BEST_LINE" ]; then
                BEST_AUC=$(echo "$BEST_LINE" | grep -oP '(\d+\.\d+)' | head -1)
                BEST_EPOCH=$(echo "$BEST_LINE" | grep -oP 'epoch \K\d+')
                BEST_EPOCH=${BEST_EPOCH:-0}
                echo "$usage,$lr,$SEED,$BEST_AUC,$BEST_EPOCH" >> "$SUMMARY_FILE"
                echo "  => Already completed (AUC: $BEST_AUC), skipping."
                continue
            fi
        fi
    fi

    echo "  => Running $usage on $DATASET..."
    python main.py \
        --task cls \
        --usage "$usage" \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --backbone "$BACKBONE" \
        --sensitive_name Sex \
        --method erm \
        --batch_size "$BATCH_SIZE" \
        --text_source "$TEXT_SOURCE" \
        --total_epochs "$EPOCHS" \
        --blr "$lr" \
        --random_seed "$SEED" \
        --exp_path "$SAVE_DIR"

    # ---------- Extract metrics from log after run ----------
    LOG=$(ls -t "$SAVE_DIR"/history_*.log 2>/dev/null | head -1)
    if [ -n "$LOG" ]; then
        BEST_LINE=$(grep -i "best val AUC" "$LOG" | tail -1)
        if [ -z "$BEST_LINE" ]; then
            BEST_LINE=$(grep -i "Loaded best checkpoint from epoch" "$LOG" | tail -1)
        fi
        if [ -z "$BEST_LINE" ]; then
            BEST_LINE=$(grep -i "Test AUC" "$LOG" | tail -1)
        fi
        if [ -n "$BEST_LINE" ]; then
            BEST_AUC=$(echo "$BEST_LINE" | grep -oP '(\d+\.\d+)' | head -1)
            BEST_EPOCH=$(echo "$BEST_LINE" | grep -oP 'epoch \K\d+')
            BEST_EPOCH=${BEST_EPOCH:-0}
            echo "$usage,$lr,$SEED,$BEST_AUC,$BEST_EPOCH" >> "$SUMMARY_FILE"
        else
            echo "$usage,$lr,$SEED,FAILED," >> "$SUMMARY_FILE"
        fi
    else
        echo "$usage,$lr,$SEED,FAILED," >> "$SUMMARY_FILE"
    fi
done

echo "========================================================="
echo "Sweep finished! Summary saved in: $SUMMARY_FILE"