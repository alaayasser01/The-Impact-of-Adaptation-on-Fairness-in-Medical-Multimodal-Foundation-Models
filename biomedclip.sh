set +e

DATASET="FairVLMed10k"
MODEL="BiomedCLIP"
BACKBONE="ViT-B/32"
EPOCHS=100
BATCH_SIZE=128
OUTPUT_DIR="/home/alaa/papersubmission/The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-Foundation-Models/biomedclip_results"
SEED=1
TEXT_SOURCE="template"

# 1. Include all 7 usages
USAGES=("clip-zs" "lp" "clip-adapt" "full-ft" "lora-lp" "lora-zs" "full-ft-zs")

mkdir -p "$OUTPUT_DIR"
SUMMARY_FILE="$OUTPUT_DIR/biomedclip.csv"

# Write header only if file doesn't exist
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "usage,lr,seed,best_val_auc,best_epoch" > "$SUMMARY_FILE"
fi

for usage in "${USAGES[@]}"; do
    # 2. Dynamic LR handling based on your rules
    if [ "$usage" == "full-ft-zs" ]; then
        lr="5e-3"
    else
        lr="1e-2"
    fi

    echo "========================================================="
    echo "Checking $usage | LR=$lr | Seed=$SEED"
    SAVE_DIR="$OUTPUT_DIR/${usage}_lr${lr}_seed${SEED}"

    # ---------- Skip if already completed ----------
    if [ -f "$SAVE_DIR/best_ckpt.pth" ]; then
        LOG=$(ls -t "$SAVE_DIR"/history_*.log 2>/dev/null | head -1)
        if [ -n "$LOG" ]; then
            BEST_LINE=$(grep "best val AUC" "$LOG" | tail -1)
            if [ -z "$BEST_LINE" ]; then
                BEST_LINE=$(grep "Loaded best checkpoint from epoch" "$LOG" | tail -1)
            fi
            if [ -n "$BEST_LINE" ]; then
                BEST_AUC=$(echo "$BEST_LINE" | grep -oP '(\d+\.\d+)(?=.*Loaded best checkpoint|best val AUC|$)' | head -1)
                BEST_EPOCH=$(echo "$BEST_LINE" | grep -oP 'epoch \K\d+')
                echo "$usage,$lr,$SEED,$BEST_AUC,$BEST_EPOCH" >> "$SUMMARY_FILE"
                echo "  => Already completed, skipping."
                continue
            fi
        fi
    fi

    echo "  => Running..."
    python main.py \
        --task cls \
        --usage "$usage" \
        --dataset $DATASET \
        --model $MODEL \
        --backbone "$BACKBONE" \
        --sensitive_name All \
        --method erm \
        --batch_size $BATCH_SIZE \
        --text_source "$TEXT_SOURCE" \
        --total_epochs $EPOCHS \
        --blr $lr \
        --random_seed "$SEED" \
        --exp_path "$SAVE_DIR"

    # ---------- After run, extract best validation AUC from log ----------
    LOG=$(ls -t "$SAVE_DIR"/history_*.log 2>/dev/null | head -1)
    if [ -n "$LOG" ]; then
        BEST_LINE=$(grep "best val AUC" "$LOG" | tail -1)
        if [ -z "$BEST_LINE" ]; then
            BEST_LINE=$(grep "Loaded best checkpoint from epoch" "$LOG" | tail -1)
        fi
        if [ -n "$BEST_LINE" ]; then
            BEST_AUC=$(echo "$BEST_LINE" | grep -oP '(\d+\.\d+)(?=.*Loaded best checkpoint|best val AUC|$)' | head -1)
            BEST_EPOCH=$(echo "$BEST_LINE" | grep -oP 'epoch \K\d+')
            echo "$usage,$lr,$SEED,$BEST_AUC,$BEST_EPOCH" >> "$SUMMARY_FILE"
        else
            echo "$usage,$lr,$SEED,FAILED," >> "$SUMMARY_FILE"
        fi
    else
        echo "$usage,$lr,$SEED,FAILED," >> "$SUMMARY_FILE"
    fi
done

echo "Sweep finished. Summary in $SUMMARY_FILE"