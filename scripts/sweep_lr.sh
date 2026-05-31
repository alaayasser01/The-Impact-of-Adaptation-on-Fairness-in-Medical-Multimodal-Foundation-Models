# #!/bin/bash
# set +e

# DATASET="FairVLMed10k"
# MODEL="CLIP"
# BACKBONE="ViT-B/32"
# EPOCHS=100               # enough for early stopping to trigger
# BATCH_SIZE=128
# OUTPUT_DIR="./lr_sweep"
# SEED=0
# TEXT_SOURCE="template"

# # All trainable usages
# USAGES=("lp" "clip-adapt" "full-ft" "full-ft-zs" "lora-lp" "lora-zs")

# # SINGLE learning rate list for ALL usages (to have comparable plots)
# LR_LIST=(1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2)

# mkdir -p "$OUTPUT_DIR"
# SUMMARY_FILE="$OUTPUT_DIR/sweep_summary.csv"
# echo "usage,lr,seed,best_val_auc,best_epoch" > "$SUMMARY_FILE"

# for usage in "${USAGES[@]}"; do
#     for lr in "${LR_LIST[@]}"; do
#         echo "========================================================="
#         echo "Running $usage | LR=$lr | Seed=$SEED"
#         echo "========================================================="
#         SAVE_DIR="$OUTPUT_DIR/${usage}_lr${lr}_seed${SEED}"
#         python main.py \
#             --task cls \
#             --usage "$usage" \
#             --dataset $DATASET \
#             --model $MODEL \
#             --backbone "$BACKBONE" \
#             --sensitive_name Sex \
#             --method erm \
#             --batch_size $BATCH_SIZE \
#             --text_source "$TEXT_SOURCE" \
#             --total_epochs $EPOCHS \
#             --blr $lr \
#             --random_seed "$SEED" \
#             --exp_path "$SAVE_DIR"

#         # Extract the early stopping line from the log
#         if [ -f "$SAVE_DIR/history.log" ]; then
#             BEST_LINE=$(grep "best val AUC" "$SAVE_DIR/history.log" | tail -1)
#             if [ -n "$BEST_LINE" ]; then
#                 # Example line: "Early stopping at epoch 27, best val AUC 0.7531 at epoch 17"
#                 # Extract epoch (the first number after "epoch") and AUC
#                 STOP_EPOCH=$(echo "$BEST_LINE" | grep -oP 'Early stopping at epoch \K\d+')
#                 BEST_EPOCH=$(echo "$BEST_LINE" | grep -oP 'best val AUC [0-9.]* at epoch \K\d+')
#                 BEST_AUC=$(echo "$BEST_LINE" | grep -oP 'best val AUC \K[0-9.]+')
#                 echo "$usage,$lr,$SEED,$BEST_AUC,$BEST_EPOCH" >> "$SUMMARY_FILE"
#             else
#                 echo "$usage,$lr,$SEED,FAILED," >> "$SUMMARY_FILE"
#             fi
#         else
#             echo "$usage,$lr,$SEED,FAILED," >> "$SUMMARY_FILE"
#         fi
#     done
# done

# echo "Sweep finished. Summary in $SUMMARY_FILE"
#!/bin/bash
set +e

DATASET="FairVLMed10k"
MODEL="CLIP"
BACKBONE="ViT-B/32"
EPOCHS=100
BATCH_SIZE=128
OUTPUT_DIR="./lr_sweep"
SEED=0
TEXT_SOURCE="template"

USAGES=("lp" "clip-adapt" "full-ft" "full-ft-zs" "lora-lp" "lora-zs")
LR_LIST=(1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2)

mkdir -p "$OUTPUT_DIR"
SUMMARY_FILE="$OUTPUT_DIR/sweep_summary.csv"

# Write header only if file doesn't exist
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "usage,lr,seed,best_val_auc,best_epoch" > "$SUMMARY_FILE"
fi

for usage in "${USAGES[@]}"; do
    for lr in "${LR_LIST[@]}"; do
        echo "========================================================="
        echo "Checking $usage | LR=$lr | Seed=$SEED"
        SAVE_DIR="$OUTPUT_DIR/${usage}_lr${lr}_seed${SEED}"

        # ---------- Skip if already completed ----------
        if [ -f "$SAVE_DIR/best_ckpt.pth" ]; then
            # Try to extract info from the most recent log
            LOG=$(ls -t "$SAVE_DIR"/history_*.log 2>/dev/null | head -1)
            if [ -n "$LOG" ]; then
                # 1. Try early stopping line
                BEST_LINE=$(grep "best val AUC" "$LOG" | tail -1)
                if [ -z "$BEST_LINE" ]; then
                    # 2. Fallback: final checkpoint loading line
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
            --sensitive_name Sex \
            --method erm \
            --batch_size $BATCH_SIZE \
            --text_source "$TEXT_SOURCE" \
            --total_epochs $EPOCHS \
            --blr $lr \
            --random_seed "$SEED" \
            --exp_path "$SAVE_DIR"

        # After run, extract best validation AUC from log
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
done

echo "Sweep finished. Summary in $SUMMARY_FILE"