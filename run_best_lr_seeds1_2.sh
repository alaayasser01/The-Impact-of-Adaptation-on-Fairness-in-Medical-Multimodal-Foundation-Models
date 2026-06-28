# #!/bin/bash
# set +e

# DATASET="FairVLMed10k"
# MODEL="CLIP"
# BACKBONE="ViT-B/32"
# EPOCHS=100               # set to 100 for the real run
# BATCH_SIZE=128
# OUTPUT_DIR="./lr_sweep"
# TEXT_SOURCE="template"

# declare -A BEST_LR
# BEST_LR[lp]="1e-2"
# BEST_LR[clip-adapt]="1e-2"
# BEST_LR[full-ft]="1e-2"
# BEST_LR[full-ft-zs]="5e-3"
# BEST_LR[lora-lp]="1e-2"
# BEST_LR[lora-zs]="1e-2"

# USAGES=("lp" "clip-adapt" "full-ft" "full-ft-zs" "lora-lp" "lora-zs")
# # USAGES=("lp")

# SEEDS=(1 2)

# mkdir -p "$OUTPUT_DIR"

# for SEED in "${SEEDS[@]}"; do
#     SUMMARY_FILE="$OUTPUT_DIR/sweep_summary_seed${SEED}.csv"
#     if [ ! -f "$SUMMARY_FILE" ]; then
#         echo "usage,lr,seed,best_val_auc,best_epoch" > "$SUMMARY_FILE"
#     fi

#     for usage in "${USAGES[@]}"; do
#         lr="${BEST_LR[$usage]}"
#         SAVE_DIR="$OUTPUT_DIR/${usage}_lr${lr}_seed${SEED}"

#         echo "========================================================="
#         echo "Seed=$SEED | $usage | LR=$lr"
#         echo "========================================================="

#         # Skip if already completed (best checkpoint exists)
#         if [ -f "$SAVE_DIR/best_ckpt.pth" ]; then
#             LOG=$(ls -t "$SAVE_DIR"/history_*.log 2>/dev/null | head -1)
#             if [ -n "$LOG" ]; then
#                 BEST_LINE=$(grep "best val AUC" "$LOG" | tail -1)
#                 if [ -z "$BEST_LINE" ]; then
#                     BEST_LINE=$(grep "Loaded best checkpoint from epoch" "$LOG" | tail -1)
#                 fi
#                 if [ -n "$BEST_LINE" ]; then
#                     BEST_AUC=$(echo "$BEST_LINE" | grep -oP '(\d+\.\d+)(?=.*val AUC)' | head -1)
#                     BEST_EPOCH=$(echo "$BEST_LINE" | grep -oP 'epoch \K\d+' | tail -1)
#                     echo "$usage,$lr,$SEED,$BEST_AUC,$BEST_EPOCH" >> "$SUMMARY_FILE"
#                     echo "  => Already completed, skipping."
#                     continue
#                 fi
#             fi
#         fi

#         echo "  => Running..."
#         python main.py \
#             --task cls \
#             --usage "$usage" \
#             --dataset "$DATASET" \
#             --model "$MODEL" \
#             --backbone "$BACKBONE" \
#             --sensitive_name Sex \
#             --method erm \
#             --batch_size $BATCH_SIZE \
#             --text_source "$TEXT_SOURCE" \
#             --total_epochs $EPOCHS \
#             --blr "$lr" \
#             --random_seed "$SEED" \
#             --exp_path "$SAVE_DIR"

#         # Extract after run
#         LOG=$(ls -t "$SAVE_DIR"/history_*.log 2>/dev/null | head -1)
#         if [ -n "$LOG" ]; then
#             BEST_LINE=$(grep "best val AUC" "$LOG" | tail -1)
#             if [ -z "$BEST_LINE" ]; then
#                 BEST_LINE=$(grep "Loaded best checkpoint from epoch" "$LOG" | tail -1)
#             fi
#             if [ -n "$BEST_LINE" ]; then
#                 BEST_AUC=$(echo "$BEST_LINE" | grep -oP '(\d+\.\d+)(?=.*val AUC)' | head -1)
#                 BEST_EPOCH=$(echo "$BEST_LINE" | grep -oP 'epoch \K\d+' | tail -1)
#                 echo "$usage,$lr,$SEED,$BEST_AUC,$BEST_EPOCH" >> "$SUMMARY_FILE"
#             else
#                 echo "$usage,$lr,$SEED,FAILED," >> "$SUMMARY_FILE"
#             fi
#         else
#             echo "$usage,$lr,$SEED,FAILED," >> "$SUMMARY_FILE"
#         fi
#     done
# done

# echo "All seeds finished. Summaries in $OUTPUT_DIR/sweep_summary_seed*.csv"

#!/bin/bash
set +e

DATASET="FairVLMed10k"
MODEL="CLIP"
BACKBONE="ViT-B/32"
EPOCHS=100
BATCH_SIZE=128
OUTPUT_DIR="./lr_sweep"
TEXT_SOURCE="template"

declare -A BEST_LR
BEST_LR[lp]="1e-2"
BEST_LR[clip-adapt]="1e-2"
BEST_LR[full-ft]="1e-2"
BEST_LR[full-ft-zs]="5e-3"
BEST_LR[lora-lp]="1e-2"
BEST_LR[lora-zs]="1e-2"

# 1. Define strictly the exact combinations that failed
# Format: "SEED:USAGE"
TARGET_RUNS=(
    "1:full-ft-zs"
    "1:lora-lp"
    "1:lora-zs"
    "2:lp"
    "2:clip-adapt"
    "2:full-ft-zs"
    "2:lora-zs"
)

mkdir -p "$OUTPUT_DIR"

for run in "${TARGET_RUNS[@]}"; do
    # Split the SEED and USAGE from our target array
    SEED="${run%%:*}"
    usage="${run##*:}"
    lr="${BEST_LR[$usage]}"

    SAVE_DIR="$OUTPUT_DIR/${usage}_lr${lr}_seed${SEED}"

    echo "========================================================="
    echo "🚀 SURGICAL RE-RUN | Seed=$SEED | $usage | LR=$lr"
    echo "========================================================="

    # 2. CRITICAL: Delete the old corrupted folder so it starts 100% fresh
    if [ -d "$SAVE_DIR" ]; then
        echo "🗑️ Deleting corrupted Epoch 0 directory: $SAVE_DIR"
        rm -rf "$SAVE_DIR"
    fi

    # 3. Run with sensitive_name All
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
        --exp_path "$SAVE_DIR"

done

echo "🎉 All surgical re-runs completed!"