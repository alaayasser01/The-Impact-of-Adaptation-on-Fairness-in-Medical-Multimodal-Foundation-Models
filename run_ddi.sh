#!/bin/bash
set +e

DATASET="DDI"
MODEL="CLIP"
BACKBONE="ViT-B/32"
EPOCHS=100            # Kept at 100 to let Early Stopping do its job
BATCH_SIZE=32         # Lowered to 32 for stable gradients on a 459-image dataset
TEXT_SOURCE="template"
SEED=2

OUTPUT_DIR="./ddi_results"
mkdir -p "$OUTPUT_DIR"

# The 4 core models
USAGES=("clip-zs" "full-ft" "lora-zs" "full-ft-zs" )

declare -A BEST_LR
BEST_LR[clip-zs]="1e-2" # Ignored by the script
BEST_LR[full-ft]="1e-2"
BEST_LR[lora-zs]="1e-2"
BEST_LR[full-ft-zs]="5e-3"


for usage in "${USAGES[@]}"; do
    lr="${BEST_LR[$usage]}"
    SAVE_DIR="$OUTPUT_DIR/${usage}"
    
    echo "========================================================="
    echo "🩺 DDI EXPERIMENT | Model: $usage"
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
            --exp_path "$SAVE_DIR" \
            --random_seed "$SEED"

    fi
done

echo "🎉 DDI Generalization Experiment Completed!"