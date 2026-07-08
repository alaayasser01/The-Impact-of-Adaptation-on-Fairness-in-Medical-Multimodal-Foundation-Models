#!/bin/bash
set +e

DATASET="FairVLMed10k"
MODEL="CLIP"
BACKBONE="ViT-B/32"
BATCH_SIZE=128
TEXT_SOURCE="template"

# Define the best LR for each usage based on your Seed 0 results
declare -A BEST_LR
BEST_LR[lp]="1e-2"
BEST_LR[clip-adapt]="1e-2"
BEST_LR[full-ft]="1e-2"
BEST_LR[full-ft-zs]="5e-3"
BEST_LR[lora-lp]="1e-2"
BEST_LR[lora-zs]="1e-2"

USAGES=("lp" "clip-adapt" "full-ft" "full-ft-zs" "lora-lp" "lora-zs")
SEEDS=(0)
SWEEP_DIR="./lr_sweep"

for SEED in "${SEEDS[@]}"; do
    for usage in "${USAGES[@]}"; do
        lr="${BEST_LR[$usage]}"
        SAVE_DIR="$SWEEP_DIR/${usage}_lr${lr}_seed${SEED}"
        
        echo "========================================================="
        echo "🧪 EVALUATING | Seed=$SEED | Usage=$usage | LR=$lr"
        echo "========================================================="

        # Run in eval_only mode. 
        # Make sure the models for seed 1 and 2 have actually been trained first!
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

echo "🎉 All evaluations for Seeds 1 and 2 completed!"