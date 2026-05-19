# #!/bin/bash

# set -e

# DATASET="FairVLMed10k"
# MODEL="CLIP"
# BACKBONE="ViT-B/32"
# EPOCHS=100       # <--- Change to 100 for the real run!
# LR=0.0001
# BATCH_SIZE=128 

# OUTPUT_DIR="/home/aym/Desktop/rhome/aym/FairMedFM/allisgood100ada[t]"
# # ALL 8 USAGES ADDED HERE!
# # USAGES=("clip-zs" "clip-adapt"  "full-ft" "full-ft-zs" "lp" "lora-lp" "lora-zs")
# # SEEDS=(0 1 2)
# USAGES=("clip-adapt")
# SEEDS=(1 )
# text_source="all_summaries"  # <-- Change this to "template" or "gpt4_avg" as needed

# for usage in "${USAGES[@]}"; do
#     for seed in "${SEEDS[@]}"; do
#         echo "========================================================="
#         echo "🚀 Running Usage: $usage | Seed: $seed | Text: $text_source"
#         echo "========================================================="
#         python main.py \
#             --task cls \
#             --usage "$usage" \
#             --dataset $DATASET \
#             --model $MODEL \
#             --backbone "$BACKBONE" \
#             --sensitive_name Sex \
#             --method erm \
#             --batch_size $BATCH_SIZE \
#             --text_source "$text_source" \
#             --total_epochs $EPOCHS \
#             --blr $LR \
#             --random_seed "$seed"\
#             --exp_path "$OUTPUT_DIR"
#     done
# done

# echo "🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"

#!/bin/bash
set +e   # Do NOT exit on error – we want to continue after failures

DATASET="FairVLMed10k"
MODEL="CLIP"
BACKBONE="ViT-B/32"
EPOCHS=1            # Quick test
LR=0.0001
BATCH_SIZE=128
OUTPUT_DIR="./output_test"
SEED=0

# Usages that DO use text
# TEXT_USAGES=("clip-zs" "clip-adapt" "full-ft-zs" "lora-zs")
TEXT_USAGES=( "full-ft-zs" )

# Usages that DO NOT use text (they ignore --text_source)
# PURE_VISION_USAGES=("lp" "full-ft" "lora-lp")

# Text sources to test
# TEXT_SOURCES=("template" "gpt4_avg" "all_summaries" "clustered_summaries")
TEXT_SOURCES=("template" )

# TEXT_SOURCES=("clustered_summaries")

# # Run pure vision methods (no text_source loop)
# for usage in "${PURE_VISION_USAGES[@]}"; do
#     echo "========================================================="
#     echo "🚀 Running Usage: $usage | Seed: $SEED"
#     echo "========================================================="
#     python main.py \
#         --task cls \
#         --usage "$usage" \
#         --dataset $DATASET \
#         --model $MODEL \
#         --backbone "$BACKBONE" \
#         --sensitive_name Sex \
#         --method erm \
#         --batch_size $BATCH_SIZE \
#         --total_epochs $EPOCHS \
#         --blr $LR \
#         --random_seed "$SEED" \
#         --exp_path "$OUTPUT_DIR" \
#         --text_source template   # dummy, not used
    
#     exit_code=$?
#     if [ $exit_code -ne 0 ]; then
#         echo "❌ ERROR: $usage failed with exit code $exit_code"
#     else
#         echo "✅ $usage completed successfully"
#     fi
# done

# Run text‑guided methods over all text sources
for usage in "${TEXT_USAGES[@]}"; do
    for text_source in "${TEXT_SOURCES[@]}"; do
        echo "========================================================="
        echo "🚀 Running Usage: $usage | Seed: $SEED | Text: $text_source"
        echo "========================================================="
        python main.py \
            --task cls \
            --usage "$usage" \
            --dataset $DATASET \
            --model $MODEL \
            --backbone "$BACKBONE" \
            --sensitive_name Sex \
            --method erm \
            --batch_size $BATCH_SIZE \
            --text_source "$text_source" \
            --total_epochs $EPOCHS \
            --blr $LR \
            --random_seed "$SEED" \
            --exp_path "$OUTPUT_DIR"
        
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "❌ ERROR: $usage + $text_source failed with exit code $exit_code"
        else
            echo "✅ $usage + $text_source completed successfully"
        fi
    done
done

echo "🎉 ALL QUICK TESTS FINISHED"