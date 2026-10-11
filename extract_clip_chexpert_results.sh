# #!/bin/bash
# set +e

# # ================= Configuration =================
# DATASET="FairVLMed10k"                               # Set to "CXP" (or "CheXpert" if registered under that name)
# MODEL="CLIP"
# BACKBONE="ViT-B/32"
# BATCH_SIZE=128
# OUTPUT_DIR="/home/alaa/papersubmission/The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-Foundation-Models/FairVLMed10k_clip_results"
# SEED=1
# TEXT_SOURCE="template"

# # 1. Usages and Specific Attributes
# # USAGES=("clip-zs" "lp" "clip-adapt" "full-ft" "lora-lp" "lora-zs" "full-ft-zs")
# # USAGES=("clip-zs" "lp" )

# # SENSITIVE_ATTRS=( "Ethnicity" "MaritalStatus")
# SENSITIVE_ATTRS=("Sex" "Age" "Race" "Language" "Ethnicity" "MaritalStatus")

# SUMMARY_FILE="$OUTPUT_DIR/eval_summary.csv"

# # Create header for the summary file
# if [ ! -f "$SUMMARY_FILE" ]; then
#     echo "usage,sensitive_attr,eval_auc" > "$SUMMARY_FILE"
# fi

# for usage in "${USAGES[@]}"; do
#     # Match the LR used during training to point to the correct folder
#     if [ "$usage" == "full-ft-zs" ] || [ "$usage" == "full-ft" ]; then
#         lr="5e-3"
#     else
#         lr="1e-2"
#     fi

#     SAVE_DIR="$OUTPUT_DIR/${usage}_lr${lr}_seed${SEED}"

#     for sens in "${SENSITIVE_ATTRS[@]}"; do
#         echo "========================================================="
#         echo "Evaluating $usage | LR=$lr | Seed=$SEED | Attribute=$sens"
        
#         # Pure Zero-Shot has no checkpoint, so run it normally
#         if [ "$usage" == "clip-zs" ]; then
#             python main.py \
#                 --task cls \
#                 --usage "$usage" \
#                 --dataset "$DATASET" \
#                 --model "$MODEL" \
#                 --backbone "$BACKBONE" \
#                 --sensitive_name "$sens" \
#                 --method erm \
#                 --batch_size "$BATCH_SIZE" \
#                 --text_source "$TEXT_SOURCE" \
#                 --random_seed "$SEED" \
#                 --exp_path "$SAVE_DIR"
#         else
#             # For all other methods, append --eval_only to load best_ckpt.pth
#             python main.py \
#                 --task cls \
#                 --usage "$usage" \
#                 --dataset "$DATASET" \
#                 --model "$MODEL" \
#                 --backbone "$BACKBONE" \
#                 --sensitive_name "$sens" \
#                 --method erm \
#                 --batch_size "$BATCH_SIZE" \
#                 --text_source "$TEXT_SOURCE" \
#                 --random_seed "$SEED" \
#                 --exp_path "$SAVE_DIR" \
#                 --eval_only
#         fi
        
#         # --- Auto-Extract Result into CSV ---
#         # It looks for the most recently modified log in the specific attribute folder
#         LOG=$(ls -t "$SAVE_DIR"/cls/"$usage"/"$TEXT_SOURCE"/*/*/CXP/"$MODEL"/"$sens"/seed"$SEED"/*.log 2>/dev/null | head -1)
#         if [ -n "$LOG" ]; then
#             EVAL_AUC=$(grep -i "overall-auc" "$LOG" | tail -1 | grep -oP 'overall-auc: \K(\d+\.\d+)')
#             if [ -n "$EVAL_AUC" ]; then
#                 echo "$usage,$sens,$EVAL_AUC" >> "$SUMMARY_FILE"
#             else
#                 echo "$usage,$sens,FAILED" >> "$SUMMARY_FILE"
#             fi
#         else
#             echo "$usage,$sens,LOG_NOT_FOUND" >> "$SUMMARY_FILE"
#         fi
#     done
# done

# echo "========================================================="
# echo "Evaluation sweep finished! Summary saved to: $SUMMARY_FILE"
#!/bin/bash
set +e

# ================= Configuration =================
DATASET="FairVLMed10k"
MODEL="BiomedCLIP" # <--- UPDATED
BACKBONE="ViT-B/32"
BATCH_SIZE=128
# <--- UPDATED DIRECTORIES BELOW
OUTPUT_DIR="/home/alaa/papersubmission/The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-Foundation-Models/biomedclip_results"
OUTPUT_DIR_final="/home/alaa/papersubmission/The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-Foundation-Models/Fairmedvlm_biomedclip_results_BINARY"
SEED=1
TEXT_SOURCE="template"

USAGES=("full-ft")

# USAGES=("clip-zs" "lp" "clip-adapt" "full-ft" "lora-lp" "lora-zs" "full-ft-zs")
SENSITIVE_ATTRS=("Sex" "Age" "Race" "Language" "Ethnicity" "MaritalStatus")

SUMMARY_FILE="$OUTPUT_DIR_final/eval_summary_binary.csv"

# ================= 1. Prep the Final Directory =================
echo "Creating final directory and linking old checkpoints..."
mkdir -p "$OUTPUT_DIR_final"

# Safely symlink all .pth files, handling the "All" directory edge case!
find "$OUTPUT_DIR" -name "*.pth" | while read -r pth_file; do
    rel_path="${pth_file#$OUTPUT_DIR/}"
    
    # If the checkpoint is inside an "All" folder, link it to all specific attributes
    if [[ "$rel_path" == *"All/seed"* ]]; then
        for sens in "${SENSITIVE_ATTRS[@]}"; do
            # Replace "All" with the specific attribute in the path
            dest_file="$OUTPUT_DIR_final/${rel_path/All/$sens}"
            mkdir -p "$(dirname "$dest_file")"
            ln -sf "$pth_file" "$dest_file"
        done
    else
        # Standard linking if it's already in a specific attribute folder
        dest_file="$OUTPUT_DIR_final/$rel_path"
        mkdir -p "$(dirname "$dest_file")"
        ln -sf "$pth_file" "$dest_file"
    fi
done

# Create header for the summary file
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "usage,sensitive_attr,eval_auc" > "$SUMMARY_FILE"
fi

# ================= 2. Evaluation Loop =================
for usage in "${USAGES[@]}"; do
    
    # Match the LR used during training to point to the correct folder
    if [ "$usage" == "full-ft-zs" ]; then
        lr="5e-3"
    else
        lr="1e-2"
    fi

    SAVE_DIR="$OUTPUT_DIR_final/${usage}_lr${lr}_seed${SEED}"

    for sens in "${SENSITIVE_ATTRS[@]}"; do
        echo "========================================================="
        echo "Evaluating $usage | LR=$lr | Seed=$SEED | Attribute=$sens"
        
        # Base Python command
        CMD=(python main.py
            --task cls
            --usage "$usage"
            --dataset "$DATASET"
            --model "$MODEL"
            --backbone "$BACKBONE"
            --sensitive_name "$sens"
            --method erm
            --batch_size "$BATCH_SIZE"
            --text_source "$TEXT_SOURCE"
            --random_seed "$SEED"
            --exp_path "$SAVE_DIR"
        )

        # Append eval_only for models that require checkpoints
        if [ "$usage" != "clip-zs" ]; then
            CMD+=(--eval_only)
        fi

        # Execute the command
        "${CMD[@]}"
        
        # --- Auto-Extract Result into CSV ---
        # Robust log finder: gets the most recently modified .log file inside SAVE_DIR
        LOG=$(find "$SAVE_DIR" -type f -name "*.log" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        
        if [ -n "$LOG" ]; then
            EVAL_AUC=$(grep -i "overall-auc" "$LOG" | tail -1 | grep -oP 'overall-auc: \K(\d+\.\d+)')
            if [ -n "$EVAL_AUC" ]; then
                echo "$usage,$sens,$EVAL_AUC" >> "$SUMMARY_FILE"
            else
                echo "$usage,$sens,FAILED" >> "$SUMMARY_FILE"
            fi
        else
            echo "$usage,$sens,LOG_NOT_FOUND" >> "$SUMMARY_FILE"
        fi
    done
done

echo "========================================================="
echo "Evaluation sweep finished! Summary saved to: $SUMMARY_FILE"