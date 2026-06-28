# #!/bin/bash

# # Activate conda and go to project root
# source /home/aym/Desktop/rhome/aym/miniconda3/etc/profile.d/conda.sh
# conda activate fairmedfm
# cd "$(dirname "$0")" || { echo "Cannot cd to script directory"; exit 1; }

# # Create log directory
# mkdir -p logs

# # Define configurations: usage and best learning rate (blr)
# configs=(
#     "lp         1e-2"
#     "clip-adapt 1e-2"
#     "full-ft    1e-2"
#     "full-ft-zs 5e-3"
#     "lora-lp    1e-2"
#     "lora-zs    1e-2"
# )

# logfile="logs/sequential_runs.log"
# echo "All experiments started at $(date)" | tee -a "$logfile"

# for config in "${configs[@]}"; do
#     read -r usage blr <<< "$config"
#     echo "-------------------------------------------------" | tee -a "$logfile"
#     echo "STARTING $usage with lr=$blr at $(date)" | tee -a "$logfile"

#     python main.py \
#         --task cls \
#         --usage "$usage" \
#         --model CLIP \
#         --dataset FairVLMed10k \
#         --sensitive_name All \
#         --blr "$blr" \
#         --total_epochs 100 \
#         --random_seed 0 \
#         --save_test_per_epoch \
#         --warmup_epochs 5 \
#         --early_stopping 1 \
#         2>&1 | tee -a "logs/${usage}.log"

#     echo "FINISHED $usage at $(date)" | tee -a "$logfile"
# done

# echo "All experiments completed at $(date)" | tee -a "$logfile"


#!/bin/bash
source /home/aym/Desktop/rhome/aym/miniconda3/etc/profile.d/conda.sh
conda activate fairmedfm
cd "$(dirname "$0")" || exit 1
mkdir -p logs

configs=(
    "lp         1e-2"
    "clip-adapt 1e-2"
    "full-ft    1e-2"
    "full-ft-zs 5e-3"
    "lora-lp    1e-2"
    "lora-zs    1e-2"
)

logfile="logs/seed0_runs.log"
echo "Seed 0 experiments started at $(date)" | tee -a "$logfile"

for config in "${configs[@]}"; do
    read -r usage blr <<< "$config"
    echo "=================================================" | tee -a "$logfile"
    echo "STARTING $usage with lr=$blr, seed=0 at $(date)" | tee -a "$logfile"

    python main.py \
        --task cls \
        --usage "$usage" \
        --model CLIP \
        --dataset FairVLMed10k \
        --sensitive_name All \
        --blr "$blr" \
        --total_epochs 100 \
        --random_seed 0 \
        --save_test_per_epoch \
        --warmup_epochs 5 \
        --early_stopping 1 \
        --exp_path ./output_seed0 \
        2>&1 | tee -a "logs/${usage}_seed0.log"

    echo "FINISHED $usage at $(date)" | tee -a "$logfile"
done

echo "All seed 0 experiments completed at $(date)" | tee -a "$logfile"