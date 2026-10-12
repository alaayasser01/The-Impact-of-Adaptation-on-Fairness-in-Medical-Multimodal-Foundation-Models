#!/bin/bash
set -e

# ==============================================================================
# Remote Reproduction Test Runner
# ==============================================================================
# Usage:
#   bash scripts/test_reproduction.sh             # Fast Zero-Shot verification on FairVLMed10k
#   bash scripts/test_reproduction.sh DDI         # Zero-Shot on DDI
#   bash scripts/test_reproduction.sh CXP         # Zero-Shot on CheXpert
#   bash scripts/test_reproduction.sh --env-only  # Only check environment & datasets
# ==============================================================================

DATASET="${1:-FairVLMed10k}"

if [ "$1" == "--env-only" ]; then
    python scripts/verify_reproduction.py --mode check_env
    exit 0
fi

echo "=========================================================================="
echo " Running Remote Reproduction Verification on: $DATASET"
echo "=========================================================================="

python scripts/verify_reproduction.py \
    --mode all \
    --dataset "$DATASET" \
    --model CLIP \
    --batch_size 128

echo ""
echo "Reproduction verification script completed."
