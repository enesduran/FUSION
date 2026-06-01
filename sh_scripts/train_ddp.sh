#!/bin/bash
# Multi-GPU DDP training via HTCondor
# Usage: submitted by train_ddp.sub

export PYTHON="/is/cluster/eduran2/miniconda3/envs/fusion/bin/python"
export PATH=$PATH

TRAINER_FILE=${1:-"configs/trainer1_setting1.yaml"}

# Detect how many GPUs were allocated by Condor
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES (count: $NUM_GPUS)"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching DDP training with $NUM_GPUS GPUs..."
    # torchrun handles RANK, LOCAL_RANK, WORLD_SIZE env vars automatically
    # for a single-node multi-GPU setup
    $(dirname $PYTHON)/torchrun \
        --standalone \
        --nproc_per_node="$NUM_GPUS" \
        train.py --trainer-file "$TRAINER_FILE"
else
    echo "Single GPU detected, launching standard training..."
    $PYTHON train.py --trainer-file "$TRAINER_FILE"
fi

echo DONE
