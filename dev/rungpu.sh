#!/bin/bash

# GPU training script (assumes initial setup already done via runcpu.sh)
# Run as:
# bash dev/rungpu.sh

# NOTE: This script uses small model configurations for debugging purposes.
# For serious training, increase depth, batch sizes, and iterations.

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Switch to GPU PyTorch and activate
uv sync --extra gpu
source .venv/bin/activate

# train a very small 4 layer model (debugging config)
# each optimization step processes a single sequence of 1024 tokens
# we only run 50 steps of optimization (bump this to get better results)
python -m scripts.base_train \
    --depth=8 \
    --max_seq_len=2048 \
    --device_batch_size=1 \
    --total_batch_size=1024 \
    --eval_every=50 \
    --eval_tokens=4096 \
    --core_metric_every=50 \
    --core_metric_max_per_task=12 \
    --sample_every=50 \
    --num_iterations=50
python -m scripts.base_loss --device_batch_size=1 --split_tokens=4096
python -m scripts.base_eval --max-per-task=16

# midtraining
python -m scripts.mid_train \
    --max_seq_len=2048 \
    --device_batch_size=1 \
    --eval_every=50 \
    --eval_tokens=4096 \
    --total_batch_size=1024 \
    --num_iterations=100
# eval results will be terrible, this is just to execute the code paths.
python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=20

# SFT
python -m scripts.chat_sft \
    --device_batch_size=1 \
    --target_examples_per_step=4 \
    --num_iterations=100 \
    --eval_steps=4 \
    --eval_metrics_max_problems=16

# Chat CLI
# python -m scripts.chat_cli -p "Why is the sky blue?"

# Chat Web
# python -m scripts.chat_web

python -m nanochat.report generate
