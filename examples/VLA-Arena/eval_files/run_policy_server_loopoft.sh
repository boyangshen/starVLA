#!/bin/bash
# run_policy_server_loopoft.sh
#
# Launches the LoopOFT WebSocket policy server for VLA-Arena evaluation.
# Run this script first, then launch eval_vla_arena.sh in a separate terminal.

export PYTHONPATH=$(pwd):${PYTHONPATH}

###########################################################################################
# === Please modify the following paths according to your environment ===
export starVLA_python=python   # or: /path/to/conda/envs/starVLA/bin/python

your_ckpt=/path/to/your/loopoft_checkpoint.pt
gpu_id=0
port=10090
###########################################################################################

CUDA_VISIBLE_DEVICES=${gpu_id} ${starVLA_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16
