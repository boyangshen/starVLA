#!/bin/bash

cd /home/shenboyang/myProjects/starVLA

###########################################################################################
# === 环境配置 ===
export PYTHONPATH=$(pwd):${PYTHONPATH}

# TODO: 修改为你的 checkpoint 路径
your_ckpt=/memory/shenboyang/outputs/train/starvla/libero_spatial_qwen3_2b_oft/checkpoints/steps_50000_pytorch_model.pt

# GPU 设置
gpu_id=0

# 端口（需要与 eval_libero.sh 中的 port 一致）
port=5694
# === End of environment variable configuration ===
###########################################################################################

echo "=========================================="
echo "Checkpoint: $your_ckpt"
echo "GPU ID: $gpu_id"
echo "Port: $port"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$gpu_id python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16
