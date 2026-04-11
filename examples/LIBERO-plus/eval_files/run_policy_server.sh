#!/bin/bash

your_ckpt=/memory/shenboyang/outputs/train/starvla/libero_loopoft_6x4_s1/checkpoints/steps_120000_pytorch_model.pt
base_port=9888
# export ABot_python=path_to_ABot_env_python

# export DEBUG=1

CUDA_VISIBLE_DEVICES=1 python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16