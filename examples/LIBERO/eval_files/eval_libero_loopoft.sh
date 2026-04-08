#!/bin/bash

cd /home/shenboyang/myProjects/starVLA

###########################################################################################
# === 环境配置 ===
export LIBERO_HOME=/home/shenboyang/myProjects/starVLA_eval/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}

# 使用 OSMESA 渲染
export MUJOCO_GL=osmesa

host="127.0.0.1"
base_port=5696

# TODO: 修改为你的 checkpoint 路径
your_ckpt=/memory/shenboyang/outputs/train/starvla/libero_loopoft_6x4_s1/checkpoints/steps_100000_pytorch_model.pt

task_suite_name=libero_goal
num_trials_per_task=30

# 从 checkpoint 路径提取目录
model_root=$(echo "$your_ckpt" | awk -F'/checkpoints/' '{print $1}')
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

# 添加时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")

video_out_path="${model_root}/videos/${task_suite_name}/${folder_name}"
log_path="${model_root}/logs/${task_suite_name}"

mkdir -p "$video_out_path"
mkdir -p "$log_path"

echo "=========================================="
echo "LIBERO_HOME: $LIBERO_HOME"
echo "Checkpoint: $your_ckpt"
echo "Task Suite: $task_suite_name"
echo "MUJOCO_GL: $MUJOCO_GL"
echo "Log Path: ${log_path}/${folder_name}_${timestamp}.log"
echo "=========================================="

python ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    2>&1 | tee ${log_path}/${folder_name}_${timestamp}.log
