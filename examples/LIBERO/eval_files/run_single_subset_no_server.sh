#!/bin/bash

# 这个脚本用于运行单个 LIBERO 子集的评估
# 用法: bash run_single_subset_no_server.sh <task_suite> <port> <gpu_id> <checkpoint> <num_trials>

task_suite=$1
port=$2
checkpoint=$3
num_trials=$4

cd /home/shenboyang/myProjects/starVLA

# 环境配置
export LIBERO_HOME=/home/shenboyang/myProjects/starVLA_eval/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}
export MUJOCO_GL=osmesa

# 从 checkpoint 路径提取目录
model_root=$(echo "$checkpoint" | awk -F'/checkpoints/' '{print $1}')
folder_name=$(echo "$checkpoint" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

# 日志和视频路径
log_root="${model_root}/logs"
video_root="${model_root}/videos"
log_path="${log_root}/${task_suite}"
video_out_path="${video_root}/${task_suite}/${folder_name}"

mkdir -p "$log_path"
mkdir -p "$video_out_path"

# 添加时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")
eval_log_file="${log_path}/${folder_name}_${timestamp}.log"

echo "=========================================="
echo "Starting evaluation for: $task_suite"
echo "Port: $port"
echo "Checkpoint: $checkpoint"
echo "Eval Log: $eval_log_file"
echo "=========================================="

# 等待服务器启动（确保服务器已经启动）
echo "Waiting for server to be ready on port $port..."
sleep 5

# 运行评估
echo "Running evaluation for: $task_suite"
# 使用 libero 环境的 Python 解释器运行评估
 ~/miniforge3/envs/libero/bin/python ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${checkpoint} \
    --args.host "127.0.0.1" \
    --args.port $port \
    --args.task-suite-name "$task_suite" \
    --args.num-trials-per-task "$num_trials" \
    --args.video-out-path "$video_out_path" \
    2>&1 | tee "$eval_log_file"

echo "Evaluation completed for: $task_suite"
echo "=========================================="
