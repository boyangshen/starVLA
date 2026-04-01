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

# TODO: 修改为你的 checkpoint 路径
your_ckpt=/memory/shenboyang/outputs/train/starvla/libero_loopoft_6x4_s2/checkpoints/steps_20000_pytorch_model.pt

# 4 个子集
task_suites=("libero_10" "libero_goal" "libero_spatial" "libero_object")

# Policy Server 端口
server_port=5696

# 每个任务的尝试次数
num_trials_per_task=30

# GPU 设置
gpu_id=2

# 从 checkpoint 路径提取目录
model_root=$(echo "$your_ckpt" | awk -F'/checkpoints/' '{print $1}')
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

# 日志目录
log_root="${model_root}/logs"
mkdir -p "$log_root"

# 视频输出目录
video_root="${model_root}/videos"
mkdir -p "$video_root"

# 服务器进程 ID 存储
server_pid_file="/tmp/libero_server.pid"

# 函数：启动服务器
start_server() {
    local port=$1
    # 使用 /tmp 目录存储服务器日志（避免只读文件系统问题）
    local log_file="/tmp/server_${port}_$(date +"%Y%m%d_%H%M%S").log"
    
    echo "=========================================="
    echo "Starting Policy Server"
    echo "Port: $port"
    echo "GPU ID: $gpu_id"
    echo "Server Log: $log_file"
    echo "=========================================="
    
    # 使用 starVLA 环境的 Python 解释器启动服务器
    CUDA_VISIBLE_DEVICES=$gpu_id /home/shenboyang/miniforge3/envs/starVLA/bin/python deployment/model_server/server_policy.py \
        --ckpt_path ${your_ckpt} \
        --port ${port} \
        --use_bf16 > "$log_file" 2>&1 &
    
    server_pid=$!
    echo $server_pid > "$server_pid_file"
    echo "Server started with PID: $server_pid"
    
    # 等待服务器启动
    echo "Waiting for server to be ready..."
    sleep 20
    
    # 检查服务器是否启动成功
    if ! kill -0 $server_pid 2>/dev/null; then
        echo "Error: Server failed to start"
        echo "Check server log: $log_file"
        exit 1
    fi
    
    echo "Server is ready!"
    echo "=========================================="
}

# 函数：停止服务器
stop_server() {
    if [ -f "$server_pid_file" ]; then
        server_pid=$(cat "$server_pid_file")
        if kill -0 $server_pid 2>/dev/null; then
            echo ""
            echo "=========================================="
            echo "Stopping server with PID: $server_pid"
            kill $server_pid
            sleep 2
            echo "Server stopped"
            echo "=========================================="
        fi
        rm -f "$server_pid_file"
    fi
}

# 主函数
main() {
    echo "=========================================="
    echo "LIBERO Parallel Evaluation Script (Single Server)"
    echo "Checkpoint: $your_ckpt"
    echo "GPU ID: $gpu_id"
    echo "Server Port: $server_port"
    echo "Task Suites: ${task_suites[@]}"
    echo "=========================================="
    
    # 启动服务器
    start_server $server_port
    
    # 为每个子集创建并启动 screen 会话
    echo ""
    echo "Starting evaluation for all subsets..."
    echo ""
    
    for i in "${!task_suites[@]}"; do
        task_suite=${task_suites[$i]}
        
        # 创建 screen 会话名称
        screen_name="libero_${task_suite}"
        
        echo "=== Starting screen session: $screen_name ==="
        echo "Task Suite: $task_suite"
        echo "Port: $server_port"
        
        # 在新的 screen 会话中运行评估
        screen -dmS "$screen_name" bash ./examples/LIBERO/eval_files/run_single_subset_no_server.sh \
            "$task_suite" \
            $server_port \
            "$your_ckpt" \
            $num_trials_per_task
        
        echo "Screen session started: $screen_name"
        sleep 1
    done
    
    echo ""
    echo "=========================================="
    echo "All evaluation sessions started!"
    echo "=========================================="
    echo ""
    echo "Useful commands:"
    echo "  List all screen sessions: screen -ls"
    echo "  Attach to a session: screen -r <session_name>"
    echo "  Detach from session: Ctrl+A, D"
    echo "  Kill a session: screen -X -S <session_name> quit"
    echo ""
    echo "Screen session names:"
    for task_suite in "${task_suites[@]}"; do
        screen_name="libero_${task_suite}"
        echo "  - $screen_name (Task: $task_suite)"
    done
    echo ""
    echo "Server PID: $server_pid (stored in $server_pid_file)"
    echo "=========================================="
    
    # 等待所有评估完成
    echo ""
    echo "Waiting for all evaluations to complete..."
    echo "Press Ctrl+C to stop all evaluations and server"
    echo ""
    
    # 设置 trap 来捕获 Ctrl+C，确保服务器被停止
    trap 'echo ""; echo "Stopping all evaluations and server..."; stop_server; exit 0' INT TERM
    
    # 等待所有 screen 会话结束
    while true; do
        all_finished=true
        for task_suite in "${task_suites[@]}"; do
            screen_name="libero_${task_suite}"
            if screen -list | grep -q "$screen_name"; then
                all_finished=false
                break
            fi
        done
        
        if $all_finished; then
            echo ""
            echo "All evaluations completed!"
            break
        fi
        
        sleep 10
    done
    
    # 停止服务器
    stop_server
    
    echo ""
    echo "=========================================="
    echo "All done!"
    echo "=========================================="
}

# 执行主函数
main
