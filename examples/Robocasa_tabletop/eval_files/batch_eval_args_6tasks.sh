#!/bin/bash

# ============================================================
# Argument Parsing
# ============================================================
###########################################################################################
# === Please modify the paths to Python executables in conda environments ===
export MUJOCO_GL=osmesa

cd /home/shenboyang/myProjects/starVLA
starVLA_PYTHON=~/miniforge3/envs/starVLA/bin/python
ROBOCASA_PYTHON=~/miniforge3/envs/robocasa/bin/python
export PYTHONPATH=$(pwd):${PYTHONPATH}
CKPT_DEFAULT="StarVLA/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt"


# === End of environment variable configuration ===
###########################################################################################


N_ENVS_DEFAULT=1
N_ACTION_STEPS_DEFAULT=16

BASE_PORT=6398

# Parse command-line arguments
CKPT_PATH=${1:-$CKPT_DEFAULT}
N_ENVS=${2:-$N_ENVS_DEFAULT}
CUDA_VISIBLE_DEVICES=${3:-"0,1,2,3,4,5,6,7"}  # 默认使用 8 个 GPU
N_ACTION_STEPS=${4:-$N_ACTION_STEPS_DEFAULT}


# MAX_EPISODE_STEPS 写死为 720
MAX_EPISODE_STEPS=720

# 解析 CUDA_VISIBLE_DEVICES 为 GPU 列表
IFS="," read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_LIST[@]}

# 验证 GPU 列表
if [ $NUM_GPUS -eq 0 ]; then
    echo "Error: No GPU specified in CUDA_VISIBLE_DEVICES"
    exit 1
fi


echo "=== Evaluation Configuration ==="
echo "Checkpoint Path      : ${CKPT_PATH}"
echo "Number of Envs       : ${N_ENVS}"
echo "Max Episode Steps    : ${MAX_EPISODE_STEPS}"
echo "Action Chunk Length  : ${N_ACTION_STEPS}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo "================================"

# ============================================================
# Evaluation Function
# ============================================================

EvalEnv() {
    local GPU_ID=$1
    local PORT=$2
    local ENV_NAME=$3
    local CKPT_PATH=$4
    local LOG_DIR=$5
    local ROBOCASA_PYTHON=$6
    local N_ENVS=$7
    local MAX_EPISODE_STEPS=$8
    local N_ACTION_STEPS=$9
    # save root CKPT_PATH 的 parent 文件夹
    local SAVE_ROOT=$(dirname "$(dirname "$CKPT_PATH")")
    local ckpt_name=$(basename "$CKPT_PATH" .pt)
    # 使用统一的视频输出目录
    local VIDEO_OUT_PATH="${VIDEO_BASE_DIR}/${ENV_NAME}"
    mkdir -p "${VIDEO_OUT_PATH}"

    echo "Launching evaluation | GPU ${GPU_ID} | Port ${PORT} | Env ${ENV_NAME}"
    echo "Log file: ${LOG_DIR}/eval_env_${ENV_NAME//\//_}_gpu${GPU_ID}.log"

    # 同时输出到终端和日志文件
    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ${ROBOCASA_PYTHON} -u examples/Robocasa_tabletop/eval_files/simulation_env.py \
        --args.env_name "${ENV_NAME}" \
        --args.port "${PORT}" \
        --args.n_episodes 50 \
        --args.n_envs "${N_ENVS}" \
        --args.max_episode_steps "${MAX_EPISODE_STEPS}" \
        --args.n_action_steps "${N_ACTION_STEPS}" \
        --args.video_out_path "${VIDEO_OUT_PATH}" \
        --args.pretrained_path "${CKPT_PATH}" \
        2>&1 | tee "${LOG_DIR}/eval_env_${ENV_NAME//\//_}_gpu${GPU_ID}.log"
}

# ============================================================
# Environment List (Only 6 tasks from fourier_gr1_unified_1000_6)
# ============================================================

ENV_NAMES=(
  gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
)

# ============================================================
# Runtime Configuration
# ============================================================



# 从 checkpoint 路径提取模型根目录（参考 LIBERO 逻辑）
model_root=$(echo "$CKPT_PATH" | awk -F'/checkpoints/' '{print $1}')

# 使用统一的评估路径，包含所有日志和视频
EVAL_ROOT="${model_root}/eval_robocasa_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${EVAL_ROOT}/logs"
VIDEO_BASE_DIR="${EVAL_ROOT}/videos"
SERVER_LOG_DIR="${EVAL_ROOT}/server_logs"

# 创建目录
mkdir -p "${LOG_DIR}"
mkdir -p "${VIDEO_BASE_DIR}"
mkdir -p "${SERVER_LOG_DIR}"

echo "=== Launching Multi-GPU Evaluation ==="
echo "GPUs            : ${NUM_GPUS}"
echo "Num Environments: ${#ENV_NAMES[@]}"
echo "Evaluation Root : ${EVAL_ROOT}"
echo "Log Directory   : ${LOG_DIR}"
echo "Video Directory : ${VIDEO_BASE_DIR}"
echo "Server Log Dir  : ${SERVER_LOG_DIR}"

# ============================================================
# Step 1: Launch Policy Servers
# ============================================================

SERVER_PIDS=()

for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=${GPU_LIST[$i]}
    PORT=$((BASE_PORT + i))
    echo "Starting policy server | GPU ${GPU_ID} | Port ${PORT}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ${starVLA_PYTHON} deployment/model_server/server_policy.py \
        --ckpt_path "${CKPT_PATH}" \
        --port "${PORT}" \
        --use_bf16 \
        > "${SERVER_LOG_DIR}/server_gpu${GPU_ID}_port${PORT}.log" 2>&1 &

    SERVER_PIDS[$i]=$!

    sleep 10
done

sleep 30

# ============================================================
# Step 2: Dispatch Environments to GPUs
# ============================================================

COUNT=0
for ENV_NAME in "${ENV_NAMES[@]}"; do
    gpu_index=$((COUNT % NUM_GPUS))
    GPU_ID=${GPU_LIST[$gpu_index]}
    PORT=$((BASE_PORT + gpu_index))

    if (( (COUNT + 1) % NUM_GPUS == 0 )); then
        EvalEnv "${GPU_ID}" "${PORT}" "${ENV_NAME}" "${CKPT_PATH}" "${LOG_DIR}" \
                "${ROBOCASA_PYTHON}" "${N_ENVS}" "${MAX_EPISODE_STEPS}" "${N_ACTION_STEPS}"
    else
        EvalEnv "${GPU_ID}" "${PORT}" "${ENV_NAME}" "${CKPT_PATH}" "${LOG_DIR}" \
                "${ROBOCASA_PYTHON}" "${N_ENVS}" "${MAX_EPISODE_STEPS}" "${N_ACTION_STEPS}" &
        # Wait for osmesa rendering context to initialize before launching next env.
        # Concurrent osmesa init across processes causes a race condition / segfault.
        sleep 60
    fi

    COUNT=$((COUNT + 1))
done

# ============================================================
# Step 3: Cleanup
# ============================================================

# 等待仿真环境完成
echo "Waiting for all evaluation environments to finish..."
echo "Checking for simulation_env.py processes..."

# 使用更可靠的进程匹配方式
wait_count=0
max_wait=60  # 最大等待次数（约30分钟）

while true; do
    # 检查是否有 simulation_env.py 进程在运行
    sim_processes=$(pgrep -f "simulation_env.py" 2>/dev/null | wc -l)
    
    if [ $sim_processes -eq 0 ]; then
        echo "No simulation processes found. All evaluations completed."
        break
    fi
    
    echo "Found $sim_processes simulation processes still running..."
    echo "Current processes:"
    pgrep -f "simulation_env.py" -l 2>/dev/null
    
    wait_count=$((wait_count + 1))
    if [ $wait_count -ge $max_wait ]; then
        echo "Timeout: Exceeded maximum wait time. Forcing cleanup."
        break
    fi
    
    echo "Waiting for 30 seconds..."
    sleep 30
done

echo ""
echo "Shutting down policy servers..."

for PID in "${SERVER_PIDS[@]}"; do
    if kill "${PID}" 2>/dev/null; then
        echo "Killed server PID ${PID}"
    else
        echo "Server PID ${PID} already exited"
    fi
done

echo "=== Evaluation Finished ==="

# ============================================================
# Step 4: Log Aggregation
# ============================================================

echo "=== Aggregating Evaluation Results ==="

# 创建汇总报告文件
SUMMARY_FILE="${EVAL_ROOT}/evaluation_summary.log"
echo "# StarVLA Robocasa Evaluation Summary" > "${SUMMARY_FILE}"
echo "# Evaluation Time: $(date)" >> "${SUMMARY_FILE}"
echo "# Evaluation Root: ${EVAL_ROOT}" >> "${SUMMARY_FILE}"
echo "# Checkpoint: ${CKPT_PATH}" >> "${SUMMARY_FILE}"
echo "# GPUs: ${CUDA_VISIBLE_DEVICES}" >> "${SUMMARY_FILE}"
echo "# Environments: ${#ENV_NAMES[@]}" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "## Environment Results" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# 遍历所有环境日志，提取关键信息
for ENV_NAME in "${ENV_NAMES[@]}"; do
    env_log_file="${LOG_DIR}/eval_env_${ENV_NAME//\//_}_gpu*.log"
    if [ -f $env_log_file ]; then
        # 提取成功率
        success_rate=$(grep -o "Success rate: [0-9.]*" $env_log_file | tail -1 | cut -d' ' -f3)
        # 提取平均时间
        avg_time=$(grep -o "Average forward time: [0-9.]*" $env_log_file | tail -1 | cut -d' ' -f4)
        
        echo "### ${ENV_NAME}" >> "${SUMMARY_FILE}"
        echo "- Success rate: ${success_rate:-N/A}" >> "${SUMMARY_FILE}"
        echo "- Average forward time: ${avg_time:-N/A} seconds" >> "${SUMMARY_FILE}"
        echo "- Log file: $(basename $env_log_file)" >> "${SUMMARY_FILE}"
        echo "- Video directory: ${VIDEO_BASE_DIR}/${ENV_NAME}" >> "${SUMMARY_FILE}"
        echo "" >> "${SUMMARY_FILE}"
    fi
done

echo "## Server Logs" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# 提取服务器日志信息
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=${GPU_LIST[$i]}
    server_log_file="${SERVER_LOG_DIR}/server_gpu${GPU_ID}_port$((BASE_PORT + i)).log"
    if [ -f "${server_log_file}" ]; then
        # 提取服务器启动信息
        start_info=$(grep -o "Starting server on port" "${server_log_file}" | head -1)
        
        echo "- Server GPU ${GPU_ID}: ${start_info:-N/A}" >> "${SUMMARY_FILE}"
        echo "  Log file: ${server_log_file}" >> "${SUMMARY_FILE}"
    fi
done

echo "" >> "${SUMMARY_FILE}"
echo "=== Summary Generated ==="
echo "Summary file: ${SUMMARY_FILE}"
echo "All evaluation files saved to: ${EVAL_ROOT}"