#!/bin/bash

# ============================================================
# Argument Parsing
# ============================================================
###########################################################################################
# === Please modify the paths to Python executables in conda environments ===

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
    local VIDEO_OUT_PATH="${SAVE_ROOT}/videos/${ckpt_name}/n_action_steps_${N_ACTION_STEPS}_max_episode_steps_${MAX_EPISODE_STEPS}_n_envs_${N_ENVS}_${ENV_NAME}"
    mkdir -p "${VIDEO_OUT_PATH}"

    echo "Launching evaluation | GPU ${GPU_ID} | Port ${PORT} | Env ${ENV_NAME}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ${ROBOCASA_PYTHON} examples/Robocasa_tabletop/eval_files/simulation_env.py \
        --args.env_name "${ENV_NAME}" \
        --args.port "${PORT}" \
        --args.n_episodes 50 \
        --args.n_envs "${N_ENVS}" \
        --args.max_episode_steps "${MAX_EPISODE_STEPS}" \
        --args.n_action_steps "${N_ACTION_STEPS}" \
        --args.video_out_path "${VIDEO_OUT_PATH}" \
        --args.pretrained_path "${CKPT_PATH}" \
        > "${LOG_DIR}/eval_env_${ENV_NAME//\//_}_gpu${GPU_ID}.log" 2>&1
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



LOG_DIR="${CKPT_PATH}.log/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "=== Launching Multi-GPU Evaluation ==="
echo "GPUs            : ${NUM_GPUS}"
echo "Num Environments: ${#ENV_NAMES[@]}"
echo "Log Directory   : ${LOG_DIR}"

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
        > "${LOG_DIR}/server_gpu${GPU_ID}_port${PORT}.log" 2>&1 &

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
    fi

    COUNT=$((COUNT + 1))

    sleep 2
done

# ============================================================
# Step 3: Cleanup
# ============================================================

# 判断是否还有 examples/Robocasa_tabletop/eval_files/simulation_env.py
while pgrep -f "examples/Robocasa_tabletop/eval_files/simulation_env.py" > /dev/null; do
    echo "Waiting for all evaluation environments to finish..."
    sleep 30
done

echo ""
echo "Shutting down policy servers..."

for PID in "${SERVER_PIDS[@]}"; do
    kill "${PID}" 2>/dev/null && echo "Killed server PID ${PID}"
done

echo "=== Evaluation Finished ==="
echo ""
echo "n_envs 参数说明："
echo "  n_envs 是指每个环境并行运行的实例数量，即同时运行多少个相同的环境来加速评估。"
echo "  例如，n_envs=2 表示每个环境会同时启动 2 个实例，这样可以在相同时间内完成更多的评估 episode。"
echo "  注意：增加 n_envs 会增加 GPU 内存的使用量，请根据 GPU 内存大小合理设置。"
echo ""
echo "使用方法："
echo "  bash examples/Robocasa_tabletop/eval_files/batch_eval_args_6tasks.sh [checkpoint_path] [n_envs] [n_action_steps] [cuda_visible_devices]"
echo ""
echo "示例："
echo "  # 使用默认 8 个 GPU"
echo "  bash examples/Robocasa_tabletop/eval_files/batch_eval_args_6tasks.sh /path/to/checkpoint.pt 1 12"
echo ""
echo "  # 使用指定的 3 个 GPU"
echo "  bash examples/Robocasa_tabletop/eval_files/batch_eval_args_6tasks.sh /path/to/checkpoint.pt 1 12 \"0,2,3\""
echo ""
echo "  # 使用单个 GPU"
echo "  bash examples/Robocasa_tabletop/eval_files/batch_eval_args_6tasks.sh /path/to/checkpoint.pt 1 12 \"1\""
echo ""
echo "参数说明："
echo "  checkpoint_path      - 模型 checkpoint 路径（可选，默认使用 CKPT_DEFAULT）"
echo "  n_envs              - 每个环境并行运行的实例数量（可选，默认 1）"
echo "  n_action_steps      - 动作块长度（可选，默认 12）"
echo "  cuda_visible_devices - 可见的 GPU 列表，以逗号分隔（可选，默认 \"0,1,2,3,4,5,6,7\"）"
echo "  MAX_EPISODE_STEPS   - 固定为 720，不再作为命令行参数"
