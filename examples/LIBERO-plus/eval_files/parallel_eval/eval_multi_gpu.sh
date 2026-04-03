#!/bin/bash
# eval_multi_gpu.sh — 单机多卡并行评测脚本
#
# 用法（从 starVLA 仓库根目录执行）：
#   bash examples/LIBERO-plus/eval_files/parallel_eval/eval_multi_gpu.sh \
#       <ckpt_path> <output_dir> [num_trials] [procs_per_gpu] [gpu_ids]
#
# 参数说明：
#   ckpt_path     模型 checkpoint 路径（必填）
#   output_dir    评测结果输出目录（必填）
#   num_trials    每个任务的 rollout 次数，默认 30
#   procs_per_gpu 每块 GPU 上的并行 Python 进程数，默认 1
#                 建议：显存 >= 40GB 可设 2，< 40GB 建议保持 1
#   gpu_ids       指定使用哪些 GPU，逗号分隔，默认自动使用全部
#                 示例：0,1,2,3
#
# 示例：
#   # 使用全部 GPU，每 GPU 1 个进程：
#   bash examples/LIBERO-plus/eval_files/parallel_eval/eval_multi_gpu.sh \
#       /memory/shenboyang/outputs/train/starvla/libero_qwen3_2boft/checkpoints/steps_100000_pytorch_model.pt \
#       /memory/shenboyang/outputs/eval/multi_gpu_test
#
#   # 只用 GPU 0 和 1，每 GPU 2 个进程：
#   bash examples/LIBERO-plus/eval_files/parallel_eval/eval_multi_gpu.sh \
#       /path/to/ckpt /path/to/output 30 2 0,1

set -euo pipefail

###########################################################################################
# === 请按实际环境修改以下路径 ===
export LIBERO_HOME=/home/shenboyang/myProjects/starVLA_eval/LIBERO-plus
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export MUJOCO_GL=osmesa
export PYTHONPATH=${PYTHONPATH:-}:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}
# === 路径配置结束 ===
###########################################################################################

# ---------- 参数解析 ----------
YOUR_CKPT=${1:?"[ERROR] 请提供 checkpoint 路径作为第 1 个参数"}
OUTPUT_DIR=${2:?"[ERROR] 请提供输出目录作为第 2 个参数"}
NUM_TRIALS=${3:-1}
PROCS_PER_GPU=${4:-1}
GPU_IDS_ARG=${5:-""}

# ---------- 检测 GPU ----------
if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] 未找到 nvidia-smi，请确认 GPU 驱动已安装"
    exit 1
fi

ALL_GPU_IDS=($(nvidia-smi --query-gpu=index --format=csv,noheader))
TOTAL_GPUS=${#ALL_GPU_IDS[@]}

if [ "$TOTAL_GPUS" -eq 0 ]; then
    echo "[ERROR] 未检测到 GPU"
    exit 1
fi

# 解析用户指定的 GPU 列表
if [ -n "$GPU_IDS_ARG" ]; then
    IFS=',' read -ra GPU_LIST <<< "$GPU_IDS_ARG"
else
    GPU_LIST=("${ALL_GPU_IDS[@]}")
fi
NUM_GPUS=${#GPU_LIST[@]}

echo "========================================"
echo "[INFO] checkpoint  : ${YOUR_CKPT}"
echo "[INFO] output_dir  : ${OUTPUT_DIR}"
echo "[INFO] num_trials  : ${NUM_TRIALS}"
echo "[INFO] procs/GPU   : ${PROCS_PER_GPU}"
echo "[INFO] 使用 GPU    : ${GPU_LIST[*]}  (共 ${NUM_GPUS} 块)"
echo "========================================"

# ---------- 任务套件配置 ----------
# LIBERO-plus 各 suite 的任务总数（来自 run_nebula_libero_plus.sh）
TASK_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
TASK_SIZES=(2402 2518 2591 2519)
NUM_SUITES=${#TASK_SUITES[@]}

mkdir -p "${OUTPUT_DIR}"

# ---------- 启动所有进程 ----------
declare -a PIDS=()
declare -a PID_LABELS=()

for ((suite_idx=0; suite_idx < NUM_SUITES; suite_idx++)); do
    task="${TASK_SUITES[$suite_idx]}"
    size="${TASK_SIZES[$suite_idx]}"

    # 轮询分配 GPU
    gpu_id="${GPU_LIST[$((suite_idx % NUM_GPUS))]}"

    echo ""
    echo "[INFO] Suite: ${task}  (0~${size})  →  GPU ${gpu_id}"

    mkdir -p "${OUTPUT_DIR}/${task}"

    # 将该 suite 的任务范围切分给 PROCS_PER_GPU 个进程
    chunk_size=$((size / PROCS_PER_GPU))
    remainder=$((size % PROCS_PER_GPU))
    cur_start=0

    for ((proc_idx=0; proc_idx < PROCS_PER_GPU; proc_idx++)); do
        if [ "$proc_idx" -lt "$remainder" ]; then
            cur_end=$((cur_start + chunk_size + 1))
        else
            cur_end=$((cur_start + chunk_size))
        fi
        [ "$cur_end" -gt "$size" ] && cur_end=$size

        label="GPU${gpu_id}|${task}[${cur_start},${cur_end})"
        echo "[INFO]   启动进程: ${label}"

        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        python ./examples/LIBERO-plus/eval_files/parallel_eval/eval_libero_model.py \
            --pretrained_path "${YOUR_CKPT}" \
            --task_suite_name "${task}" \
            --num_trials_per_task "${NUM_TRIALS}" \
            --output_dir "${OUTPUT_DIR}" \
            --start_idx "${cur_start}" \
            --end_idx "${cur_end}" &

        PIDS+=($!)
        PID_LABELS+=("${label}")

        cur_start=$cur_end
        [ "$cur_start" -ge "$size" ] && break
    done
done

echo ""
echo "[INFO] 全部 ${#PIDS[@]} 个进程已启动，等待完成..."
echo "[INFO] PIDs: ${PIDS[*]}"
echo ""

# ---------- 等待并检查退出码 ----------
FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    label="${PID_LABELS[$i]}"
    if wait "$pid"; then
        echo "[OK]   ${label}  (pid=${pid})"
    else
        echo "[FAIL] ${label}  (pid=${pid}, exit=$?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ "$FAILED" -gt 0 ]; then
    echo "[WARN] ${FAILED} 个进程异常退出，请检查对应日志：${OUTPUT_DIR}/logs/"
else
    echo "[INFO] 所有进程正常完成"
fi

# ---------- 聚合结果 ----------
echo ""
echo "[INFO] 聚合各 suite 结果..."
python ./examples/LIBERO-plus/eval_files/parallel_eval/aggregate_results.py \
    --root_path "${OUTPUT_DIR}"

echo ""
echo "[INFO] 完成！整体结果：${OUTPUT_DIR}/overall_results.json"
