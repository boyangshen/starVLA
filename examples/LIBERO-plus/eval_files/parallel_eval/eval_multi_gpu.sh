#!/bin/bash
# eval_multi_gpu.sh — 单机多卡并行评测脚本（增强版）

# export MUJOCO_GL=osmesa

set -euo pipefail

###########################################################################################
# === 路径配置（按需修改） ===
export LIBERO_HOME=/home/shenboyang/myProjects/starVLA_eval/LIBERO-plus
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export MUJOCO_GL=osmesa
export PYTHONPATH=${PYTHONPATH:-}:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}
###########################################################################################

# ---------- 参数解析 ----------
YOUR_CKPT=${1:?"[ERROR] 请提供 checkpoint 路径"}
OUTPUT_DIR=${2:?"[ERROR] 请提供输出目录"}
NUM_TRIALS=${3:-1}

SLICES_PER_SUITE=${4:-""}
if [ -z "$SLICES_PER_SUITE" ]; then
    echo "[ERROR] 请显式指定 SLICES_PER_SUITE（第4个参数，每个 suite 的切片数）"
    exit 1
fi
if ! [[ "$SLICES_PER_SUITE" =~ ^[0-9]+$ ]] || [ "$SLICES_PER_SUITE" -lt 1 ]; then
    echo "[ERROR] SLICES_PER_SUITE 必须是正整数，当前值: ${SLICES_PER_SUITE}"
    exit 1
fi

GPU_IDS_ARG=${5:-""}

# 可选参数
SELECTED_SUITES_ARG=${6:-""}
BASE_PORT=${7:-10093}

# ---------- 检测 GPU ----------
if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] 未找到 nvidia-smi"
    exit 1
fi

ALL_GPU_IDS=($(nvidia-smi --query-gpu=index --format=csv,noheader))
TOTAL_GPUS=${#ALL_GPU_IDS[@]}

if [ "$TOTAL_GPUS" -eq 0 ]; then
    echo "[ERROR] 未检测到 GPU"
    exit 1
fi

# 解析 GPU 列表
if [ -n "$GPU_IDS_ARG" ]; then
    IFS=',' read -ra GPU_LIST <<< "$GPU_IDS_ARG"
else
    GPU_LIST=("${ALL_GPU_IDS[@]}")
fi

NUM_GPUS=${#GPU_LIST[@]}

# ---------- 任务套件配置 ----------
ALL_TASK_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
DEFAULT_TASK_SIZES=(2402 2518 2591 2519)

SUITE_SIZES_ARG=${8:-""}

declare -a TASK_SUITES=()
declare -a TASK_SIZES=()

if [ -n "$SELECTED_SUITES_ARG" ]; then
    IFS=',' read -ra SELECTED_SUITES <<< "$SELECTED_SUITES_ARG"
else
    SELECTED_SUITES=("${ALL_TASK_SUITES[@]}")
fi

if [ -n "$SUITE_SIZES_ARG" ]; then
    IFS=',' read -ra USER_SIZES <<< "$SUITE_SIZES_ARG"
fi

for i in "${!ALL_TASK_SUITES[@]}"; do
    suite="${ALL_TASK_SUITES[$i]}"
    for selected in "${SELECTED_SUITES[@]}"; do
        if [ "$suite" = "$selected" ]; then
            TASK_SUITES+=("$suite")
            if [ -n "${USER_SIZES[$i]:-}" ]; then
                TASK_SIZES+=("${USER_SIZES[$i]}")
            else
                TASK_SIZES+=("${DEFAULT_TASK_SIZES[$i]}")
            fi
            break
        fi
    done
done

if [ ${#TASK_SUITES[@]} -eq 0 ]; then
    echo "[ERROR] 未找到匹配的任务 suite: ${SELECTED_SUITES_ARG}"
    echo "[ERROR] 可用的 suite: ${ALL_TASK_SUITES[*]}"
    exit 1
fi

NUM_SUITES=${#TASK_SUITES[@]}

echo "========================================"
echo "[INFO] checkpoint  : ${YOUR_CKPT}"
echo "[INFO] output_dir  : ${OUTPUT_DIR}"
echo "[INFO] num_trials  : ${NUM_TRIALS}"
echo "[INFO] slices/suite: ${SLICES_PER_SUITE}"
echo "[INFO] 使用 GPU    : ${GPU_LIST[*]}  (共 ${NUM_GPUS} 块)"
echo "[INFO] 评测 suites : ${TASK_SUITES[*]}"
echo "[INFO] suite 大小  : ${TASK_SIZES[*]}"
echo "[INFO] base_port   : ${BASE_PORT}"
echo "========================================"

mkdir -p "${OUTPUT_DIR}"

# ---------- 切片生成 ----------
declare -a slices=()

for ((suite_idx=0; suite_idx < NUM_SUITES; suite_idx++)); do
    task="${TASK_SUITES[$suite_idx]}"
    size="${TASK_SIZES[$suite_idx]}"

    mkdir -p "${OUTPUT_DIR}/${task}"

    # ceil 切片
    chunk_size=$(( (size + SLICES_PER_SUITE - 1) / SLICES_PER_SUITE ))

    cur_start=0

    for ((proc_idx=0; proc_idx < SLICES_PER_SUITE; proc_idx++)); do
        cur_end=$((cur_start + chunk_size))

        # clamp
        [ "$cur_start" -gt "$size" ] && cur_start=$size
        [ "$cur_end" -gt "$size" ] && cur_end=$size

        # 避免空任务
        if [ "$cur_start" -lt "$cur_end" ]; then
            slices+=("${task}:${cur_start}:${cur_end}")
        fi

        cur_start=$cur_end
    done
done

echo ""
echo "[DEBUG] 实际 slices 数量: ${#slices[@]}"
echo "[DEBUG] 预计 slices 数量: $((NUM_SUITES * SLICES_PER_SUITE))"
echo ""

# ---------- 启动进程 ----------
declare -a PIDS=()
declare -a PID_LABELS=()
declare -a LOG_FILES=()

# 修复 gpu_counts（用 map）
declare -A gpu_counts
for gpu in "${GPU_LIST[@]}"; do
    gpu_counts[$gpu]=0
done

num_slices=${#slices[@]}

for ((slice_idx=0; slice_idx < num_slices; slice_idx++)); do
    gpu_id="${GPU_LIST[$((slice_idx % NUM_GPUS))]}"

    IFS=":" read -r task cur_start cur_end <<< "${slices[$slice_idx]}"

    # 计算端口号，每个进程使用不同端口
    port=$((BASE_PORT + slice_idx))

    label="GPU${gpu_id} | ${task} [${cur_start}, ${cur_end}) | port ${port}"
    log_file="${OUTPUT_DIR}/logs_${task}_${cur_start}_${cur_end}.log"

    echo "[INFO] 启动: ${label}"

    # 直接输出到日志文件，不输出到终端
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python ./examples/LIBERO-plus/eval_files/parallel_eval/eval_libero_model.py \
        --pretrained_path "${YOUR_CKPT}" \
        --task_suite_name "${task}" \
        --num_trials_per_task "${NUM_TRIALS}" \
        --output_dir "${OUTPUT_DIR}" \
        --start_idx "${cur_start}" \
        --end_idx "${cur_end}" \
        --host "127.0.0.1" \
        --port "${port}" \
        > "${log_file}" 2>&1 &

    PIDS+=($!)
    PID_LABELS+=("${label}")
    LOG_FILES+=("${log_file}")

    gpu_counts[$gpu_id]=$((gpu_counts[$gpu_id] + 1))

    # 每个进程启动间隔30秒
    if [ $((slice_idx + 1)) -lt "$num_slices" ]; then
        echo "[INFO] 等待30秒后启动下一个进程..."
        sleep 30
    fi
done

# ---------- GPU 分配统计 ----------
echo ""
echo "[INFO] GPU分配统计:"
for gpu in "${GPU_LIST[@]}"; do
    echo "[INFO]   GPU ${gpu}: ${gpu_counts[$gpu]} 个进程"
done

echo ""
echo "[INFO] 共启动 ${#PIDS[@]} 个进程"
echo ""

# ---------- 等待 ----------
FAILED=0

for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    label="${PID_LABELS[$i]}"
    log_file="${LOG_FILES[$i]}"

    if wait "$pid"; then
        echo "[OK]   ${label}"
    else
        echo "[FAIL] ${label}"
        echo ""
        echo "========================================"
        echo "[ERROR] 查看日志: ${log_file}"
        echo "========================================"
        if [ -f "$log_file" ]; then
            echo "[ERROR] 日志内容:"
            cat "$log_file"
        else
            echo "[ERROR] 日志文件不存在: ${log_file}"
        fi
        echo "========================================"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ "$FAILED" -gt 0 ]; then
    echo "[WARN] ${FAILED} 个进程失败"
else
    echo "[INFO] 所有进程完成"
fi

# ---------- 聚合 ----------
# echo ""
# echo "[INFO] 聚合结果..."

# python ./examples/LIBERO-plus/eval_files/parallel_eval/aggregate_results.py \
#     --root_path "${OUTPUT_DIR}"

# echo ""
# echo "[INFO] 完成：${OUTPUT_DIR}/overall_results.json"

# bash examples/LIBERO-plus/eval_files/parallel_eval/eval_multi_gpu.sh /memory/shenboyang/outputs/train/starvla/libero_loopoft_8x3_s1/checkpoints/steps_70000_pytorch_model.pt /memory/shenboyang/outputs/eval/libero_plus_loopoft_step7_fixed_0/ 1 3  "3" libero_object 10199