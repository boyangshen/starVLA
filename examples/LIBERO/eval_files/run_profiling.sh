#!/bin/bash

cd /home/shenboyang/myProjects/starVLA

###########################################################################################
# === Configuration ===
# Modify these paths according to your environment
CONDA_ENV=star_libero
PROFILE_MODULE="qwen_vl_interface"
CHECKPOINT=/memory/shenboyang/outputs/train/starvla/libero_loopoft_8x3_s2/checkpoints/steps_10000_pytorch_model.pt
# CHECKPOINT=/memory/shenboyang/outputs/train/starvla/libero_qwen3_2boft/checkpoints/steps_100000_pytorch_model.pt
DATASET_YAML=/home/shenboyang/myProjects/starVLA/examples/LIBERO/train_files/starvla_train_libero_loopoft.yaml
NUM_SAMPLES=1000
RANDOM_SEED=42
BATCH_SIZE=1
GPU_ID=3
OUTPUT_DIR=results/profiling
OUTPUT_FILENAME=loopoft_8x3_s2_fix_7.json

# === End of configuration ===
###########################################################################################

echo "========================================"
echo "StarVLA Profiling Script"
echo "========================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Dataset YAML: ${DATASET_YAML}"
echo "Num Samples: ${NUM_SAMPLES}"
echo "Random Seed: ${RANDOM_SEED}"
echo "Batch Size: ${BATCH_SIZE}"
echo "GPU ID: ${GPU_ID}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Output Filename: ${OUTPUT_FILENAME}"
echo "========================================"

# Activate conda environment
if ! conda env list | grep -q "${CONDA_ENV}"; then
    echo "[ERROR] Conda environment '${CONDA_ENV}' not found"
    echo "Please create or modify CONDA_ENV in this script"
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run profiling
python examples/profile_flops_memory.py \
    --checkpoint ${CHECKPOINT} \
    --dataset_yaml ${DATASET_YAML} \
    --num_samples ${NUM_SAMPLES} \
    --random_seed ${RANDOM_SEED} \
    --batch_size ${BATCH_SIZE} \
    --profile_module ${PROFILE_MODULE} \
    --gpu_id ${GPU_ID} \
    --output_dir ${OUTPUT_DIR} \
    --output_filename ${OUTPUT_FILENAME}

echo ""
echo "========================================"
echo "Profiling complete! Results saved to:"
echo "${OUTPUT_DIR}/${OUTPUT_FILENAME}"
echo "========================================"
