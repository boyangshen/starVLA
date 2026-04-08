# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

# Debug CUDA device order
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
###########################################################################################
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=/memory/shenboyang/myCache/huggingface/hub/Qwen--Qwen3-VL-2B-Instruct
config_yaml=./examples/Robocasa_tabletop/train_files/qwen3oft2b_robocasa.yaml
robocasa_data_root=/memory/shenboyang/myCache/starvla/robocase_dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/
data_mix=fourier_gr1_unified_1000_6
run_root_dir=/memory/shenboyang/outputs/train/starvla
run_id=robocasa_qwen3_2boft
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
mkdir -p ${output_dir}/logs
cp $0 ${output_dir}/

log_file=${output_dir}/logs/train_${run_id}_$(date +%Y%m%d_%H%M%S).log

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${robocasa_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.gradient_accumulation_steps 1 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 150000 \
  --trainer.is_resume true \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project loop_vla \
  --wandb_entity boyangs235-hust \
  --is_debug false \
  2>&1 | tee ${log_file}
  # --is_debug True
