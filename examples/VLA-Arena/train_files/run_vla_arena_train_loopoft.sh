# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Debug CUDA device order
export CUDA_VISIBLE_DEVICES=1,3
# export NCCL_DEBUG=INFO
# ###########################################################################################
Framework_name=LoopOFT
freeze_module_list=""

config_yaml=./examples/VLA-Arena/train_files/starvla_cotrain_vla_arena_loopoft.yaml
vla_arena_data_root=/memory/shenboyang/myCache/starvla/vla-Arena/vla_arena/
data_mix=vla_arena_L0_S
run_root_dir=/memory/shenboyang/outputs/train/starvla
run_id=vla_arena_loopoft_8x3_s1
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
mkdir -p ${output_dir}/logs
cp $0 ${output_dir}/

log_file=${output_dir}/logs/train_${run_id}_$(date +%Y%m%d_%H%M%S).log


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 2 \
  --main_process_port 26791 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.num_preserved_layers 3 \
  --framework.qwenvl.num_loop 8 \
  --framework.use_entropy_loss false \
  --framework.halting_entropy_weight 0.0001 \
  --framework.use_loop_effectiveness_loss true \
  --framework.loop_effectiveness_weight 1.0 \
  --framework.use_cosine_loss false \
  --framework.cosine_loss_weight 0.001 \
  --framework.max_loss_updates 600 \
  --framework.reduce_in_full_precision true \
  --datasets.vla_data.data_root_dir ${vla_arena_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.gradient_accumulation_steps 1\
  --trainer.max_train_steps 60000 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.is_resume true \
  --trainer.save_interval 6000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project loop_vla \
  --wandb_entity boyangs235-hust \
  --is_debug false \
  2>&1 | tee ${log_file}

  # --trainer.lr_scheduler_type constant \
  # --trainer.pretrained_checkpoint /memory/shenboyang/outputs/train/starvla/libero_loopgroot_12x2_s1/checkpoints/steps_20000_pytorch_model.pt \
  # --trainer.freeze_modules ${freeze_module_list} \
