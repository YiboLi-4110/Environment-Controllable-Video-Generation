# --- 离线配置与缓存配置 ---
export WANDB_API_KEY="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" # 填入你的 Key
export WANDB_MODE="offline"               # 核心：设置为离线模式，确保训练时不断网崩溃
export WANDB_PROJECT="gravity1"    # 设置项目名称
export HF_HUB_OFFLINE=1

# for 81 frames
#   max = 6  when using accelerate_config_4_gpu_zero_stage_2.yaml
#   max = 10 when using accelerate_config_4_gpu_zero_stage_2_offload_optimizer.yaml
#   Wan2.2-TI2V-5B: 30 DiT blocks; 8 control layers. Resolution 480x832 (32-aligned); 720p = 704x1280.
#   max_timestep_boundary=1.0 for full range (single DiT); was 0.358 for high-noise-only on 14B.
CONTROLNET_NUM_LAYERS=8

CONTROL_SIGNAL_TYPE="gravity"

# Control signal encoding: --num_encode (numerical, default) or --visual_encode (arrow-based)
CONTROL_SIGNAL_ENCODING="--num_encode"

DATASET_BASE_PATH_FALLING="datasets/gravity/train/falling_4k"
DATASET_METADATA_PATH_FALLING="datasets/gravity/train/falling_4k.csv"

DATASET_BASE_PATH_SLIDING="datasets/gravity/train/sliding_4k"
DATASET_METADATA_PATH_SLIDING="datasets/gravity/train/sliding_4k.csv"

# 本地 wandb 元数据目录名；并行多卡/多任务时改为不同名称（如 wandb1、wandb2）避免冲突
WANDB_DIR="wandb"

accelerate launch \
  --config_file scripts/accelerate/accelerate_config_4_gpu_multi_gpu.yaml \
  scripts/train/train.py \
  --dataset_base_path ${DATASET_BASE_PATH_FALLING} ${DATASET_BASE_PATH_SLIDING} \
  --dataset_metadata_path ${DATASET_METADATA_PATH_FALLING} ${DATASET_METADATA_PATH_SLIDING} \
  --control_signal_type ${CONTROL_SIGNAL_TYPE} \
  --controlnet_num_layers ${CONTROLNET_NUM_LAYERS} \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 10 \
  --save_steps 250 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --trainable_models "controlnet" \
  --output_path "outputs/${CONTROL_SIGNAL_TYPE}" \
  --extra_inputs "input_image" \
  --max_timestep_boundary 1.0 \
  --min_timestep_boundary 0 \
  --max_grad_norm 1 \
  --gradient_accumulation_steps 2 \
  --dataset_num_workers 2 \
  --offline_load \
  --wandb_logging \
  --wandb_dir ${WANDB_DIR} \
  ${CONTROL_SIGNAL_ENCODING}