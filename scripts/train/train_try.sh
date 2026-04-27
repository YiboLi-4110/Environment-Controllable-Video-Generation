#!/usr/bin/env bash
# =============================================================================
# train_gravity_curriculum.sh
# Three-phase curriculum training for the Gravity ControlNet
# (Wan2.2-TI2V-5B backbone, synthetic + real data)
# =============================================================================

# --- W&B / offline config ---
export WANDB_API_KEY="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"
export WANDB_MODE="offline"
export WANDB_PROJECT="gravity_curriculum"
export HF_HUB_OFFLINE=1

# --- ControlNet architecture ---
# Wan2.2-TI2V-5B: 30 DiT blocks; 8 control layers. Resolution 480x832.
CONTROLNET_NUM_LAYERS=8
CONTROL_SIGNAL_TYPE="gravity"
CONTROL_SIGNAL_ENCODING="--num_encode"   # or --visual_encode

# --- Synthetic datasets (same as standard training) ---
DATASET_BASE_PATH_FALLING="datasets/gravity/train/falling_mini"
DATASET_METADATA_PATH_FALLING="datasets/gravity/train/falling_mini.csv"

DATASET_BASE_PATH_SLIDING="datasets/gravity/train/sliding_mini"
DATASET_METADATA_PATH_SLIDING="datasets/gravity/train/sliding_mini.csv"

# --- Real-world dataset ---
DATASET_BASE_PATH_REAL="datasets/gravity/train/real_mini"
DATASET_METADATA_PATH_REAL="datasets/gravity/train/real_mini.csv"

# --- Curriculum learning rate schedule ---
# Phase 1: linear warm-up from 0.01*lr1 → lr1 over WARMUP_STEPS, then constant lr1
# Phase 2: cosine decay lr1 → lr2
# Phase 3: constant lr2
LR1=3e-5          # peak learning rate (end of Phase-1 warm-up)
LR2=5e-6          # final learning rate (Phase-3 constant)
WARMUP_STEPS=5  # optimizer steps for Phase-1 warm-up (~10% of Phase-1 total)

# --- Curriculum epoch counts (tune as needed) ---
PHASE1_EPOCHS=2   # synthetic-only, WarmUp
PHASE2_EPOCHS=1   # mixed 50/50, cosine decay
PHASE3_EPOCHS=1   # mixed 25syn/75real, constant lr2

# --- Fixed per-epoch sample budget ---
# Every epoch across all three phases will draw exactly DATA_VOLUME samples.
# Set to "" to disable (original behaviour: Phase 1 uses full syn set;
# mixed phases use all syn + real data).
DATA_VOLUME=10

# --- Mixed data ratios (synthetic fraction, 0–1) ---
# phase2_syn_ratio=0.5 → equal mix; phase3_syn_ratio=0.25 → real-biased (recommended)
PHASE2_SYN_RATIO=0.5
PHASE3_SYN_RATIO=0.25

# --- Control signal dropout (CFG-style null conditioning) ---
# Probability of zeroing out the text prompt for a given sample.
# Synthetic data: high dropout, high signal quality.
# Real data: low dropout, scenes may need text prompt to generate.
SYN_CTRL_DROPOUT_PROB=0.3
REAL_CTRL_DROPOUT_PROB=0.1

# --- Optional: apply light ColorJitter to real data ---
# Set to "--real_color_jitter" to enable, or "" to disable.
REAL_COLOR_JITTER=""   # e.g. "--real_color_jitter"

# --- Resume from a specific phase (set to "" to train from scratch) ---
# Example: "--resume_phase 2 --controlnet_checkpoint outputs/gravity/2025-.../phase1_epoch1.safetensors"
RESUME_ARGS=""

# --- General training settings ---
WANDB_DIR="wandb_curriculum"

accelerate launch \
  --config_file scripts/accelerate/accelerate_config_4_gpu_multi_gpu.yaml \
  scripts/train/train.py \
  --dataset_base_path ${DATASET_BASE_PATH_FALLING} ${DATASET_BASE_PATH_SLIDING} \
  --dataset_metadata_path ${DATASET_METADATA_PATH_FALLING} ${DATASET_METADATA_PATH_SLIDING} \
  --dataset_base_path_real ${DATASET_BASE_PATH_REAL} \
  --dataset_metadata_path_real ${DATASET_METADATA_PATH_REAL} \
  --control_signal_type ${CONTROL_SIGNAL_TYPE} \
  --controlnet_num_layers ${CONTROLNET_NUM_LAYERS} \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --save_steps 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --trainable_models "controlnet" \
  --output_path "outputs/${CONTROL_SIGNAL_TYPE}_curriculum" \
  --extra_inputs "input_image" \
  --max_timestep_boundary 1.0 \
  --min_timestep_boundary 0 \
  --max_grad_norm 1 \
  --gradient_accumulation_steps 2 \
  --dataset_num_workers 2 \
  --weight_decay 0.01 \
  --offline_load \
  --wandb_logging \
  --wandb_dir ${WANDB_DIR} \
  --curriculum_mode \
  --lr1 ${LR1} \
  --lr2 ${LR2} \
  --warmup_steps ${WARMUP_STEPS} \
  --phase1_epochs ${PHASE1_EPOCHS} \
  --phase2_epochs ${PHASE2_EPOCHS} \
  --phase3_epochs ${PHASE3_EPOCHS} \
  --phase2_syn_ratio ${PHASE2_SYN_RATIO} \
  --phase3_syn_ratio ${PHASE3_SYN_RATIO} \
  --syn_ctrl_dropout_prob ${SYN_CTRL_DROPOUT_PROB} \
  --real_ctrl_dropout_prob ${REAL_CTRL_DROPOUT_PROB} \
  ${REAL_COLOR_JITTER} \
  ${RESUME_ARGS} \
  ${CONTROL_SIGNAL_ENCODING}
