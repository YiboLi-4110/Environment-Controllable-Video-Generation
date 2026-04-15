#!/bin/bash
# ============================================================================
# Gravity-controlled video generation inference script
#
# Multi-GPU support: set DEVICE_IDS to use multiple GPUs in parallel.
# Each GPU processes a contiguous partition of EXAMPLE_PATHS.
#
# Example:
#   Single GPU:   DEVICE_IDS=(0)
#   Multi-GPU:    DEVICE_IDS=(0 1 2 3)
# ============================================================================

DEVICE_IDS=(0)
SEED=0
MODEL_CKPT_PATH="checkpoints/gravity/step-3000.safetensors"

# Control signal encoding: --num_encode (numerical, default) or --visual_encode (arrow-based)
CONTROL_SIGNAL_ENCODING="--num_encode"

# CSV file paths for inference examples
EXAMPLE_PATHS=(
  "datasets/gravity/test/benchmark/a.csv"
)

# ---------- Derived from DEVICE_IDS ----------
WORLD_SIZE=${#DEVICE_IDS[@]}

echo "========================================="
echo " Gravity Inference"
echo " GPUs:       ${DEVICE_IDS[*]}"
echo " World size: ${WORLD_SIZE}"
echo " Seed:       ${SEED}"
echo " Checkpoint: ${MODEL_CKPT_PATH}"
echo " Examples:   ${EXAMPLE_PATHS[*]}"
echo " Encoding:   ${CONTROL_SIGNAL_ENCODING}"
echo "========================================="

PIDS=()

for i in "${!DEVICE_IDS[@]}"; do
    GPU_ID=${DEVICE_IDS[$i]}
    echo "[Launcher] Starting process $i on GPU ${GPU_ID} ..."

    CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/inference/inference_gravity.py \
        --device_id "$i" \
        --world_size "${WORLD_SIZE}" \
        --seed "${SEED}" \
        --model_ckpt_path "${MODEL_CKPT_PATH}" \
        --example_paths "${EXAMPLE_PATHS[@]}" \
        --controlnet \
        ${CONTROL_SIGNAL_ENCODING} &

    PIDS+=($!)
done

echo "[Launcher] Waiting for ${#PIDS[@]} process(es) to finish ..."

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid"
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[Launcher] Process $pid exited with code $EXIT_CODE"
        FAILED=1
    fi
done

if [ $FAILED -eq 0 ]; then
    echo "[Launcher] All inference processes completed successfully."
else
    echo "[Launcher] Some processes failed. Check logs above."
    exit 1
fi
