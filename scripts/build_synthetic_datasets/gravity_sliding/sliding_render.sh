#!/bin/bash
# sliding_render.sh

# 配置项
BLENDER_BIN="${BLENDER_BIN:-blender}" # 对应你的 4.4.0 路径
RENDER_DIR_ROOT=".cache/gravity_sliding/jpgs"
NUM_GPUS=1
TOTAL_SAMPLES=1  

# 创建输出目录
mkdir -p "$RENDER_DIR_ROOT"

SOURCE_BLEND="scripts/build_synthetic_datasets/gravity_sliding/sliding.blend"
PYTHON_SCRIPT="scripts/build_synthetic_datasets/gravity_sliding/sliding_render.py"

echo "Launching Gravity Dataset Pipeline on $NUM_GPUS GPUs..."

job_idx=0
while [ "$job_idx" -lt "$TOTAL_SAMPLES" ]; do
    gpu_id=$(( job_idx % NUM_GPUS ))
    
    # 每个视频分配独立的子目录，防止多进程写入冲突
    SAMPLE_ID=$(printf "%d" $job_idx)
    
    # 使用 CUDA_VISIBLE_DEVICES 隔离 GPU
    # 限制每进程使用 8 个线程 (32核/4卡)
    # 注意：-s/-e/-a 必须出现在 Blender 侧；若写在「--」之后只会进 Python 的 sys.argv，
    # Blender 不会执行动画渲染。序列帧由 sliding_render.py 内 bpy.ops.render.render(animation=True) 完成。
    CUDA_VISIBLE_DEVICES="$gpu_id" "$BLENDER_BIN" \
        --enable-autoexec \
        -b "${SOURCE_BLEND}" \
        -P "$PYTHON_SCRIPT" \
        -- \
        --sample_id "$SAMPLE_ID" \
        --render_dir "$RENDER_DIR_ROOT" &

    job_idx=$(( job_idx + 1 ))

    # 并发控制：每启动 NUM_GPUS 个任务等待一波
    if [ $(( job_idx % NUM_GPUS )) -eq 0 ]; then
        wait
    fi
done

wait
echo "All gravity datasets generated."