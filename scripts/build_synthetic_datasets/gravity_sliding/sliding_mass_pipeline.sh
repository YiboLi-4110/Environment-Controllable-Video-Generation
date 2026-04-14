#!/usr/bin/env bash
# 大规模 gravity_sliding：分批渲染 → 转 MP4 → 仅保留每样本第 1 帧 JPEG + params.json
# 请在仓库根目录执行本脚本（与 sliding_render 的 .cache 相对路径一致）。

set -euo pipefail

export PATH="/usr/bin:$PATH"
hash -r

BLENDER_BIN="${BLENDER_BIN:-blender}"
# 与 sliding_render.sh 一致：JPEG 输出根目录
RENDER_DIR=".cache/gravity_sliding/jpgs"
# blend文件
SOURCE_BLEND="scripts/build_synthetic_datasets/gravity_sliding/sliding.blend"
# 渲染脚本
RENDER_SCRIPT="scripts/build_synthetic_datasets/gravity_sliding/sliding_render.py"

# 总样本数、每批样本数、并行 Blender 进程数（通常等于 GPU 数）
TOTAL=1000
BATCH=40
NUM_GPUS=4

python scripts/build_synthetic_datasets/gravity_sliding/sliding_mass_pipeline.py \
  --total "$TOTAL" \
  --batch-size "$BATCH" \
  --num-gpus "$NUM_GPUS" \
  --render-dir "$RENDER_DIR" \
  --blend "$SOURCE_BLEND" \
  --render-script "$RENDER_SCRIPT" \
  --blender-bin "$BLENDER_BIN"
