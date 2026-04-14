#!/usr/bin/env bash
# 与 sliding_render.sh 中 RENDER_DIR_ROOT 对齐：JPEG 帧位于 .../gravity_sliding/pngs/sample_xxxxxx/
export PATH="/usr/bin:$PATH"
hash -r
RENDER_DIR=".cache/gravity_sliding/jpgs"
python scripts/build_synthetic_datasets/gravity_sliding/sliding_jpg_to_mp4.py $RENDER_DIR
