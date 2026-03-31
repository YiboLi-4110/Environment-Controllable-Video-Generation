#!/usr/bin/env bash
# 与 falling_render.sh 中 RENDER_DIR_ROOT 对齐：JPEG 帧位于 .../gravity_falling/pngs/sample_xxxxxx/
export PATH="/usr/bin:$PATH"
hash -r
RENDER_DIR=".cache/gravity_falling/jpgs"
python scripts/build_synthetic_datasets/gravity_falling/falling_jpg_to_mp4.py $RENDER_DIR
