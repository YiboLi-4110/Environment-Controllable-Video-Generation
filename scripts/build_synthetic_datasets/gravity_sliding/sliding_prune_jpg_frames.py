#!/usr/bin/env python3
"""
在已成功生成 MP4 后，删除某样本目录内除第 1 帧外的 JPEG 序列帧，保留 params.json 与 frame1.*。
用于大规模流水线节省磁盘。
"""
import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from sliding_jpg_to_mp4 import glob_sample_dir  # noqa: E402


def _frame_index(path):
    m = re.search(r"frame(\d+)", os.path.basename(path), re.I)
    return int(m.group(1)) if m else -1


def prune_sample_dir(frames_dir, keep_frame_index=1):
    """
    删除 frames_dir 下除第 keep_frame_index 帧外的 frame*.jpg / frame*.jpeg。
    不删除 params.json 及其他文件。
    """
    if not os.path.isdir(frames_dir):
        return False

    to_delete = []
    for ext in (".jpg", ".jpeg"):
        for p in Path(frames_dir).glob(f"frame*{ext}"):
            if _frame_index(str(p)) != keep_frame_index:
                to_delete.append(str(p))

    if not to_delete:
        print(f"No frames to prune in {frames_dir}")
        return True

    try:
        subprocess.run(["rm"] + to_delete, check=True)
        print(f"Pruned {len(to_delete)} frame file(s) in {frames_dir} (kept frame {keep_frame_index:04d})")
    except subprocess.CalledProcessError as e:
        print(f"Error removing files: {e}")
        return False

    return True


def process_sample_range(base_dir, start_id, end_id, keep_frame_index=1):
    ok = 0
    bad = 0
    for sid in range(int(start_id), int(end_id) + 1):
        d = glob_sample_dir(base_dir, sid)
        if not d:
            print(f"Prune: no directory for sample_id={sid}")
            bad += 1
            continue
        if prune_sample_dir(d, keep_frame_index=keep_frame_index):
            ok += 1
        else:
            bad += 1
    print(
        f"Prune range [{start_id}, {end_id}]: {ok} directory(ies) ok, {bad} failed or missing"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove all JPEG frames except the first in each sample_*_g_* directory"
    )
    parser.add_argument(
        "base_dir",
        help="与 sliding_render 的 RENDER_DIR_ROOT 一致（含 sample_*_g_* 子目录）",
    )
    parser.add_argument(
        "--sample-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        required=True,
        help="只处理 sample_id ∈ [START, END] 的目录",
    )
    parser.add_argument(
        "--keep-frame",
        type=int,
        default=1,
        help="保留的帧序号（默认 1 即 frame1）",
    )
    args = parser.parse_args()
    lo, hi = args.sample_range
    process_sample_range(args.base_dir, lo, hi, keep_frame_index=args.keep_frame)