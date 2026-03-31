#!/usr/bin/env python3
"""
将 gravity_falling 渲染目录下的 JPEG 序列合成为 MP4（与 falling_render.py 输出一致）。
目录布局须与 falling_render.py / falling_render.sh 一致：
  <RENDER_DIR_ROOT>/sample_<id>/frame0001.jpg ... params.json
视频输出到与帧目录同级的 videos/ 目录（见 create_mp4_from_frames 中的路径推导）。
"""
import os
import json
import subprocess
import glob
import argparse
import math


def glob_sample_dir(base_dir, sample_id):
    """
    与 falling_render 输出目录一致：sample_<id>_g_<gravity>.2f
    返回匹配到的单个目录路径，若无或多个则返回 None / 取字典序第一个并打印警告。
    """
    pattern = os.path.join(base_dir, f"sample_{int(sample_id)}_g_*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        print(
            f"Warning: multiple dirs for sample_id={sample_id}, using {matches[0]}"
        )
    return matches[0]


def create_mp4_from_frames(frames_dir):
    """从单个子目录（含 params.json 与 frame*.jpg）生成 MP4。"""
    if not os.path.isdir(frames_dir):
        return False

    params_file = os.path.join(frames_dir, "params.json")
    if not os.path.exists(params_file):
        return False

    try:
        with open(params_file, "r") as f:
            params = json.load(f)
        sample_id = str(params.get("sample_id", "unknown"))
        gravity_z = float(params.get("gravity_z", 0.0))
        fps = int(params.get("fps", 16))
        total_frames = int(params.get("total_frames", 81))
        frame_ext = str(params.get("frame_ext", "jpg")).lower().lstrip(".")
    except Exception as e:
        print(f"Error reading params: {e}")
        return False

    if frame_ext not in ("jpg", "jpeg"):
        print(
            f"Expected frame_ext jpg/jpeg in params.json, got {frame_ext!r}; "
            f"this script only supports JPEG frames."
        )
        return False

    output_filename = (
        f"sample_{sample_id}_g_{gravity_z:.2f}.mp4"
    )
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(frames_dir)), "videos"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    jpg_glob = os.path.join(frames_dir, "frame*.jpg")
    jpeg_glob = os.path.join(frames_dir, "frame*.jpeg")
    jpg_files = sorted(glob.glob(jpg_glob) + glob.glob(jpeg_glob))
    jpg_files = sorted(set(jpg_files))
    if not jpg_files:
        print(f"No JPEG frames in {frames_dir}")
        return False

    if len(jpg_files) < total_frames:
        print(
            f"Expected {total_frames} frames, got {len(jpg_files)} in {frames_dir}"
        )
        return False

    if os.path.isfile(output_path):
        print(f"Skip (exists): {output_path}")
        return False

    # ffmpeg 序列需单一扩展名；优先 .jpg（与 falling_render 默认一致）
    if glob.glob(jpg_glob):
        frame_pattern = os.path.join(frames_dir, "frame%04d.jpg")
    else:
        frame_pattern = os.path.join(frames_dir, "frame%04d.jpeg")

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-start_number",
            "1",
            "-i",
            frame_pattern,
            "-frames:v",
            str(total_frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"Created {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e}")
        return False


def process_all_render_dirs(base_dir):
    """处理 base_dir 下每个 sample_* 子目录（base_dir 即 falling_render.sh 中的 RENDER_DIR_ROOT/jpgs）。"""
    processed = 0
    skipped = 0

    try:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        num_jobs = int(os.environ.get("NUM_SLURM_JOBS", 100))
        dirs = sorted(os.listdir(base_dir))
        print(dirs)
        lower = math.floor(task_id * len(dirs) / num_jobs)
        upper = math.floor((task_id + 1) * len(dirs) / num_jobs)
        dirs = dirs[lower:upper]
        print(dirs)
    except Exception:
        dirs = sorted(os.listdir(base_dir))

    for d in dirs:
        sample_dir = os.path.join(base_dir, d)
        if not os.path.isdir(sample_dir):
            continue
        if not d.startswith("sample_"):
            continue
        if create_mp4_from_frames(sample_dir):
            processed += 1
        else:
            skipped += 1

    print(f"Done: {processed} mp4(s) created, {skipped} directory(ies) skipped")


def process_sample_range(base_dir, start_id, end_id):
    """只处理 sample_id ∈ [start_id, end_id] 的目录（用于分批流水线）。"""
    processed = 0
    skipped = 0
    for sid in range(int(start_id), int(end_id) + 1):
        d = glob_sample_dir(base_dir, sid)
        if not d:
            print(f"No sample directory for sample_id={sid} under {base_dir}")
            skipped += 1
            continue
        if create_mp4_from_frames(d):
            processed += 1
        else:
            skipped += 1
    print(
        f"Sample range [{start_id}, {end_id}]: {processed} mp4(s) created, "
        f"{skipped} skipped or failed"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create MP4s from gravity_falling Blender JPEG frame sequences"
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=None,
        help="帧根目录（与 falling_render.sh 中 RENDER_DIR_ROOT 一致，通常为 .../gravity_falling/jpgs）",
    )
    parser.add_argument(
        "--dir",
        "-d",
        dest="single_dir",
        help="只处理单个样本目录（例如 .../jpgs/sample_000000）",
    )
    parser.add_argument(
        "--sample-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        dest="sample_range",
        help="只处理 sample_id 在 [START, END] 内的子目录（含端点），用于大规模分批流水线",
    )

    args = parser.parse_args()

    if args.sample_range is not None and not args.base_dir:
        parser.error("--sample-range 需要同时提供 base_dir（JPEG 根目录）")

    if args.single_dir:
        ok = create_mp4_from_frames(args.single_dir)
        raise SystemExit(0 if ok else 1)
    if args.base_dir and args.sample_range is not None:
        lo, hi = args.sample_range
        process_sample_range(args.base_dir, lo, hi)
        raise SystemExit(0)
    if args.base_dir:
        process_all_render_dirs(args.base_dir)
    else:
        print(
            "Usage: falling_jpg_to_mp4.py <base_dir>   "
            "or  falling_jpg_to_mp4.py -d <sample_dir>   "
            "or  falling_jpg_to_mp4.py <base_dir> --sample-range START END"
        )
        parser.print_help()
        raise SystemExit(1)
