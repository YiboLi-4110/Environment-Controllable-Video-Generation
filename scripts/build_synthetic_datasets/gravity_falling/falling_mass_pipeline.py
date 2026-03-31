#!/usr/bin/env python3
"""
大规模 gravity_falling 流水线：按批渲染 JPEG → 转 MP4 → 删除除第 1 帧外的图片，循环直到总样本数完成。
sample_id 从 0 连续递增（与 falling_render.sh 一致）。

须在仓库/项目根目录下执行本脚本，以便 Blender 的 cwd 与 falling_render 中 .cache 等相对路径一致。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from falling_jpg_to_mp4 import create_mp4_from_frames, glob_sample_dir  # noqa: E402
from falling_prune_jpg_frames import prune_sample_dir  # noqa: E402


def run_render_batch(
    cwd: Path,
    blend_file: Path,
    render_py: Path,
    blender_bin: str,
    render_dir: Path,
    start_id: int,
    end_id: int,
    num_gpus: int,
) -> list[int]:
    """
    在 [start_id, end_id] 上启动 Blender 渲染；多卡时按 CUDA_VISIBLE_DEVICES 轮询。
    num_gpus=0 时串行且不改 CUDA_VISIBLE_DEVICES。
    返回失败的 sample_id 列表。
    """
    failed: list[int] = []
    pending: list[tuple[subprocess.Popen, int]] = []
    wave = num_gpus if num_gpus > 0 else 1

    def wait_all():
        nonlocal pending
        for p, sid in pending:
            code = p.wait()
            if code != 0:
                print(f"Blender failed sample_id={sid} exit={code}")
                failed.append(sid)
        pending = []

    for sid in range(start_id, end_id + 1):
        env = os.environ.copy()
        if num_gpus > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(sid % num_gpus)
        cmd = [
            blender_bin,
            "--enable-autoexec",
            "-b",
            str(blend_file),
            "-P",
            str(render_py),
            "--",
            "--sample_id",
            str(sid),
            "--render_dir",
            str(render_dir),
        ]
        cuda_show = env.get("CUDA_VISIBLE_DEVICES", "(default)")
        print(f"Launch: sample_id={sid} CUDA_VISIBLE_DEVICES={cuda_show}")
        p = subprocess.Popen(cmd, cwd=str(cwd), env=env)
        pending.append((p, sid))
        if len(pending) == wave or sid == end_id:
            wait_all()

    return failed


def mp4_and_prune_range(render_dir: Path, start_id: int, end_id: int) -> list[int]:
    """对区间内每个样本：转 MP4，成功则删帧。返回 MP4 或删帧失败的 sample_id。"""
    bad: list[int] = []
    for sid in range(start_id, end_id + 1):
        d = glob_sample_dir(str(render_dir), sid)
        if not d:
            print(f"Missing render directory for sample_id={sid}")
            bad.append(sid)
            continue
        if create_mp4_from_frames(d):
            if not prune_sample_dir(d):
                bad.append(sid)
        else:
            bad.append(sid)
    return bad


def main():
    parser = argparse.ArgumentParser(
        description="Batch render → MP4 → prune frames for gravity_falling"
    )
    parser.add_argument("--total", type=int, required=True, help="总样本数（如 1000，则 id 0..999）")
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="每批完成的样本数（如 100；每批内先全部渲染再 MP4+删帧）",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="并行 Blender 进程数；与 CUDA_VISIBLE_DEVICES 0..num_gpus-1 轮询。设为 0 则始终使用当前可见 GPU",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        required=True,
        help="falling_render 输出根目录（其下为 sample_<id>_g_*）",
    )
    parser.add_argument(
        "--blend",
        type=Path,
        default=Path("scripts/build_synthetic_datasets/gravity_falling/falling.blend"),
        help="相对当前工作目录的 .blend 路径（请在项目根目录执行）",
    )
    parser.add_argument(
        "--render-script",
        type=Path,
        default=Path("scripts/build_synthetic_datasets/gravity_falling/falling_render.py"),
        help="相对当前工作目录的 falling_render.py（请在项目根目录执行）",
    )
    parser.add_argument(
        "--blender-bin",
        default=os.environ.get("BLENDER_BIN", "blender"),
        help="Blender 可执行文件（可用环境变量 BLENDER_BIN）",
    )
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    blend_file = (project_root / args.blend).resolve()
    render_py = (project_root / args.render_script).resolve()
    render_dir = args.render_dir.resolve()
    render_dir.mkdir(parents=True, exist_ok=True)

    total = args.total
    batch = args.batch_size
    if total < 1 or batch < 1:
        print("total and batch-size must be >= 1", file=sys.stderr)
        sys.exit(1)

    num_gpus = max(0, args.num_gpus)

    all_render_fail: list[int] = []
    all_post_fail: list[int] = []

    b = 0
    while b < total:
        start_id = b
        end_id = min(b + batch - 1, total - 1)
        print(
            f"\n=== Batch sample_id [{start_id}, {end_id}] "
            f"({end_id - start_id + 1} samples) ===\n"
        )

        rf = run_render_batch(
            cwd=project_root,
            blend_file=blend_file,
            render_py=render_py,
            blender_bin=args.blender_bin,
            render_dir=render_dir,
            start_id=start_id,
            end_id=end_id,
            num_gpus=num_gpus,
        )
        all_render_fail.extend(rf)

        pf = mp4_and_prune_range(render_dir, start_id, end_id)
        all_post_fail.extend(pf)

        b += batch

    if all_render_fail or all_post_fail:
        print(
            f"\nDone with errors. Render failures: {sorted(set(all_render_fail))}; "
            f"MP4/prune failures: {sorted(set(all_post_fail))}"
        )
        sys.exit(1)
    print("\nAll batches finished successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
