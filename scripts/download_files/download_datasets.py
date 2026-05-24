"""
Download variable_gravity_dataset from Hugging Face and organize it locally.

Target layout:
    datasets/gravity/train/
    ├── falling_4k/
    ├── falling_4k.csv
    ├── sliding_4k/
    ├── sliding_4k.csv
    ├── real_8k/
    └── real_8k.csv
"""

import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

# === HF mirror (must be set before any hub call) ===
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_BASE_URL"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# === Configuration ===
REPO_ID = "easybobLee/variable_gravity_dataset"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TARGET_DIR = PROJECT_ROOT / "datasets" / "gravity" / "train"
TEMP_DOWNLOAD_DIR = PROJECT_ROOT / "hf_temp_gravity_dataset"

DATASET_ITEMS = (
    "falling_4k",
    "sliding_4k",
    "real_8k",
)

ALLOW_PATTERNS = [f"{name}.{ext}" for name in DATASET_ITEMS for ext in ("zip", "csv")]

MAX_RETRIES = 5
RETRY_BASE_DELAY_SEC = 10
DOWNLOAD_KWARGS = {
    "repo_id": REPO_ID,
    "repo_type": "dataset",
    "local_dir": str(TEMP_DOWNLOAD_DIR),
    "local_dir_use_symlinks": False,
    "allow_patterns": ALLOW_PATTERNS,
    "etag_timeout": 60,
    "max_workers": 6,
    "resume_download": True,
}


def log(message: str) -> None:
    print(message, flush=True)


def download_snapshot_with_retry() -> Path:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log(f"⬇️  开始下载数据集（第 {attempt}/{MAX_RETRIES} 次尝试）...")
            snapshot_path = snapshot_download(**DOWNLOAD_KWARGS)
            log(f"✅  下载完成：{snapshot_path}")
            return Path(snapshot_path)
        except Exception as exc:  # noqa: BLE001 - retry on any hub/network failure
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            delay = RETRY_BASE_DELAY_SEC * attempt
            log(f"⚠️  下载失败：{exc}")
            log(f"⏳  {delay} 秒后重试...")
            time.sleep(delay)

    raise RuntimeError(f"下载失败，已重试 {MAX_RETRIES} 次") from last_error


def copy_csv_files(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    for name in DATASET_ITEMS:
        src = source_dir / f"{name}.csv"
        dst = target_dir / f"{name}.csv"
        if not src.is_file():
            raise FileNotFoundError(f"缺少 CSV 文件：{src}")
        shutil.copy2(src, dst)
        log(f"✅  CSV → {dst.relative_to(PROJECT_ROOT)}")


def extract_zip_to_dataset_folder(zip_path: Path, target_dir: Path, folder_name: str) -> None:
    if not zip_path.is_file():
        raise FileNotFoundError(f"缺少压缩包：{zip_path}")

    extract_tmp = target_dir / f"_extract_tmp_{folder_name}"
    final_dir = target_dir / folder_name

    if extract_tmp.exists():
        shutil.rmtree(extract_tmp)
    extract_tmp.mkdir(parents=True, exist_ok=True)

    try:
        log(f"📦  解压 {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_tmp)

        top_level_entries = [p for p in extract_tmp.iterdir() if p.name != "__MACOSX"]

        if len(top_level_entries) == 1 and top_level_entries[0].is_dir():
            extracted_root = top_level_entries[0]
            if final_dir.exists():
                shutil.rmtree(final_dir)
            if extracted_root.name == folder_name:
                shutil.move(str(extracted_root), str(final_dir))
            else:
                final_dir.mkdir(parents=True, exist_ok=True)
                for item in extracted_root.iterdir():
                    shutil.move(str(item), str(final_dir / item.name))
        else:
            if final_dir.exists():
                shutil.rmtree(final_dir)
            final_dir.mkdir(parents=True, exist_ok=True)
            for item in top_level_entries:
                shutil.move(str(item), str(final_dir / item.name))

        log(f"✅  解压完成 → {final_dir.relative_to(PROJECT_ROOT)}")
    finally:
        if extract_tmp.exists():
            shutil.rmtree(extract_tmp)


def cleanup_temp_paths() -> None:
    if TEMP_DOWNLOAD_DIR.exists():
        shutil.rmtree(TEMP_DOWNLOAD_DIR)
        log(f"🗑️  已清理临时下载目录：{TEMP_DOWNLOAD_DIR.relative_to(PROJECT_ROOT)}")


def main() -> None:
    log(f"📁  目标目录：{TARGET_DIR.relative_to(PROJECT_ROOT)}")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_dir = download_snapshot_with_retry()
        copy_csv_files(snapshot_dir, TARGET_DIR)

        for name in DATASET_ITEMS:
            zip_src = snapshot_dir / f"{name}.zip"
            extract_zip_to_dataset_folder(zip_src, TARGET_DIR, name)
        log("🎉  数据集下载、解压与整理完成！")
    finally:
        cleanup_temp_paths()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        log(f"❌  脚本执行失败：{exc}")
        sys.exit(1)
