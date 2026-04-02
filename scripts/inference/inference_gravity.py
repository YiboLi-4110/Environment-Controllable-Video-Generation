import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from diffsynth import save_video
from PIL import Image

import sys
for path in sys.path:
    path_to_this_file = "scripts/inference"
    if path.endswith(path_to_this_file):
        sys.path.append(path.replace(path_to_this_file, ""))
        break

from src.physical_constant.wan_video_new import WanVideoPipeline, ModelConfig
from src.physical_constant.unified_dataset import ControlSignalDataset_Falling
from src.physical_constant.utils import safe_collate, add_aesthetic_gravity_prompt_to_video
import numpy as np
import argparse
import json

CONTROLNET_NUM_LAYERS = 10
NUM_FRAMES = 81

SKIP_MODEL_LOADING_FOR_DEBUGGING_DATA = False
TORCH_DTYPE = torch.bfloat16
OFFLOAD_DEVICE = "cpu"


def split_list_across_devices_contiguous(items, world_size, device_id):
    """
    Split a list of items into contiguous chunks across devices.

    Example: [a, b, c, d, e] with world_size=2
    - device 0 gets [a, b, c]
    - device 1 gets [d, e]
    """
    n = len(items)
    base_size = n // world_size
    remainder = n % world_size
    if device_id < remainder:
        chunk_size = base_size + 1
        start_index = device_id * chunk_size
    else:
        chunk_size = base_size
        start_index = remainder * (base_size + 1) + (device_id - remainder) * base_size
    end_index = start_index + chunk_size
    return items[start_index:end_index]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0,
                        help='Logical device ID for data partitioning (0-indexed)')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total number of devices/processes (for CSV partitioning)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for video generation')
    parser.add_argument('--model_ckpt_path', type=str, required=True,
                        help='Path to the ControlNet checkpoint file')
    parser.add_argument('--example_paths', type=str, nargs='+', required=True,
                        help='Path(s) to CSV file(s) for inference')
    parser.add_argument('--controlnet', action="store_true",
                        help='Whether to use controlnet')
    return parser.parse_args()


def main(args):
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        device = "cuda:0"
    else:
        device = f"cuda:{args.device_id}"

    print(f"[Device {args.device_id}] Initialized inference:")
    print(f"[Device {args.device_id}]   - World size: {args.world_size}")
    print(f"[Device {args.device_id}]   - Device: {device}")
    print(f"[Device {args.device_id}]   - Seed: {args.seed}")

    ckpt_dir_controlnet = os.path.dirname(args.model_ckpt_path)
    step_num = os.path.basename(args.model_ckpt_path).split(".safetensors")[0].split("-")[-1]
    step_dir = os.path.join(ckpt_dir_controlnet, f"step-{step_num}-videos")
    os.makedirs(step_dir, exist_ok=True)

    if not SKIP_MODEL_LOADING_FOR_DEBUGGING_DATA:
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=TORCH_DTYPE,
            device=device,
            tokenizer_config=ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B",
                origin_file_pattern="google/*",
                path="./models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
            ),
            model_configs=[
                ModelConfig(
                    model_id="Wan-AI/Wan2.2-TI2V-5B",
                    origin_file_pattern="diffusion_pytorch_model*.safetensors",
                    offload_device="cpu",
                    path=[
                        './models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors',
                        './models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors',
                        './models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors',
                    ]
                ),
                ModelConfig(
                    model_id="Wan-AI/Wan2.2-TI2V-5B",
                    origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                    offload_device="cpu",
                    path="./models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
                ),
                ModelConfig(
                    model_id="Wan-AI/Wan2.2-TI2V-5B",
                    origin_file_pattern="Wan2.2_VAE.pth",
                    offload_device="cpu",
                    path="./models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
                ),
            ],
            controlnet=args.controlnet,
            controlnet_num_layers=CONTROLNET_NUM_LAYERS,
        )

        if args.controlnet:
            pipe.load_controlnet_weights(
                pipe.controlnet, args.model_ckpt_path, torch_dtype=TORCH_DTYPE)

        pipe.enable_vram_management()

    device_examples = split_list_across_devices_contiguous(
        args.example_paths, args.world_size, args.device_id)
    print(f"\n[Device {args.device_id}, seed {args.seed}] "
          f"Processing {len(device_examples)} out of {len(args.example_paths)} examples: {device_examples}")

    for csv_path in device_examples:
        print(f"\nProcessing CSV: {csv_path}")

        base_path = os.path.dirname(csv_path)

        dataset = ControlSignalDataset_Falling(
            base_path=base_path,
            metadata_path=csv_path,
            is_validation_dataset=True,
            num_frames=NUM_FRAMES,
            height=480,
            width=832,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=False, collate_fn=safe_collate, num_workers=0)

        for data in dataloader:
            if data is None:
                continue

            prompt              = data["prompt"]
            input_image         = data["video"]
            control_signal_video = data["control_video"]
            gravity             = data["gravity"]
            file_id             = data["file_id"]

            assert len(input_image) == 1

            fname_str  = f"step-{step_num}_{file_id}"
            fname_str += f"__gravity_{gravity:.2f}"
            fname_str += f"__seed_{args.seed}"

            print(f"\nCurrently working on: {fname_str}\n")

            fname_control_video             = os.path.join(step_dir, f"{fname_str}-control-signal.mp4")
            fname_image_condition           = os.path.join(step_dir, f"{fname_str}-image_condition.png")
            fname_output_video              = os.path.join(step_dir, f"{fname_str}.mp4")
            fname_output_video_with_prompt  = os.path.join(step_dir, f"{fname_str}-with-prompt.mp4")
            fname_text                      = os.path.join(step_dir, f"{fname_str}-text.json")

            input_image[0].save(fname_image_condition)
            input_image = input_image[0].convert("RGB")

            control_vis = ((control_signal_video.to(float).numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
            save_video(control_vis, fname_control_video, fps=15, quality=5)

            with open(fname_text, 'w') as f:
                json.dump({"text_prompt": prompt, "gravity": gravity}, f, indent=4)

            if not SKIP_MODEL_LOADING_FOR_DEBUGGING_DATA:
                video = pipe(
                    prompt=prompt,
                    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    input_image=input_image,
                    num_frames=NUM_FRAMES,
                    seed=args.seed,
                    tiled=True,
                    controlnet=args.controlnet,
                    control_signal_video=control_signal_video.to(device),
                )
                save_video(video, fname_output_video, fps=15, quality=5)

                video_array = np.asarray(
                    torch.stack([dataset.to_tensor_transform(image) for image in video]))
                video_array = np.moveaxis(video_array, 1, -1)

                video_with_prompt = add_aesthetic_gravity_prompt_to_video(
                    video_array, gravity, num_frames_with_signal=16)
                video_with_prompt = [
                    dataset.to_pil_transform(torch.from_numpy(frame).permute(2, 0, 1))
                    for frame in video_with_prompt
                ]
                save_video(video_with_prompt, fname_output_video_with_prompt, fps=15, quality=5)


if __name__ == "__main__":
    args = parse_args()
    main(args)
