import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 让可能触发的HF请求走国内镜像

import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

# 1. 加载 Wan2.2-TI2V-5B 权重
# 如果本地没有，对应权重会通过 ModelScope 自动下载
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # 主扩散模型（单一专家，不再区分 high / low noise）
        ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
            offload_device="cpu",
        ),
        # 文本编码器，仍然是 umt5-xxl
        ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            offload_device="cpu",
        ),
        # Wan2.2 的高压缩 VAE
        ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="Wan2.2_VAE.pth",
            offload_device="cpu",
        ),
    ],
)

# 启用显存管理
pipe.enable_vram_management()

# 2. 准备输入图像和提示词
# 官方 TI2V-5B 的 720p 配置是 1280x704，这里也用这个分辨率
input_image = Image.open("diffsynth/cat_fightning.jpg").resize((1280, 704))
prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

# 3. 推理：根据图片 + 文本生成视频
video = pipe(
    prompt=prompt,
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0,
    # 720p 分辨率：与上面的 resize 保持一致
    height=704,
    width=1280,
    # 帧数可以沿用 I2V demo 的 81，也可以根据需要调整
    num_frames=81,
    # TI2V 模型同样建议开启 tiling，节省显存
    tiled=True,
    input_image=input_image,
)

# 4. 保存视频
# 官方 TI2V-5B 是 24fps，可以改回 15fps
save_video(video, "temp/cat_fightning_ti2v_5b.mp4", fps=24, quality=5)