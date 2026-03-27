import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1", origin_file_pattern="model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

# Text-to-video
video = pipe(
    prompt="Documentary photography style, a lively puppy running swiftly across a lush green meadow. The puppy has a tan coat, ears perked up, with a focused and cheerful expression. Sunlight falls on it, making its fur look especially soft and shiny. The background is an open meadow occasionally dotted with wildflowers, with a faint blue sky and a few white clouds visible in the distance. Strong sense of perspective, capturing the energy of the puppy running and the vitality of the surrounding grass. Medium shot, side-moving perspective.",
    negative_prompt="vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, illustration, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, cluttered background, three legs, many people in background, walking backwards",
    seed=1, tiled=True,
    motion_bucket_id=0
)
save_video(video, "video_slow_Wan2.1-1.3b-speedcontrol-v1.mp4", fps=15, quality=5)

video = pipe(
    prompt="Documentary photography style, a lively puppy running swiftly across a lush green meadow. The puppy has a tan coat, ears perked up, with a focused and cheerful expression. Sunlight falls on it, making its fur look especially soft and shiny. The background is an open meadow occasionally dotted with wildflowers, with a faint blue sky and a few white clouds visible in the distance. Strong sense of perspective, capturing the energy of the puppy running and the vitality of the surrounding grass. Medium shot, side-moving perspective.",
    negative_prompt="vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, illustration, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, cluttered background, three legs, many people in background, walking backwards",
    seed=1, tiled=True,
    motion_bucket_id=100
)
save_video(video, "video_fast_Wan2.1-1.3b-speedcontrol-v1.mp4", fps=15, quality=5)
