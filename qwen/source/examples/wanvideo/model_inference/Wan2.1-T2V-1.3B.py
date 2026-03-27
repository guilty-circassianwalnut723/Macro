import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

# Text-to-video
video = pipe(
    prompt="Documentary photography style, a lively puppy running swiftly across a lush green meadow. The puppy has a tan coat, ears perked up, with a focused and cheerful expression. Sunlight falls on it, making its fur look especially soft and shiny. The background is an open meadow occasionally dotted with wildflowers, with a faint blue sky and a few white clouds visible in the distance. Strong sense of perspective, capturing the energy of the puppy running and the vitality of the surrounding grass. Medium shot, side-moving perspective.",
    negative_prompt="vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, illustration, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, cluttered background, three legs, many people in background, walking backwards",
    seed=0, tiled=True,
)
save_video(video, "video_1_Wan2.1-T2V-1.3B.mp4", fps=15, quality=5)

# Video-to-video
video = VideoData("video_1_Wan2.1-T2V-1.3B.mp4", height=480, width=832)
video = pipe(
    prompt="Documentary photography style, a lively puppy wearing black sunglasses running swiftly across a lush green meadow. The puppy has a tan coat, wearing black sunglasses, ears perked up, with a focused and cheerful expression. Sunlight falls on it, making its fur look especially soft and shiny. The background is an open meadow occasionally dotted with wildflowers, with a faint blue sky and a few white clouds visible in the distance. Strong sense of perspective, capturing the energy of the puppy running and the vitality of the surrounding grass. Medium shot, side-moving perspective.",
    negative_prompt="vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, illustration, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, cluttered background, three legs, many people in background, walking backwards",
    input_video=video, denoising_strength=0.7,
    seed=1, tiled=True
)
save_video(video, "video_2_Wan2.1-T2V-1.3B.mp4", fps=15, quality=5)
