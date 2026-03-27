import torch
from diffsynth.utils.data import save_video,VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from PIL import Image
from modelscope import dataset_snapshot_download

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=["data/examples/wan/control_video.mp4", "data/examples/wan/reference_image_girl.png"]
)

# Control video
control_video = VideoData("data/examples/wan/control_video.mp4", height=832, width=576)
reference_image = Image.open("data/examples/wan/reference_image_girl.png").resize((576, 832))
video = pipe(
    prompt="Flat-style anime, a long-haired girl dances gracefully. She has delicate features, bright and expressive large eyes, and glossy black long hair. She wears a light blue T-shirt and dark blue denim shorts. The background is pink.",
    negative_prompt="vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, illustration, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, cluttered background, three legs, many people in background, walking backwards",
    control_video=control_video, reference_image=reference_image,
    height=832, width=576, num_frames=49,
    seed=1, tiled=True
)
save_video(video, "video_Wan2.2-Fun-A14B-Control.mp4", fps=15, quality=5)
