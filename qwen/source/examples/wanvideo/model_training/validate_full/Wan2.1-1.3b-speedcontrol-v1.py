import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1", origin_file_pattern="model.safetensors"),
    ],
)
state_dict = load_state_dict("models/train/Wan2.1-1.3b-speedcontrol-v1_full/epoch-1.safetensors")
pipe.motion_controller.load_state_dict(state_dict)

# Text-to-video
video = pipe(
    prompt="from sunset to night, a small town, light, house, river",
    negative_prompt="vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, illustration, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, cluttered background, three legs, many people in background, walking backwards",
    seed=1, tiled=True,
    motion_bucket_id=50
)
save_video(video, "video_Wan2.1-1.3b-speedcontrol-v1.mp4", fps=15, quality=5)
