import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
)
pipe.load_lora(pipe.vace, "models/train/Wan2.2-VACE-Fun-A14B_high_noise_lora/epoch-4.safetensors", alpha=1)
pipe.load_lora(pipe.vace2, "models/train/Wan2.2-VACE-Fun-A14B_low_noise_lora/epoch-4.safetensors", alpha=1)

video = VideoData("data/example_video_dataset/video1_softedge.mp4", height=480, width=832)
video = [video[i] for i in range(17)]
reference_image = VideoData("data/example_video_dataset/video1.mp4", height=480, width=832)[0]

video = pipe(
    prompt="from sunset to night, a small town, light, house, river",
    negative_prompt="vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, illustration, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, cluttered background, three legs, many people in background, walking backwards",
    vace_video=video, vace_reference_image=reference_image, num_frames=17,
    seed=1, tiled=True
)
save_video(video, "video_Wan2.2-VACE-Fun-A14B.mp4", fps=15, quality=5)
