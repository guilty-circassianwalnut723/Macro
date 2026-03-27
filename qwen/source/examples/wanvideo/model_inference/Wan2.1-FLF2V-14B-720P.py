import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-FLF2V-14B-720P", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-FLF2V-14B-720P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-FLF2V-14B-720P", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-FLF2V-14B-720P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=["data/examples/wan/first_frame.jpeg", "data/examples/wan/last_frame.jpeg"]
)

# First and last frame to video
video = pipe(
    prompt="Realistic style, a girl holding a withered flower stands in a garden, the camera gradually pulls back to capture the full view of the garden.",
    negative_prompt="vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, illustration, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, cluttered background, three legs, many people in background, walking backwards",
    input_image=Image.open("data/examples/wan/first_frame.jpeg").resize((960, 960)),
    end_image=Image.open("data/examples/wan/last_frame.jpeg").resize((960, 960)),
    seed=0, tiled=True,
    height=960, width=960, num_frames=33,
    sigma_shift=16,
)
save_video(video, "video_Wan2.1-FLF2V-14B-720P.mp4", fps=15, quality=5)
