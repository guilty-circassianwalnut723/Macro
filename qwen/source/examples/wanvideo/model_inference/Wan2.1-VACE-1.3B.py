import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)


dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=["data/examples/wan/depth_video.mp4", "data/examples/wan/cat_fightning.jpg"]
)

# Depth video -> Video
control_video = VideoData("data/examples/wan/depth_video.mp4", height=480, width=832)
video = pipe(
    prompt="Two adorable orange cats put on boxing gloves and fight in a boxing ring.",
    negative_prompt="vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, illustration, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, cluttered background, three legs, many people in background, walking backwards",
    vace_video=control_video,
    seed=1, tiled=True
)
save_video(video, "video_1_Wan2.1-VACE-1.3B.mp4", fps=15, quality=5)

# Reference image -> Video
video = pipe(
    prompt="Two adorable orange cats put on boxing gloves and fight in a boxing ring.",
    negative_prompt="vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, illustration, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, cluttered background, three legs, many people in background, walking backwards",
    vace_reference_image=Image.open("data/examples/wan/cat_fightning.jpg").resize((832, 480)),
    seed=1, tiled=True
)
save_video(video, "video_2_Wan2.1-VACE-1.3B.mp4", fps=15, quality=5)

# Depth video + Reference image -> Video
video = pipe(
    prompt="Two adorable orange cats put on boxing gloves and fight in a boxing ring.",
    negative_prompt="vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, illustration, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, cluttered background, three legs, many people in background, walking backwards",
    vace_video=control_video,
    vace_reference_image=Image.open("data/examples/wan/cat_fightning.jpg").resize((832, 480)),
    seed=1, tiled=True
)
save_video(video, "video_3_Wan2.1-VACE-1.3B.mp4", fps=15, quality=5)
