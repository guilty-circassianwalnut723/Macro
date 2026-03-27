import torch
from diffsynth.utils.data import save_video,VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from PIL import Image
from modelscope import dataset_snapshot_download

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control-Camera", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control-Camera", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control-Camera", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control-Camera", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)


dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=f"data/examples/wan/input_image.jpg"
)
input_image = Image.open("data/examples/wan/input_image.jpg")

video = pipe(
    prompt="A small boat bravely sails through the waves. The deep blue sea surges with waves, white foam splashing against the hull, but the boat presses forward fearlessly toward the horizon. Sunlight glitters on the water's surface with golden brilliance, adding a touch of warmth to this magnificent scene. The camera zooms in to reveal the flag on the boat fluttering in the wind, symbolizing an indomitable spirit and the courage of adventure. The footage is full of power and inspiration, showcasing fearlessness and determination when facing challenges.",
    negative_prompt="vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, illustration, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, cluttered background, three legs, many people in background, walking backwards",
    seed=0, tiled=True,
    input_image=input_image,
    camera_control_direction="Left", camera_control_speed=0.01,
)
save_video(video, "video_left_Wan2.2-Fun-A14B-Control-Camera.mp4", fps=15, quality=5)

video = pipe(
    prompt="A small boat bravely sails through the waves. The deep blue sea surges with waves, white foam splashing against the hull, but the boat presses forward fearlessly toward the horizon. Sunlight glitters on the water's surface with golden brilliance, adding a touch of warmth to this magnificent scene. The camera zooms in to reveal the flag on the boat fluttering in the wind, symbolizing an indomitable spirit and the courage of adventure. The footage is full of power and inspiration, showcasing fearlessness and determination when facing challenges.",
    negative_prompt="vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, illustration, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, cluttered background, three legs, many people in background, walking backwards",
    seed=0, tiled=True,
    input_image=input_image,
    camera_control_direction="Up", camera_control_speed=0.01,
)
save_video(video, "video_up_Wan2.2-Fun-A14B-Control-Camera.mp4", fps=15, quality=5)
