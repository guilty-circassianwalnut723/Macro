from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, ControlNetInput
from PIL import Image
import torch
from modelscope import dataset_snapshot_download


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth", origin_file_pattern="model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
pipe.load_lora(pipe.dit, "models/train/Qwen-Image-Blockwise-ControlNet-Depth_lora/epoch-4.safetensors")

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/example_image_dataset",
    local_dir="./data/example_image_dataset",
    allow_file_pattern="depth/image_1.jpg"
)

controlnet_image = Image.open("data/example_image_dataset/depth/image_1.jpg").resize((1328, 1328))

prompt = "Exquisite portrait, underwater girl, blue dress flowing, hair gently swaying, light and shadow translucent, bubbles surrounding, serene expression, intricate details, dreamy and beautiful."
image = pipe(
    prompt, seed=0,
    blockwise_controlnet_inputs=[ControlNetInput(image=controlnet_image)]
)
image.save("image.jpg")
