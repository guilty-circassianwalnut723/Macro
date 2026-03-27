from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from modelscope import snapshot_download
import torch

snapshot_download("DiffSynth-Studio/Qwen-Image-Distill-LoRA", local_dir="models/DiffSynth-Studio/Qwen-Image-Distill-LoRA")
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
pipe.load_lora(pipe.dit, "models/DiffSynth-Studio/Qwen-Image-Distill-LoRA/model.safetensors")

prompt = "Exquisite portrait, underwater girl, blue dress flowing, hair gently swaying, light and shadow translucent, bubbles surrounding, serene expression, intricate details, dreamy and beautiful."
image = pipe(prompt, seed=0, num_inference_steps=15, cfg_scale=1)
image.save("image.jpg")