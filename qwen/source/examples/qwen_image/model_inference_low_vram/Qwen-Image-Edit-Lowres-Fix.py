from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
from modelscope import snapshot_download


vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
snapshot_download("DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix", local_dir="models/DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix", allow_file_pattern="model.safetensors")
pipe.load_lora(pipe.dit, "models/DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix/model.safetensors", hotload=True)

prompt = "Exquisite portrait, underwater girl, blue dress flowing, hair gently swaying, light and shadow translucent, bubbles surrounding, serene expression, intricate details, dreamy and beautiful."
image = pipe(prompt=prompt, seed=0, num_inference_steps=40, height=1024, width=768)
image.save("image.jpg")

prompt = "Turn the dress pink"
image = image.resize((512, 384))
image = pipe(prompt, edit_image=image, seed=1, num_inference_steps=40, height=1024, width=768, edit_rope_interpolation=True, edit_image_auto_resize=False)
image.save(f"image2.jpg")
