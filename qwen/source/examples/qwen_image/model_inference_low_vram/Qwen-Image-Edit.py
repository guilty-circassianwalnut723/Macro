from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch


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
prompt = "Exquisite portrait, underwater girl, blue dress flowing, hair gently swaying, light and shadow translucent, bubbles surrounding, serene expression, intricate details, dreamy and beautiful."
input_image = pipe(prompt=prompt, seed=0, num_inference_steps=40, height=1328, width=1024)
input_image.save("image1.jpg")

prompt = "Change the dress to pink"
# edit_image_auto_resize=True: auto resize input image to match the area of 1024*1024 with the original aspect ratio
image = pipe(prompt, edit_image=input_image, seed=1, num_inference_steps=40, height=1328, width=1024, edit_image_auto_resize=True)
image.save(f"image2.jpg")

# edit_image_auto_resize=False: do not resize input image
image = pipe(prompt, edit_image=input_image, seed=1, num_inference_steps=40, height=1328, width=1024, edit_image_auto_resize=False)
image.save(f"image3.jpg")
