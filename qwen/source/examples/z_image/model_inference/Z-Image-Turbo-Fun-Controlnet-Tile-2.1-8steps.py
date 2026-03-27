from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig, ControlNetInput
from modelscope import dataset_snapshot_download
from PIL import Image
import torch


pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1", origin_file_pattern="Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern="data/examples/upscale/low_res.png"
)
controlnet_image = Image.open("data/examples/upscale/low_res.png").resize((1024, 1024))
prompt = "An outdoor portrait photo full of urban energy. The subject is a young male presenting a fashionable and confident image. The person has carefully styled short hair, trimmed shorter on the sides with some length on top, showing the popular undercut style. He wears a pair of stylish light-colored sunglasses or transparent-framed glasses, adding a trendy touch to the overall look. His face shows a gentle and friendly smile, relaxed and natural, giving a sunny and cheerful impression. He wears a classic denim jacket, a timeless piece that shows a casual yet stylish dressing style. The blue tone of the denim jacket is very harmonious with the overall atmosphere, with a glimpse of inner clothing visible at the collar. The background is a typical urban street scene, with blurred buildings, streets and pedestrians creating a bustling city atmosphere. The background is appropriately blurred to make the subject more prominent. The lighting is bright and soft, possibly natural daylight, bringing a fresh and clear visual effect to the photo. The entire photo is professionally composed with well-controlled depth of field, perfectly capturing a moment full of vitality and confidence of a modern urban young person, showing a positive and upward attitude towards life."
image = pipe(prompt=prompt, seed=0, height=1024, width=1024, controlnet_inputs=[ControlNetInput(image=controlnet_image, scale=0.7)])
image.save("image_tile.jpg")
