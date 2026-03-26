#!/usr/bin/env python3
"""
Test script for Bagel model inference.

Usage:
    cd /path/to/Macro
    python scripts/test_bagel.py \
        --model_path /path/to/checkpoint \
        --base_model_path /path/to/base \
        --prompt "your prompt here" \
        --input_images path/to/img1.jpg path/to/img2.jpg \
        --output outputs/result.jpg \
        --seed 42
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Path setup - use relative paths
SCRIPT_DIR = Path(__file__).parent
MACRO_DIR = SCRIPT_DIR.parent
BAGEL_DIR = MACRO_DIR / "bagel"
BAGEL_SOURCE = BAGEL_DIR / "source"
INFERENCE_UTILS = MACRO_DIR / "inference_utils"
CKPTS_DIR = MACRO_DIR / "ckpts" / "Macro-Bagel"
ASSETS_DIR = MACRO_DIR / "assets" / "test_example"

for p in [str(BAGEL_SOURCE), str(INFERENCE_UTILS), str(MACRO_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

OUTPUT_DIR = MACRO_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Bagel inference test")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to Bagel checkpoint (default: ckpts/Macro-Bagel/)")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Path to BAGEL-7B-MoT base model directory")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation")
    parser.add_argument("--input_images", type=str, nargs="*", default=None,
                        help="Paths to input images")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR / "test_bagel.jpg"),
                        help="Output image path")
    parser.add_argument("--height", type=int, default=None,
                        help="Output image height (default: last input image height)")
    parser.add_argument("--width", type=int, default=None,
                        help="Output image width (default: last input image width)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    args = parser.parse_args()

    # Determine model paths
    model_path = args.model_path or str(CKPTS_DIR)
    base_model_path = args.base_model_path or os.environ.get("BAGEL_BASE_MODEL_PATH") or model_path
    
    # Load default sample if prompt/images not provided
    prompt = args.prompt
    input_images = args.input_images
    
    if prompt is None or input_images is None:
        print("Prompt or input images not provided, loading default sample...")
        default_sample_path = ASSETS_DIR / "4-5_sample1.json"
        if not default_sample_path.exists():
            raise FileNotFoundError(f"Default sample not found: {default_sample_path}")
            
        with open(default_sample_path, 'r', encoding='utf-8') as f:
            sample = json.load(f)
            
        if prompt is None:
            prompt = sample.get("prompt", "")
            
        if input_images is None:
            input_images = []
            for img_name in sample.get("input_images", []):
                img_path = ASSETS_DIR / img_name
                if not img_path.exists():
                    raise FileNotFoundError(f"Input image not found: {img_path}")
                input_images.append(str(img_path))
    
    print(f"  Prompt: {prompt[:80]}...")
    print(f"  Inputs: {len(input_images)} images")
    
    # Verify input images
    for img_path in input_images:
        if not Path(img_path).exists():
            raise FileNotFoundError(f"Input image not found: {img_path}")
    print("  All input images verified ✓")

    # Import model libraries
    print("\nImporting Bagel libraries...")
    import torch
    from PIL import Image
    from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

    from data.transforms import ImageTransform
    from data.data_utils import pil_img2rgb, add_special_tokens
    from modeling.bagel import (
        BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
        SiglipVisionConfig, SiglipVisionModel
    )
    from modeling.qwen2 import Qwen2Tokenizer
    from modeling.autoencoder import load_ae
    from inferencer import InterleaveInferencer

    # Load model
    print("\nLoading Bagel model...")
    base_model_path = Path(base_model_path)
    model_path = Path(model_path)
    
    print(f"  Base model: {base_model_path}")
    print(f"  Checkpoint: {model_path}")

    # Find ema.safetensors
    ema_path = model_path / "ema.safetensors"
    if not ema_path.exists():
        if model_path.suffix == ".safetensors":
            ema_path = model_path
        else:
            raise FileNotFoundError(f"No ema.safetensors found in {model_path}")

    # LLM config
    llm_config = Qwen2Config.from_json_file(base_model_path / "llm_config.json")
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config
    vit_config = SiglipVisionConfig.from_json_file(base_model_path / "vit_config.json")
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE
    vae_model, vae_config = load_ae(local_path=str(base_model_path / "ae.safetensors"))

    # Bagel config
    config = BagelConfig(
        visual_gen=True, visual_und=True,
        llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2, max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(str(base_model_path))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    max_input_pixels = [1048576, 1048576, 589824, 589824, 589824, 262144, 262144, 262144, 262144, 262144]
    vae_transform = ImageTransform(1024, 512, 16, max_pixels=max_input_pixels)
    vit_transform = ImageTransform(980, 224, 14, max_pixels=max_input_pixels)

    device_map = infer_auto_device_map(
        model,
        max_memory={i: "40GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    same_device_modules = [
        'language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed',
        'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed'
    ]
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        device_map[k] = first_device

    model = load_checkpoint_and_dispatch(
        model, checkpoint=str(ema_path),
        device_map=device_map, offload_buffers=True,
        dtype=torch.bfloat16, force_hooks=True,
        offload_folder="/tmp/offload_bagel_test"
    )
    model = model.eval()

    inferencer = InterleaveInferencer(
        model=model, vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform, vit_transform=vit_transform,
        new_token_ids=new_token_ids
    )

    # Run inference
    print("\nRunning inference...")
    images = [Image.open(p).convert("RGB") for p in input_images]

    # Use last input image size as default output size
    if args.height is None or args.width is None:
        if images:
            last_img = images[-1]
            default_width, default_height = last_img.size
        else:
            default_height, default_width = 768, 768
        height = args.height if args.height is not None else default_height
        width = args.width if args.width is not None else default_width
        print(f"  Output size: {width}x{height} (from last input image)")
    else:
        height = args.height
        width = args.width
        print(f"  Output size: {width}x{height}")

    # Fixed inference hyperparameters
    inference_hyper = dict(
        cfg_text_scale=4.0, cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0], timestep_shift=3.0,
        num_timesteps=50, cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    )
    
    inference_hyper["image_shapes"] = (height, width)

    torch.manual_seed(args.seed)

    output_dict = inferencer(image=images, text=prompt, **inference_hyper)
    generated_image = output_dict['image']

    # Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generated_image.save(str(out_path))
    print(f"\n✓ Output saved to: {out_path}")
    print(f"  Image size: {generated_image.size}")


if __name__ == "__main__":
    main()
