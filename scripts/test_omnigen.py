#!/usr/bin/env python3
"""
Test script for OmniGen2 model inference.

Usage:
    cd /path/to/Macro
    python scripts/test_omnigen.py \
        --model_path /path/to/omnigen2 \
        --prompt "your prompt here" \
        --input_images path/to/img1.jpg path/to/img2.jpg \
        --output outputs/result.jpg \
        --height 1024 --width 1024 \
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
OMNIGEN_DIR = MACRO_DIR / "omnigen"
OMNIGEN_SOURCE = OMNIGEN_DIR / "source"
INFERENCE_UTILS = MACRO_DIR / "inference_utils"
CKPTS_DIR = MACRO_DIR / "ckpts" / "Macro-OmniGen2"
ASSETS_DIR = MACRO_DIR / "assets" / "test_example"

for p in [str(OMNIGEN_SOURCE), str(INFERENCE_UTILS), str(MACRO_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

OUTPUT_DIR = MACRO_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="OmniGen2 inference test")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to OmniGen2 base model directory")
    parser.add_argument("--transformer_path", type=str, default=None,
                        help="Path to fine-tuned transformer checkpoint")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation")
    parser.add_argument("--input_images", type=str, nargs="*", default=None,
                        help="Paths to input images")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR / "test_omnigen.jpg"))
    parser.add_argument("--height", type=int, default=None,
                        help="Output image height (default: last input image height)")
    parser.add_argument("--width", type=int, default=None,
                        help="Output image width (default: last input image width)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable_model_cpu_offload", action="store_true")
    args = parser.parse_args()

    # Determine model paths
    model_path = args.model_path or os.environ.get("OMNIGEN_MODEL_PATH")
    if not model_path:
        model_path = str(CKPTS_DIR)

    # Check for fine-tuned transformer
    transformer_path = args.transformer_path
    if transformer_path is None:
        default_transformer = CKPTS_DIR / "transformer"
        if default_transformer.exists():
            transformer_path = str(default_transformer)
            print(f"Using fine-tuned transformer from: {transformer_path}")

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

    for img_path in input_images:
        if not Path(img_path).exists():
            raise FileNotFoundError(f"Input image not found: {img_path}")
    print("  All input images verified ✓")

    # Import libraries
    print("\nImporting OmniGen2 libraries...")
    import torch
    from PIL import Image, ImageOps
    from accelerate import Accelerator

    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
    from omnigen2.models.transformers import OmniGen2Transformer2DModel
    from omnigen2.models.transformers import transformer_omnigen2
    
    # Register module to fix diffusers loading
    sys.modules['transformer_omnigen2'] = transformer_omnigen2

    # Load model
    print("\nLoading OmniGen2 pipeline...")
    accelerator = Accelerator(mixed_precision='bf16')
    weight_dtype = torch.bfloat16

    pipeline = OmniGen2Pipeline.from_pretrained(
        model_path,
        torch_dtype=weight_dtype,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )

    if transformer_path:
        print(f"  Loading fine-tuned transformer from: {transformer_path}")
        transformer = OmniGen2Transformer2DModel.from_pretrained(
            transformer_path,
            torch_dtype=weight_dtype,
            max_ref_images=10
        )
        pipeline.transformer = transformer

    if args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(accelerator.device)

    # Prepare inputs
    pil_images = [
        ImageOps.exif_transpose(Image.open(p).convert("RGB"))
        for p in input_images
    ]

    # Use last input image size as default output size
    if args.height is None or args.width is None:
        if pil_images:
            last_img = pil_images[-1]
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
    
    # Dynamic resolution: max_pixels based on number of input images
    max_input_pixels = [1048576, 1048576, 589824, 589824, 589824, 262144, 262144, 262144, 262144, 262144]
    num_images = len(pil_images)
    if num_images > 0:
        idx = min(num_images - 1, len(max_input_pixels) - 1)
        idx = max(0, idx)
        max_pixels = max_input_pixels[idx]
    else:
        max_pixels = max_input_pixels[0]
    
    # Fixed inference hyperparameters
    negative_prompt = (
        "(((deformed))), blurry, over saturation, bad anatomy, disfigured, "
        "poorly drawn face, mutation, mutated, (extra_limb), (ugly), "
        "(poorly drawn hands), fused fingers, messy drawing, broken legs"
    )

    # Run inference
    print("\nRunning inference...")
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    results = pipeline(
        prompt=prompt,
        input_images=pil_images,
        width=width,
        height=height,
        num_inference_steps=50,
        max_sequence_length=1024,
        text_guidance_scale=5.0,
        image_guidance_scale=2.0,
        cfg_range=(0.0, 1.0),
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        generator=generator,
        output_type="pil",
        max_pixels=max_pixels,
    )

    generated_image = results.images[0]

    # Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generated_image.save(str(out_path))
    print(f"\n✓ Output saved to: {out_path}")
    print(f"  Image size: {generated_image.size}")


if __name__ == "__main__":
    main()
