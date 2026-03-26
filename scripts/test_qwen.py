#!/usr/bin/env python3
"""
Test script for Qwen-Image-Edit model inference.

Usage:
    cd /path/to/Macro
    python scripts/test_qwen.py \
        --model_base_path /path/to/ckpts \
        --prompt "your prompt here" \
        --input_images path/to/img1.jpg path/to/img2.jpg \
        --output outputs/result.jpg \
        --height 768 --width 768 \
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
QWEN_DIR = MACRO_DIR / "qwen"
QWEN_SOURCE = QWEN_DIR / "source"
INFERENCE_UTILS = MACRO_DIR / "inference_utils"
CKPTS_DIR = MACRO_DIR / "ckpts"
ASSETS_DIR = MACRO_DIR / "assets" / "test_example"

for p in [str(QWEN_SOURCE), str(INFERENCE_UTILS), str(MACRO_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

OUTPUT_DIR = MACRO_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit inference test")
    parser.add_argument("--model_base_path", type=str, default=None,
                        help="Base directory containing Qwen model files")
    parser.add_argument("--model_id", type=str, default="Macro-Qwen-Image-Edit",
                        help="Model ID / sub-path relative to model_base_path")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation")
    parser.add_argument("--input_images", type=str, nargs="*", default=None,
                        help="Paths to input images")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR / "test_qwen.jpg"))
    parser.add_argument("--height", type=int, default=None,
                        help="Output image height (default: last input image height)")
    parser.add_argument("--width", type=int, default=None,
                        help="Output image width (default: last input image width)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Determine model paths
    model_base_path = args.model_base_path or os.environ.get("DIFFSYNTH_MODEL_BASE_PATH")
    if not model_base_path:
        model_base_path = str(CKPTS_DIR)

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
    print("\nImporting Qwen-Image-Edit libraries...")
    import torch
    from PIL import Image
    from diffsynth.pipelines.qwen_image import QwenImagePipeline
    from diffsynth.core import ModelConfig

    # Load model
    print("\nLoading Qwen-Image-Edit pipeline...")
    
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = model_base_path
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    print(f"  Model base path: {model_base_path}")
    print(f"  Model ID: {args.model_id}")

    model_configs = [
        ModelConfig(model_id=args.model_id, origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id=args.model_id, origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id=args.model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ]
    
    tokenizer_config = ModelConfig(model_id=args.model_id, origin_file_pattern="tokenizer/")
    processor_config = ModelConfig(model_id=args.model_id, origin_file_pattern="processor/")

    pipeline = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        processor_config=processor_config,
        tokenizer_config=tokenizer_config,
    )

    # Prepare inputs
    edit_images = [Image.open(p).convert("RGB") for p in input_images]

    # Use last input image size as default output size
    if args.height is None or args.width is None:
        if edit_images:
            last_img = edit_images[-1]
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
    num_images = len(edit_images)
    if num_images > 0:
        idx = min(num_images - 1, len(max_input_pixels) - 1)
        idx = max(0, idx)
        max_pixels = max_input_pixels[idx]
    else:
        max_pixels = max_input_pixels[0]

    # Run inference
    print("\nRunning inference...")
    import random
    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    generated_image = pipeline(
        prompt,
        edit_image=edit_images,
        seed=args.seed,
        num_inference_steps=40,
        height=height,
        width=width,
        edit_image_auto_resize=True,
        zero_cond_t=True,
        max_input_pixels=max_pixels,
    )

    # Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generated_image.save(str(out_path))
    print(f"\n✓ Output saved to: {out_path}")
    print(f"  Image size: {generated_image.size}")


if __name__ == "__main__":
    main()
