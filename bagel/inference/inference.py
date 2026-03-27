#!/usr/bin/env python3
"""
Inference script

Load data from the filter directory, run inference with the Bagel model, and save results.
Supports multi-GPU parallel inference and checkpoint resume.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import random
import numpy as np

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

# Add bagel source path
SCRIPT_DIR = Path(__file__).parent
BAGEL_DIR = SCRIPT_DIR.parent
MACRO_DIR = BAGEL_DIR.parent
BAGEL_SOURCE_PATH = BAGEL_DIR / "source"
if str(BAGEL_SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(BAGEL_SOURCE_PATH))

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer

# Import local utils
from utils import (
    load_data_for_task,  # Unified data loading interface
    SUPPORTED_TASKS,
    IMAGE_NUM_CATEGORIES,
    check_sample_exists,
    save_sample
)

# Fixed base model path (contains llm_config.json, vit_config.json, ae.safetensors, etc.)
BASE_MODEL_PATH = Path(os.environ.get("BAGEL_BASE_MODEL_PATH", "/path/to/BAGEL-7B-MoT"))


def load_model(
    model_path: str,
    base_model_path: Optional[str] = None,
    vae_transform_size: tuple = (1024, 512),
    vit_transform_size: tuple = (980, 224),
    max_mem_per_gpu: str = "40GiB",
    offload_folder: str = "/tmp/offload",
    device_id: Optional[int] = None,
    max_input_pixels = None,  # Can be int or list, used to dynamically adjust input image resolution
):
    """
    Load the Bagel model onto the specified GPU (no sharding).

    Args:
        model_path: Path containing ema.safetensors (can be a directory or a safetensors file path)
        base_model_path: Base model path (contains llm_config.json, vit_config.json, ae.safetensors).
                        If None, use the default BASE_MODEL_PATH.
        vae_transform_size: VAE transform size (height, width)
        vit_transform_size: ViT transform size (height, width)
        max_mem_per_gpu: Maximum memory per GPU
        offload_folder: Offload folder path
        device_id: Specified GPU device ID (0-based); if None, use accelerate auto-assignment
        max_input_pixels: Maximum input pixel count; can be int or list for dynamic resolution adjustment

    Returns:
        (model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)
    """
    model_path = Path(model_path)

    # Determine base model path
    if base_model_path is None:
        base_model_path = BASE_MODEL_PATH
    else:
        base_model_path = Path(base_model_path)

    # Determine ema.safetensors path
    if model_path.suffix == ".safetensors":
        ema_checkpoint_path = model_path
    else:
        # If it's a directory, look for ema.safetensors
        ema_checkpoint_path = model_path / "ema.safetensors"
        if not ema_checkpoint_path.exists():
            # If not found, assume the whole directory is the model directory
            ema_checkpoint_path = model_path

    # LLM config preparing (loaded from base model path)
    llm_config = Qwen2Config.from_json_file(base_model_path / "llm_config.json")
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config preparing (loaded from base model path)
    vit_config = SiglipVisionConfig.from_json_file(base_model_path / "vit_config.json")
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE loading (loaded from base model path)
    vae_model, vae_config = load_ae(local_path=str(base_model_path / "ae.safetensors"))

    # Bagel config preparing
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # Tokenizer Preparing (loaded from base model path)
    tokenizer = Qwen2Tokenizer.from_pretrained(str(base_model_path))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Image Transform Preparing
    # If max_input_pixels is specified, pass it to ImageTransform
    vae_transform = ImageTransform(vae_transform_size[0], vae_transform_size[1], 16, max_pixels=max_input_pixels)
    vit_transform = ImageTransform(vit_transform_size[0], vit_transform_size[1], 14, max_pixels=max_input_pixels)

    # Device map
    if device_id is not None:
        # Assign all modules to the specified GPU (no sharding)
        device_map = {"": f"cuda:{device_id}"}
    else:
        # Auto-assign GPU (original logic)
        device_map = infer_auto_device_map(
            model,
            max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )

        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]

        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

    # Load checkpoint (load ema.safetensors from model_path)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(ema_checkpoint_path),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder=offload_folder
    )

    model = model.eval()

    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


def process_sample(
    sample: Dict[str, Any],
    output_dir: Path,
    inferencer: InterleaveInferencer,
    inference_hyper: Dict[str, Any],
    gpu_id: int,
    lock: threading.Lock,
    seed: Optional[int] = None
) -> Tuple[bool, Optional[str]]:
    """
    Process a single sample (thread-safe).

    Args:
        sample: Sample data
        output_dir: Output directory
        inferencer: Inferencer (protected by thread lock)
        inference_hyper: Inference hyperparameters
        gpu_id: GPU ID (used for logging)
        lock: Thread lock

    Returns:
        (success flag, error message)
    """
    try:
        idx = sample.get("idx", 0)
        task = sample.get("task", "")

        # Resume checkpoint check
        if check_sample_exists(output_dir, idx):
            return (True, None)

        prompt = sample.get("prompt", "")
        input_images = sample.get("input_images", [])
        output_image_path = sample.get("output_image", "")

        # Load input images
        images = []
        for img_item in input_images:
            if isinstance(img_item, (str, Path)):
                img_path = str(img_item)
                if os.path.exists(img_path):
                    images.append(Image.open(img_path).convert("RGB"))
                else:
                    print(f"[GPU {gpu_id}] Warning: image file not found: {img_path}")
            elif isinstance(img_item, Image.Image):
                images.append(img_item.convert("RGB"))
            else:
                print(f"[GPU {gpu_id}] Warning: unsupported image type: {type(img_item)}")

        if len(images) == 0:
            return (False, f"Skipping sample {idx}: no valid input images")

        # Protect inference with thread lock
        with lock:
            # If image_shapes is specified in inference_hyper, use that size; otherwise use the last input image size
            output_dict = inferencer(image=images, text=prompt, **inference_hyper)
            generated_image = output_dict['image']

        # Save image
        output_image_file = output_dir / f"{idx:08d}.jpg"
        generated_image.save(output_image_file)
        # Set file permissions to 777 (read/write/execute for all)
        os.chmod(output_image_file, 0o777)

        # Use unified save interface
        save_sample(
            output_dir=output_dir,
            idx=idx,
            sample=sample,
            output_image_path=str(output_image_file),
            target_image_path=output_image_path,
        )

        return (True, None)

    except Exception as e:
        sample_idx = sample.get("idx", "unknown")
        error_msg = f"Error processing sample {sample_idx}: {e}"
        import traceback
        traceback.print_exc()
        return (False, error_msg)


def run_inference(
    model_path: str,
    task: str,
    image_num_category: str = "all",
    output_dir: str = "./outputs",
    base_model_path: Optional[str] = None,
    vae_transform: Optional[tuple] = None,
    vit_transform: Optional[tuple] = None,
    inference_hyper: Optional[Dict[str, Any]] = None,
    max_mem_per_gpu: str = "40GiB",
    offload_folder: str = "/tmp/offload",
    num_workers: int = 8,
    max_input_pixels = None,  # Can be int or list, used to dynamically adjust input image resolution
    data_root: Optional[str] = None,  # Data root directory; if None, use default path
) -> int:
    # Default transform sizes
    if vae_transform is None:
        vae_transform = (1024, 512)
    if vit_transform is None:
        vit_transform = (980, 224)

    # Default inference hyperparameters
    if inference_hyper is None:
        inference_hyper = dict(
            seed=42,
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            cfg_interval=[0.0, 1.0],
            timestep_shift=3.0,
            num_timesteps=50,
            cfg_renorm_min=0.0,
            cfg_renorm_type="text_channel",
        )

    # Set random seed for reproducibility
    seed = inference_hyper.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Setting random seed: {seed}")

    # Remove seed from inference_hyper (if present) since global seed is already set
    inference_hyper = {k: v for k, v in inference_hyper.items() if k != 'seed'}

    # Determine the number of available GPUs
    available_gpus = torch.cuda.device_count()

    if available_gpus <= 0:
        raise RuntimeError("No available GPUs")

    actual_num_workers = num_workers

    print(f"Available GPUs: {available_gpus}")
    print(f"Loading {actual_num_workers} models, assigned sequentially to all GPUs")

    # Load all models at once onto GPUs (round-robin assignment)
    print(f"Loading {actual_num_workers} models...")
    models_data = []
    for model_idx in range(actual_num_workers):
        gpu_id = model_idx % available_gpus

        print(f"Loading model {model_idx + 1}/{actual_num_workers} to GPU {gpu_id}...")
        model, vae_model, tokenizer, vae_tf, vit_tf, new_token_ids = load_model(
            model_path=model_path,
            base_model_path=base_model_path,
            vae_transform_size=vae_transform,
            vit_transform_size=vit_transform,
            max_mem_per_gpu=max_mem_per_gpu,
            offload_folder=f"{offload_folder}_{gpu_id}_{model_idx}",
            device_id=gpu_id,
            max_input_pixels=max_input_pixels,
        )

        inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_tf,
            vit_transform=vit_tf,
            new_token_ids=new_token_ids
        )

        models_data.append({
            'model_idx': model_idx,
            'gpu_id': gpu_id,
            'inferencer': inferencer,
            'lock': threading.Lock(),
        })
        print(f"Model {model_idx + 1} loaded (GPU {gpu_id})")

    print("All models loaded!")

    # Determine tasks to process
    if task == "all":
        tasks_to_process = SUPPORTED_TASKS
    else:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}. Supported tasks: {SUPPORTED_TASKS}")
        tasks_to_process = [task]

    # Process each task
    for current_task in tasks_to_process:
        print(f"\nProcessing task: {current_task}")
        print(f"Image count category: {image_num_category}")

        # Use unified data loading interface
        try:
            samples = load_data_for_task(
                task=current_task,
                image_num_category=image_num_category if image_num_category != "all" else None,
                data_root=Path(data_root) if data_root else None,
            )
            print(f"Loaded {len(samples)} samples")
        except Exception as e:
            print(f"Failed to load data: {e}")
            import traceback
            traceback.print_exc()
            continue

        if len(samples) == 0:
            print(f"No samples found, skipping")
            continue

        # Create output directory
        task_output_dir = Path(output_dir)
        task_output_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(task_output_dir, 0o777)

        # Check for resume checkpoint
        skipped_count = 0
        for sample in samples:
            idx = sample.get("idx", 0)
            if check_sample_exists(task_output_dir, idx):
                skipped_count += 1

        if skipped_count > 0:
            print(f"Found {skipped_count} already-generated samples, will skip")

        print(f"Starting parallel processing of {len(samples)} samples (using {actual_num_workers} models)...")

        samples_with_model = []
        for i, sample in enumerate(samples):
            model_idx = i % actual_num_workers
            samples_with_model.append((sample, model_idx))

        success_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=actual_num_workers) as executor:
            futures = []
            for sample, model_idx in samples_with_model:
                model_data = models_data[model_idx]
                future = executor.submit(
                    process_sample,
                    sample,
                    task_output_dir,
                    model_data['inferencer'],
                    inference_hyper,
                    model_data['gpu_id'],
                    model_data['lock'],
                )
                futures.append((future, sample.get("idx", 0)))

            for future, idx in futures:
                try:
                    success, error_msg = future.result()
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        if error_msg:
                            print(f"Sample {idx}: {error_msg}")
                except Exception as e:
                    error_count += 1
                    print(f"Sample {idx} processing exception: {e}")

        print(f"\nTask {current_task} completed:")
        print(f"  Success: {success_count + skipped_count} ({skipped_count} skipped)")
        print(f"  Failed: {error_count}")

        all_success = (error_count == 0)

        if all_success:
            finish_file = task_output_dir / ".finish"
            try:
                finish_file.touch()
                os.chmod(finish_file, 0o777)
                print(f"✓ All samples generated successfully. Created finish marker: {finish_file}")
            except Exception as e:
                print(f"Warning: Failed to create finish marker file: {e}")
        else:
            total_success = success_count + skipped_count
            print(f"✗ Some samples failed. Finish marker not created.")
            print(f"  Total: {len(samples)}, Success: {total_success}, Failed: {error_count}")

    print("\nAll tasks completed!")

    has_failure = False
    for current_task in tasks_to_process:
        task_output_dir = Path(output_dir)
        finish_file = task_output_dir / ".finish"
        if not finish_file.exists():
            has_failure = True
            break

    return 0 if not has_failure else 1


def main():
    parser = argparse.ArgumentParser(description="Bagel model inference script (supports multi-GPU parallel and checkpoint resume)")

    parser.add_argument("--model_path", type=str, required=True,
                       help="Path containing ema.safetensors (directory or safetensors file path)")
    parser.add_argument("--base_model_path", type=str, default=None,
                       help="Base model path (contains llm_config.json, vit_config.json, ae.safetensors); if None, use default path")
    parser.add_argument("--data_root", type=str, default=None,
                       help="Data root directory (contains filter/{task}/eval structure); if None, use default path")
    parser.add_argument("--task", type=str, default="all",
                       choices=SUPPORTED_TASKS + ["all"],
                       help="Task type")
    parser.add_argument("--image_num_category", type=str, default="all",
                       choices=IMAGE_NUM_CATEGORIES + ["all"],
                       help="Image count category")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--vae_transform", type=str, default="1024,512",
                       help="VAE transform size, format: height,width")
    parser.add_argument("--vit_transform", type=str, default="980,224",
                       help="ViT transform size, format: height,width")
    parser.add_argument("--max_mem_per_gpu", type=str, default="40GiB",
                       help="Maximum memory per GPU")
    parser.add_argument("--offload_folder", type=str, default="/tmp/offload",
                       help="Offload folder path")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of parallel worker threads (total models), default 8. Models are assigned sequentially to all available GPUs")

    # Inference hyperparameters
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--cfg_text_scale", type=float, default=4.0,
                       help="CFG text scale")
    parser.add_argument("--cfg_img_scale", type=float, default=2.0,
                       help="CFG image scale")
    parser.add_argument("--cfg_interval", type=str, default="0.0,1.0",
                       help="CFG interval, format: start,end")
    parser.add_argument("--timestep_shift", type=float, default=3.0,
                       help="Timestep shift")
    parser.add_argument("--num_timesteps", type=int, default=50,
                       help="Number of timesteps")
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0,
                       help="CFG renorm min")
    parser.add_argument("--cfg_renorm_type", type=str, default="text_channel",
                       choices=["global", "text_channel"],
                       help="CFG renorm type")
    parser.add_argument("--max_input_pixels", type=str, default=None,
                       help="Maximum input pixel count; can be int or JSON list string for dynamic resolution adjustment")
    parser.add_argument("--image_shapes", type=str, default=None,
                       help="Output image size, format: height,width (e.g. 768,768). If not specified, uses the last input image size")

    args = parser.parse_args()

    # Parse transform sizes
    vae_transform = tuple(map(int, args.vae_transform.split(",")))
    vit_transform = tuple(map(int, args.vit_transform.split(",")))

    # Parse max_input_pixels (may be a JSON string)
    max_input_pixels = None
    if args.max_input_pixels:
        try:
            max_input_pixels = json.loads(args.max_input_pixels)
        except json.JSONDecodeError:
            try:
                max_input_pixels = int(args.max_input_pixels)
            except ValueError:
                raise ValueError(f"Cannot parse max_input_pixels: {args.max_input_pixels}")

    # Default dynamic resolution
    if max_input_pixels is None:
        max_input_pixels = [1048576, 1048576, 589824, 589824, 589824, 262144, 262144, 262144, 262144, 262144]

    # Parse CFG interval
    cfg_interval = list(map(float, args.cfg_interval.split(",")))

    # Parse image_shapes (if specified)
    image_shapes = None
    if args.image_shapes:
        try:
            height, width = map(int, args.image_shapes.split(","))
            image_shapes = (height, width)
        except ValueError:
            raise ValueError(f"Cannot parse image_shapes: {args.image_shapes}, format should be height,width (e.g. 768,768)")

    # Build inference hyperparameters
    inference_hyper = dict(
        seed=args.seed,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=cfg_interval,
        timestep_shift=args.timestep_shift,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=args.cfg_renorm_min,
        cfg_renorm_type=args.cfg_renorm_type,
    )

    # If image_shapes is specified, add to inference_hyper
    if image_shapes is not None:
        inference_hyper['image_shapes'] = image_shapes

    # Run inference
    exit_code = run_inference(
        model_path=args.model_path,
        task=args.task,
        image_num_category=args.image_num_category,
        output_dir=args.output_dir,
        base_model_path=args.base_model_path,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        inference_hyper=inference_hyper,
        max_mem_per_gpu=args.max_mem_per_gpu,
        offload_folder=args.offload_folder,
        num_workers=args.num_workers,
        max_input_pixels=max_input_pixels,
        data_root=args.data_root,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
