#!/usr/bin/env python3
"""
Qwen-Image-Edit inference script

Loads data from the filter directory, runs inference using Qwen-Image-Edit model, and saves results.
Supports multi-GPU parallel inference and checkpoint resume.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import numpy as np

from PIL import Image
import torch
from diffsynth.pipelines.qwen_image import QwenImagePipeline
from diffsynth.core import ModelConfig
from diffsynth import load_state_dict

# Add qwen source path
SCRIPT_DIR = Path(__file__).parent
QWEN_DIR = SCRIPT_DIR.parent
MACRO_DIR = QWEN_DIR.parent  # Macro root directory
QWEN_SOURCE_PATH = QWEN_DIR / "source"
if str(QWEN_SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(QWEN_SOURCE_PATH))

# Import local utils
from utils import (
    load_data_for_task,  # unified data loading interface
    SUPPORTED_TASKS,
    THIRDPARTY_TASKS,
    IMAGE_NUM_CATEGORIES,
    check_sample_exists,
    save_sample
)


def load_pipeline(
    model_configs: List[Dict[str, str]],
    processor_config: Dict[str, str],
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    device_id: Optional[int] = None,
    model_base_path: Optional[str] = None,
    transformer_path: Optional[str] = None,
    tokenizer_config: Optional[Dict[str, str]] = None,
    lora_path: Optional[str] = None,
):
    """
    Load Qwen-Image-Edit Pipeline

    Args:
        model_configs: list of model configurations, each containing model_id and origin_file_pattern
        processor_config: processor configuration, containing model_id and origin_file_pattern
        device: device type ("cuda" or "cpu")
        torch_dtype: torch data type ("float32", "float16", "bfloat16")
        device_id: specified GPU device ID (0-based), uses default device if None
        model_base_path: model base path; if specified, sets DIFFSYNTH_MODEL_BASE_PATH env variable
        transformer_path: trained transformer weight path (optional); if specified, loads trained weights into pipe.dit (full fine-tuning)
        tokenizer_config: tokenizer configuration (optional), containing model_id and origin_file_pattern
        lora_path: LoRA weight path (optional); if specified, loads LoRA weights into pipe.dit
                   Note: transformer_path and lora_path are mutually exclusive

    Returns:
        pipeline: QwenImagePipeline instance
    """
    # Check that transformer_path and lora_path are mutually exclusive
    if transformer_path and lora_path:
        raise ValueError("transformer_path and lora_path are mutually exclusive, only one can be specified. "
                        "transformer_path is for full fine-tuning weights, lora_path is for LoRA weights.")

    # Set local model path environment variable (if specified)
    if model_base_path:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = model_base_path
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        print(f"Set local model path: {model_base_path}")
        print(f"Skipping ModelScope download, using local model")

    # Convert torch_dtype
    if torch_dtype == "float32":
        dtype = torch.float32
    elif torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

    # Determine device
    if device_id is not None:
        device_str = f"cuda:{device_id}"
    else:
        device_str = device

    # Set environment variables
    if model_base_path:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = model_base_path
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"

    # Build ModelConfig list
    model_config_objs = []
    for config in model_configs:
        model_config_objs.append(
            ModelConfig(
                model_id=config["model_id"],
                origin_file_pattern=config["origin_file_pattern"]
            )
        )

    # Build processor configuration
    processor_config_obj = ModelConfig(
        model_id=processor_config["model_id"],
        origin_file_pattern=processor_config["origin_file_pattern"]
    )

    # Build tokenizer configuration (if specified)
    tokenizer_config_obj = None
    if tokenizer_config:
        tokenizer_config_obj = ModelConfig(
            model_id=tokenizer_config["model_id"],
            origin_file_pattern=tokenizer_config["origin_file_pattern"]
        )

    # Load pipeline
    print(f"Loading Qwen-Image-Edit Pipeline (device: {device_str}, dtype: {dtype})...")
    pipeline = QwenImagePipeline.from_pretrained(
        torch_dtype=dtype,
        device=device_str,
        model_configs=model_config_objs,
        processor_config=processor_config_obj,
        tokenizer_config=tokenizer_config_obj,
    )

    # If trained transformer path is specified, load trained weights
    if transformer_path:
        # Handle relative paths (relative to model_base_path)
        if not os.path.isabs(transformer_path) and model_base_path:
            transformer_path = os.path.join(model_base_path, transformer_path)

        # Check if file exists
        if not os.path.exists(transformer_path):
            raise FileNotFoundError(f"Trained transformer weight file does not exist: {transformer_path}")

        print(f"Loading trained transformer weights: {transformer_path}")
        state_dict = load_state_dict(transformer_path)

        # Handle weight key name prefix (training may have saved with "pipe.dit." prefix)
        # If state_dict keys contain "pipe.dit." prefix, remove it
        if state_dict and any(key.startswith("pipe.dit.") for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                # Remove "pipe.dit." prefix
                if key.startswith("pipe.dit."):
                    new_key = key[len("pipe.dit."):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
            print(f"Removed 'pipe.dit.' prefix from weight key names")

        pipeline.dit.load_state_dict(state_dict)
        print(f"Trained transformer weights loaded successfully (full fine-tuning)")

    # If LoRA weight path is specified, load LoRA weights
    # Reference: examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit-2511.py
    if lora_path:
        # Handle relative paths (relative to model_base_path)
        if not os.path.isabs(lora_path) and model_base_path:
            lora_path = os.path.join(model_base_path, lora_path)

        # Check if file exists
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA weight file does not exist: {lora_path}")

        print(f"Loading LoRA weights: {lora_path}")
        # Use QwenImagePipeline's load_lora method
        pipeline.load_lora(pipeline.dit, lora_path)
        print(f"LoRA weights loaded successfully")

    print(f"Pipeline loaded successfully")
    return pipeline


def process_sample(
    sample: Dict[str, Any],
    output_dir: Path,
    pipeline: QwenImagePipeline,
    inference_hyper: Dict[str, Any],
    gpu_id: int,
    seed: Optional[int] = None
) -> Tuple[bool, Optional[str]]:
    """
    Process a single sample

    Args:
        sample: sample data
        output_dir: output directory
        pipeline: inference pipeline (each thread uses an independent instance, no lock needed)
        inference_hyper: inference hyperparameters
        gpu_id: GPU ID (for logging)
        seed: random seed (optional)

    Returns:
        (success flag, error message)
    """
    try:
        idx = sample.get("idx", 0)
        seed = sample.get("seed", None)
        task = sample.get("task", "")

        # Checkpoint resume check
        # For geneval and dpg, if seed is specified, check if file for that specific seed exists
        if seed is not None and task in ["geneval", "dpg"]:
            base_idx = sample.get("base_idx", idx)
            image_file = output_dir / f"{base_idx:08d}_seed{seed:02d}.jpg"
            json_file = output_dir / f"{base_idx:08d}_seed{seed:02d}.json"
            if image_file.exists() and json_file.exists():
                # Check if image is readable
                try:
                    with Image.open(image_file) as img:
                        img.verify()
                    with Image.open(image_file) as img:
                        img.convert("RGB").load()
                    return (True, None)
                except:
                    pass
        else:
            if check_sample_exists(output_dir, idx):
                return (True, None)

        prompt = sample.get("prompt", "")
        input_images = sample.get("input_images", [])
        output_image_path = sample.get("output_image", "")

        # For omnicontext tasks, if input_images contains PIL Image objects, save them locally first
        if task == "omnicontext" or (not task and any(isinstance(img, Image.Image) for img in input_images)):
            has_pil_images = any(isinstance(img, Image.Image) for img in input_images)
            if has_pil_images:
                try:
                    SCRIPT_DIR = Path(__file__).parent
                    MACRO_DIR = SCRIPT_DIR.parent.parent
                    INFERENCE_UTILS_DIR = MACRO_DIR / "inference_utils"
                    if str(INFERENCE_UTILS_DIR) not in sys.path:
                        sys.path.insert(0, str(INFERENCE_UTILS_DIR))

                    from thirdparty_tasks import save_omnicontext_input_images

                    saved_paths = save_omnicontext_input_images(
                        sample_idx=idx,
                        input_images=input_images,
                        prompt=prompt
                    )
                    sample["input_images"] = saved_paths
                    input_images = saved_paths
                    print(f"[GPU {gpu_id}] Saved OmniContext input images locally: {len(saved_paths)} images")
                except Exception as e:
                    print(f"[GPU {gpu_id}] Failed to save OmniContext input images: {e}")

        # Load input images
        # For geneval and dpg tasks, these are pure text-to-image generation tasks, no input images needed
        is_text_to_image_task = task in ["geneval", "dpg"]

        edit_images = []
        if not is_text_to_image_task:
            for img_item in input_images:
                if isinstance(img_item, (str, Path)):
                    img_path = str(img_item)
                    if os.path.exists(img_path):
                        edit_images.append(Image.open(img_path).convert("RGB"))
                    else:
                        print(f"[GPU {gpu_id}] Warning: image file does not exist: {img_path}")
                elif isinstance(img_item, Image.Image):
                    edit_images.append(img_item.convert("RGB"))
                else:
                    print(f"[GPU {gpu_id}] Warning: unsupported image type: {type(img_item)}")

            if len(edit_images) == 0:
                print(f"[GPU {gpu_id}] Warning: sample {idx} has no input images, skipping")
                return (False, f"Skipping sample {idx}: no valid input images")

        # If seed is specified, set random seed
        # For geneval and dpg, use base_seed (42) + seed as actual seed
        # Only set seed for current thread's GPU to avoid race conditions / segfaults from manual_seed_all in multi-threading
        if seed is not None:
            actual_seed = 42 + seed
            random.seed(actual_seed)
            np.random.seed(actual_seed)
            torch.manual_seed(actual_seed)
            if torch.cuda.is_available():
                with torch.cuda.device(gpu_id):
                    torch.cuda.manual_seed(actual_seed)
            # Update seed in inference_hyper
            inference_hyper = inference_hyper.copy()
            inference_hyper['seed'] = actual_seed

        # Inference (each thread uses independent pipeline instance, no lock needed)
        # For t2i tasks (geneval, dpg), edit_image should be None (not empty list)
        # Because QwenImageUnit_PromptEmbedder calls encode_prompt (pure text encoding) when edit_image=None
        # while edit_image=[] calls encode_prompt_edit_multi which cannot handle empty lists
        # print(f"Prompt: {prompt}, num images: {len(edit_images)}")
        if is_text_to_image_task:
            # For pure text generation tasks, pass None instead of empty list
            generated_image = pipeline(
                prompt,
                edit_image=None,  # None means pure text generation
                **inference_hyper
            )
        else:
            # Note: edit_image must be a list (non-empty) or None
            generated_image = pipeline(
                prompt,
                edit_image=edit_images if edit_images else None,  # pass None if empty list
                **inference_hyper
            )

        # Save image
        # For geneval and dpg, if seed is specified, use base_idx as filename prefix
        if seed is not None and task in ["geneval", "dpg"]:
            base_idx = sample.get("base_idx", idx)
            output_image_file = output_dir / f"{base_idx:08d}_seed{seed:02d}.jpg"
            save_idx = base_idx
        else:
            output_image_file = output_dir / f"{idx:08d}.jpg"
            save_idx = idx

        generated_image.save(output_image_file)
        # Set file permissions to 777 (readable/writable/executable by all)
        os.chmod(output_image_file, 0o777)

        # Ensure sample dict contains base_idx, idx, task_type, tag, include (consistent with bagel, for evaluation/visualization)
        sample_with_base_idx = sample.copy()
        if seed is not None and task in ["geneval", "dpg"]:
            sample_with_base_idx["base_idx"] = base_idx
        sample_with_base_idx["idx"] = save_idx  # ensure saved JSON idx matches filename
        if task == "omnicontext" and "task_type" not in sample_with_base_idx:
            sample_with_base_idx["task_type"] = sample.get("task_type", "")
        # geneval/dpg evaluation needs tag and include, ensure they are preserved
        if task in ["geneval", "dpg"] and "tag" in sample:
            sample_with_base_idx["tag"] = sample["tag"]
        if task in ["geneval", "dpg"] and "include" in sample:
            sample_with_base_idx["include"] = sample["include"]

        # Use unified save interface
        save_sample(
            output_dir=output_dir,
            idx=save_idx,
            sample=sample_with_base_idx,
            output_image_path=str(output_image_file),
            target_image_path=output_image_path,
            seed=seed  # pass seed parameter
        )

        return (True, None)

    except Exception as e:
        sample_idx = sample.get("idx", "unknown")
        error_msg = f"Error processing sample {sample_idx}: {e}"
        import traceback
        traceback.print_exc()
        return (False, error_msg)


def run_inference(
    model_configs: List[Dict[str, str]],
    processor_config: Dict[str, str],
    task: str,
    image_num_category: str = "all",
    output_dir: str = "./outputs",
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    inference_hyper: Optional[Dict[str, Any]] = None,
    num_workers: int = 1,
    use_refine_prompt: bool = False,  # for geneval, whether to use refine prompt
    model_base_path: Optional[str] = None,  # model base path for local model loading
    transformer_path: Optional[str] = None,  # trained transformer weight path (full fine-tuning)
    tokenizer_config: Optional[Dict[str, str]] = None,  # tokenizer configuration
    lora_path: Optional[str] = None,  # LoRA weight path (LoRA fine-tuning)
    data_root: Optional[str] = None,  # data root directory, uses default path if None
) -> int:
    """
    Run inference (multi-GPU parallel, supports checkpoint resume)

    Args:
        model_configs: list of model configurations
        processor_config: processor configuration
        task: task type (customization, illustration, spatial, temporal, all)
        image_num_category: image count category (1-3, 4-5, 6-7, >=8, all)
        output_dir: output directory
        device: device type
        torch_dtype: torch data type
        inference_hyper: inference hyperparameters
        num_workers: number of parallel worker threads (total model count), default 1. Models are assigned to all available GPUs in order
    """
    # Default inference hyperparameters
    if inference_hyper is None:
        inference_hyper = dict(
            seed=42,
            num_inference_steps=40,
            height=768,
            width=768,
            edit_image_auto_resize=True,
            zero_cond_t=True,  # special parameter for Qwen-Image-Edit-2511
        )

    # Set random seed for reproducibility
    seed = inference_hyper.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set random seed: {seed}")

    # Determine number of available GPUs
    if device == "cuda":
        available_gpus = torch.cuda.device_count()
        if available_gpus <= 0:
            raise RuntimeError("No available GPUs")
    else:
        available_gpus = 1

    # Assign num_workers models to all GPUs in order (round-robin)
    actual_num_workers = num_workers

    print(f"Available GPUs: {available_gpus}")
    print(f"Will load {actual_num_workers} models, assigned to all GPUs in order")

    # Load all models to GPU at once (round-robin assignment to all GPUs)
    print(f"Loading {actual_num_workers} models...")
    pipelines_data = []
    for model_idx in range(actual_num_workers):
        # Round-robin assignment to GPUs
        if device == "cuda":
            gpu_id = model_idx % available_gpus
        else:
            gpu_id = None

        print(f"Loading model {model_idx + 1}/{actual_num_workers} to GPU {gpu_id if gpu_id is not None else 'CPU'}...")
        pipeline = load_pipeline(
            model_configs=model_configs,
            processor_config=processor_config,
            device=device,
            torch_dtype=torch_dtype,
            device_id=gpu_id,
            model_base_path=model_base_path,
            transformer_path=transformer_path,
            tokenizer_config=tokenizer_config,
            lora_path=lora_path,
        )

        pipelines_data.append({
            'model_idx': model_idx,
            'gpu_id': gpu_id if gpu_id is not None else 0,
            'pipeline': pipeline,  # each model instance is independent, no lock needed
        })
        print(f"Model {model_idx + 1} loaded successfully (GPU {gpu_id if gpu_id is not None else 'CPU'})")

    print("All models loaded successfully!")

    # Determine tasks to process
    if task == "all":
        tasks_to_process = SUPPORTED_TASKS
    elif task in THIRDPARTY_TASKS:
        tasks_to_process = [task]
    else:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}. Supported tasks: {SUPPORTED_TASKS + THIRDPARTY_TASKS}")
        tasks_to_process = [task]

    # Process each task
    for current_task in tasks_to_process:
        print(f"\nProcessing task: {current_task}")
        if current_task not in THIRDPARTY_TASKS:
            print(f"Image count category: {image_num_category}")

        # Use unified data loading interface
        try:
            samples = load_data_for_task(
                task=current_task,
                image_num_category=image_num_category if image_num_category != "all" and current_task not in THIRDPARTY_TASKS else None,
                data_root=Path(data_root) if data_root else None,
                use_refine_prompt=use_refine_prompt
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
        # output_dir already contains task and category info (passed in from run.py)
        # so use output_dir directly as task_output_dir
        task_output_dir = Path(output_dir)
        task_output_dir.mkdir(parents=True, exist_ok=True)
        # Set directory permissions to 777 (readable/writable/executable by all)
        os.chmod(task_output_dir, 0o777)

        # For geneval and dpg, need to generate multiple seed results for each prompt
        # Generate 4 samples per prompt using seeds [0, 1, 2, 3]
        NUM_SEEDS_PER_PROMPT = 4

        # Check if this is geneval or dpg task
        is_multi_seed_task = current_task in ["geneval", "dpg"]

        # Expand sample list: for geneval and dpg, each prompt needs multiple seed results
        if is_multi_seed_task:
            expanded_samples = []
            for sample in samples:
                base_idx = sample.get("idx", 0)
                for seed in range(NUM_SEEDS_PER_PROMPT):
                    expanded_sample = sample.copy()
                    expanded_sample["seed"] = seed
                    expanded_sample["base_idx"] = base_idx  # save original idx
                    expanded_sample["idx"] = base_idx * NUM_SEEDS_PER_PROMPT + seed  # new idx
                    expanded_samples.append(expanded_sample)
            samples = expanded_samples
            print(f"Expanded to {len(samples)} samples ({NUM_SEEDS_PER_PROMPT} seeds per prompt)")

        # Check checkpoint resume
        skipped_count = 0
        for sample in samples:
            idx = sample.get("idx", 0)
            seed = sample.get("seed", None)
            # For multi-seed tasks, check if file for specific seed exists
            if is_multi_seed_task and seed is not None:
                image_file = task_output_dir / f"{sample['base_idx']:08d}_seed{seed:02d}.jpg"
                json_file = task_output_dir / f"{sample['base_idx']:08d}_seed{seed:02d}.json"
                if image_file.exists() and json_file.exists():
                    # Check if image is readable
                    try:
                        with Image.open(image_file) as img:
                            img.verify()
                        with Image.open(image_file) as img:
                            img.convert("RGB").load()
                        skipped_count += 1
                    except:
                        pass
            else:
                if check_sample_exists(task_output_dir, idx):
                    skipped_count += 1

        if skipped_count > 0:
            print(f"Found {skipped_count} already generated samples, will skip")

        # Use thread pool to process samples in parallel
        print(f"Starting parallel processing of {len(samples)} samples (using {actual_num_workers} models)...")

        # Assign samples to different models (round-robin)
        samples_with_model = []
        for i, sample in enumerate(samples):
            model_idx = i % actual_num_workers
            samples_with_model.append((sample, model_idx))

        # Execute with thread pool
        success_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=actual_num_workers) as executor:
            futures = []
            for sample, model_idx in samples_with_model:
                pipeline_data = pipelines_data[model_idx]
                seed = sample.get("seed", None)  # get seed
                future = executor.submit(
                    process_sample,
                    sample,
                    task_output_dir,
                    pipeline_data['pipeline'],
                    inference_hyper,
                    pipeline_data['gpu_id'],  # use GPU ID for logging
                    seed=seed  # pass seed parameter
                )
                futures.append((future, sample.get("idx", 0)))

            # Wait for all thread tasks to complete and collect results (only judge .finish after all complete)
            for future, idx in futures:
                try:
                    success, error_msg = future.result()
                    if success:
                        if error_msg is None:
                            success_count += 1
                        else:
                            # skipped case
                            success_count += 1
                    else:
                        error_count += 1
                        if error_msg:
                            print(f"Sample {idx}: {error_msg}")
                except Exception as e:
                    error_count += 1
                    print(f"Sample {idx} processing exception: {e}")

        print(f"\nTask {current_task} processing complete:")
        print(f"  Success: {success_count + skipped_count} ({skipped_count} skipped)")
        print(f"  Failed: {error_count}")

        # Only write .finish when all threads succeeded (no failures)
        total_samples = len(samples)
        total_success = success_count + skipped_count
        all_success = (error_count == 0)

        # For geneval and dpg, need to create bench directory first, then create .finish file
        bench_created = True
        if all_success and current_task in ["geneval", "dpg"]:
            print(f"\nCreating bench directory for {current_task} task in parallel (using {actual_num_workers} threads)...")
            try:
                # Add inference_utils path
                inference_utils_dir = MACRO_DIR / "inference_utils"
                if str(inference_utils_dir) not in sys.path:
                    sys.path.insert(0, str(inference_utils_dir))
                from thirdparty_tasks import create_bench_directory_parallel

                # Use thread pool to create bench directory in parallel
                bench_results = []
                with ThreadPoolExecutor(max_workers=actual_num_workers) as executor:
                    futures = []
                    for worker_idx in range(actual_num_workers):
                        future = executor.submit(
                            create_bench_directory_parallel,
                            current_task,
                            task_output_dir,
                            process_index=worker_idx,
                            num_processes=actual_num_workers
                        )
                        futures.append(future)

                    for future in futures:
                        try:
                            result = future.result()
                            bench_results.append(result)
                        except Exception as e:
                            print(f"Bench directory creation thread error: {e}")
                            bench_results.append(False)

                bench_created = all(bench_results)
                if bench_created:
                    print(f"Bench directory creation complete")
                else:
                    print(f"Warning: Bench directory creation failed")
            except Exception as e:
                print(f"Failed to create bench directory: {e}")
                import traceback
                traceback.print_exc()
                bench_created = False

        # Only create .finish file when all threads succeeded and (not geneval/dpg task or bench directory created successfully)
        if all_success and bench_created:
            finish_file = task_output_dir / ".finish"
            try:
                finish_file.touch()
                os.chmod(finish_file, 0o777)
                print(f"All samples generated successfully, created completion marker file: {finish_file}")
            except Exception as e:
                print(f"Warning: Failed to create completion marker file: {e}")
        else:
            print(f"Some samples failed, completion marker file not created")
            print(f"  Total samples: {total_samples}, success: {total_success}, failed: {error_count}")

    print("\nAll tasks processing complete!")

    # Return exit code: 0 for success, non-0 for failure
    # Check if any task failed (by checking .finish files)
    has_failure = False
    for current_task in tasks_to_process:
        task_output_dir = Path(output_dir)
        finish_file = task_output_dir / ".finish"
        if not finish_file.exists():
            has_failure = True
            print(f"Task {current_task} incomplete (missing .finish file)")

    return 0 if not has_failure else 1


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit model inference script (supports multi-GPU parallel and checkpoint resume)")

    parser.add_argument("--model_configs", type=str, required=True, action='append',
                       help="model configuration, format: model_id:origin_file_pattern (can specify multiple, use --model_configs once per entry)")
    parser.add_argument("--processor_config", type=str, required=True,
                       help="processor configuration, format: model_id:origin_file_pattern")
    parser.add_argument("--tokenizer_config", type=str, default=None,
                       help="tokenizer configuration (optional), format: model_id:origin_file_pattern")
    parser.add_argument("--task", type=str, default="all",
                       choices=SUPPORTED_TASKS + ["all"],
                       help="task type")
    parser.add_argument("--use_refine_prompt", action="store_true", default=False,
                       help="For geneval task, whether to use refine prompt.")
    parser.add_argument("--image_num_category", type=str, default="all",
                       choices=IMAGE_NUM_CATEGORIES + ["all"],
                       help="image count category")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="output directory")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="device type")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                       choices=["float32", "float16", "bfloat16"],
                       help="torch data type")
    parser.add_argument("--num_workers", type=int, default=1,
                       help="number of parallel worker threads (total model count), default 1. Models are assigned to all available GPUs in order")
    parser.add_argument("--model_base_path", type=str, default=None,
                       help="model base path; if specified, sets DIFFSYNTH_MODEL_BASE_PATH environment variable and uses local model")
    parser.add_argument("--transformer_path", type=str, default=None,
                       help="trained transformer weight path (full fine-tuning, optional); if specified, loads trained weights into pipe.dit")
    parser.add_argument("--lora_path", type=str, default=None,
                       help="LoRA weight path (LoRA fine-tuning, optional); if specified, loads LoRA weights into pipe.dit. Mutually exclusive with --transformer_path")
    parser.add_argument("--data_root", type=str, default=None,
                       help="data root directory (containing filter/{task}/eval structure), uses default path if None")

    # Inference hyperparameters
    parser.add_argument("--seed", type=int, default=42,
                       help="random seed for reproducibility")
    parser.add_argument("--num_inference_steps", type=int, default=40,
                       help="number of inference steps")
    parser.add_argument("--height", type=int, default=768,
                       help="output image height")
    parser.add_argument("--width", type=int, default=768,
                       help="output image width")
    parser.add_argument("--edit_image_auto_resize", action="store_true", default=True,
                       help="automatically resize input images")
    parser.add_argument("--zero_cond_t", action="store_true", default=True,
                       help="special parameter for Qwen-Image-Edit-2511")
    parser.add_argument("--max_input_pixels", type=str, default=None,
                       help="maximum input pixel count, can be int or JSON list string, used to dynamically adjust input image resolution")

    args = parser.parse_args()

    # Parse model configurations
    model_configs = []
    for config_str in args.model_configs:
        parts = config_str.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid model configuration format: {config_str}, should be model_id:origin_file_pattern")
        model_configs.append({
            "model_id": parts[0],
            "origin_file_pattern": parts[1]
        })

    # Parse processor configuration
    parts = args.processor_config.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid processor configuration format: {args.processor_config}, should be model_id:origin_file_pattern")
    processor_config = {
        "model_id": parts[0],
        "origin_file_pattern": parts[1]
    }

    # Parse tokenizer configuration (optional)
    tokenizer_config = None
    if args.tokenizer_config:
        parts = args.tokenizer_config.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid tokenizer configuration format: {args.tokenizer_config}, should be model_id:origin_file_pattern")
        tokenizer_config = {
            "model_id": parts[0],
            "origin_file_pattern": parts[1]
        }

    # Parse max_input_pixels (may be JSON string)
    max_input_pixels = None
    if args.max_input_pixels:
        try:
            import json
            max_input_pixels = json.loads(args.max_input_pixels)
        except json.JSONDecodeError:
            # If not JSON, try parsing as int
            try:
                max_input_pixels = int(args.max_input_pixels)
            except ValueError:
                raise ValueError(f"Cannot parse max_input_pixels: {args.max_input_pixels}")

    # Default dynamic resolution
    if max_input_pixels is None:
        max_input_pixels = [1048576, 1048576, 589824, 589824, 589824, 262144, 262144, 262144, 262144, 262144]

    # Build inference hyperparameters
    inference_hyper = dict(
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        edit_image_auto_resize=args.edit_image_auto_resize,
        zero_cond_t=args.zero_cond_t,
    )

    # If max_input_pixels is specified, add to inference_hyper
    if max_input_pixels is not None:
        inference_hyper['max_input_pixels'] = max_input_pixels

    # Run inference
    exit_code = run_inference(
        model_configs=model_configs,
        processor_config=processor_config,
        task=args.task,
        image_num_category=args.image_num_category,
        output_dir=args.output_dir,
        device=args.device,
        torch_dtype=args.torch_dtype,
        inference_hyper=inference_hyper,
        num_workers=args.num_workers,
        use_refine_prompt=getattr(args, 'use_refine_prompt', False),
        model_base_path=getattr(args, 'model_base_path', None),
        transformer_path=getattr(args, 'transformer_path', None),
        tokenizer_config=tokenizer_config,
        lora_path=getattr(args, 'lora_path', None),
        data_root=getattr(args, 'data_root', None),
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
