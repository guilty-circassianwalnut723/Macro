"""
Generic OmniGen inference script for Macro.
Only supports four tasks:
- customization
- illustration
- spatial
- temporal
"""

import dotenv

dotenv.load_dotenv(override=True)

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

# Avoid triton cache issues
os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.triton_cache"))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image, to_tensor

SCRIPT_DIR = Path(__file__).parent.absolute()
MACRO_DIR = SCRIPT_DIR.parent.parent

macro_dir = str(MACRO_DIR)
if macro_dir not in sys.path:
    sys.path.insert(0, macro_dir)

source_dir = str(MACRO_DIR / "omnigen" / "source")
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.models.transformers import transformer_omnigen2

# Register module to fix diffusers loading (avoids "No module named 'transformer_omnigen2'" error)
sys.modules['transformer_omnigen2'] = transformer_omnigen2


SUPPORTED_TASKS = ["customization", "illustration", "spatial", "temporal"]
IMAGE_NUM_CATEGORIES = ["1-3", "4-5", "6-7", ">=8"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OmniGen2 inference script for Macro")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for output directory")
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=SUPPORTED_TASKS,
        help="Task type",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory, usually .../data/filter/<task>/eval",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="euler",
        choices=["euler", "dpmsolver"],
        help="Scheduler to use",
    )
    parser.add_argument("--num_inference_step", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument(
        "--max_input_image_pixels",
        type=str,
        default=None,
        help="int or JSON list string",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text_guidance_scale", type=float, default=5.0)
    parser.add_argument("--image_guidance_scale", type=float, default=2.0)
    parser.add_argument("--cfg_range_start", type=float, default=0.0)
    parser.add_argument("--cfg_range_end", type=float, default=1.0)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "(((deformed))), blurry, over saturation, bad anatomy, disfigured, "
            "poorly drawn face, mutation, mutated, (extra_limb), (ugly), "
            "(poorly drawn hands), fused fingers, messy drawing, broken legs censor, "
            "censored, censor_bar"
        ),
    )
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--enable_model_cpu_offload", action="store_true")
    parser.add_argument("--enable_sequential_cpu_offload", action="store_true")
    parser.add_argument("--enable_group_offload", action="store_true")
    parser.add_argument("--disable_align_res", action="store_true")
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument(
        "--max_ref_images",
        type=int,
        default=None,
        help=(
            "If larger than pretrained max_ref_images, image_index_embedding will be extended "
            "with random init (std=0.02)"
        ),
    )
    parser.add_argument(
        "--max_ref_images_strict",
        action="store_true",
        help=(
            "Do NOT extend image_index_embedding. Require embedding length == max_ref_images, "
            "otherwise exit"
        ),
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate images even if outputs already exist",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        choices=IMAGE_NUM_CATEGORIES,
        help="Subset categories to run",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help="Append _step<checkpoint_step> to output directory name",
    )
    return parser.parse_args()


def load_pipeline(args: argparse.Namespace, accelerator: Accelerator, weight_dtype: torch.dtype) -> OmniGen2Pipeline:
    cache_dir = os.path.expanduser("~/.cache/huggingface/modules/diffusers_modules")
    local_dir = os.path.join(cache_dir, "local")
    os.makedirs(local_dir, exist_ok=True)

    try:
        pipeline = OmniGen2Pipeline.from_pretrained(
            args.model_path,
            torch_dtype=weight_dtype,
            trust_remote_code=True,
        )
    except (OSError, RuntimeError, AttributeError, FileNotFoundError) as e:
        error_str = str(e).lower()
        should_clear_cache = (
            "could not get source code" in str(e)
            or "triton" in error_str
            or "has no attribute" in str(e)
            or "scheduling" in error_str
            or "scheduler" in error_str
            or "transformer_omnigen2" in error_str
            or "no such file or directory" in error_str
        )
        if not should_clear_cache:
            raise

        print(f"Warning: failed to load pipeline once: {e}")
        print("Clearing diffusers local cache and retrying...")

        if os.path.exists(local_dir):
            for py_file in glob.glob(os.path.join(local_dir, "*.py")):
                try:
                    os.remove(py_file)
                except Exception:
                    pass

        pipeline = OmniGen2Pipeline.from_pretrained(
            args.model_path,
            torch_dtype=weight_dtype,
            trust_remote_code=True,
        )

    if args.transformer_path:
        print(f"Transformer weights loaded from {args.transformer_path}")
        transformer = OmniGen2Transformer2DModel.from_pretrained(
            args.transformer_path,
            torch_dtype=weight_dtype,
        )
    else:
        transformer = OmniGen2Transformer2DModel.from_pretrained(
            args.model_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )

    pretrained_max_ref_images = getattr(transformer.config, "max_ref_images", 5)
    embedding_len = transformer.image_index_embedding.shape[0]
    target_max_ref_images = getattr(args, "max_ref_images", None)
    max_ref_images_strict = getattr(args, "max_ref_images_strict", False)

    if max_ref_images_strict:
        if target_max_ref_images is None:
            print("Error: max_ref_images_strict requires --max_ref_images")
            sys.exit(1)
        if embedding_len != target_max_ref_images:
            print(
                "Error: max_ref_images_strict requires image_index_embedding length == max_ref_images. "
                f"Current embedding_len={embedding_len}, max_ref_images={target_max_ref_images}"
            )
            sys.exit(1)
        print(
            "max_ref_images_strict: verified image_index_embedding length "
            f"({embedding_len}) equals max_ref_images ({target_max_ref_images})"
        )
    elif target_max_ref_images is not None and target_max_ref_images > pretrained_max_ref_images:
        print(
            "===== Extending image_index_embedding from "
            f"{pretrained_max_ref_images} to {target_max_ref_images} ====="
        )
        hidden_size = transformer.config.hidden_size
        device = transformer.image_index_embedding.device
        dtype = transformer.image_index_embedding.dtype

        new_image_index_embedding = torch.zeros(
            target_max_ref_images,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        new_image_index_embedding[:pretrained_max_ref_images] = transformer.image_index_embedding.data
        nn.init.normal_(new_image_index_embedding[pretrained_max_ref_images:], std=0.02)

        transformer.image_index_embedding = nn.Parameter(new_image_index_embedding)
        transformer.config.max_ref_images = target_max_ref_images

    pipeline.transformer = transformer

    if args.scheduler == "dpmsolver":
        from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )
        pipeline.scheduler = scheduler

    if args.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    elif args.enable_group_offload:
        apply_group_offloading(
            pipeline.transformer,
            onload_device=accelerator.device,
            offload_type="block_level",
            num_blocks_per_group=2,
            use_stream=True,
        )
        apply_group_offloading(
            pipeline.mllm,
            onload_device=accelerator.device,
            offload_type="block_level",
            num_blocks_per_group=2,
            use_stream=True,
        )
        apply_group_offloading(
            pipeline.vae,
            onload_device=accelerator.device,
            offload_type="block_level",
            num_blocks_per_group=2,
            use_stream=True,
        )
    else:
        pipeline = pipeline.to(accelerator.device)

    if accelerator.num_processes > 1:
        try:
            import torch.distributed as dist

            if dist.is_initialized() and dist.is_available():
                dist.barrier()
        except Exception:
            pass

    return pipeline


def _resolve_data_root(data_dir: str) -> Path:
    """Resolve data root directory from the given data_dir path.

    Accepts two conventions:
    - ``.../data/filter/{task}/eval`` – returns ``.../data/filter``
    - any other path – returned as-is (treated as the filter root)
    """
    data_path = Path(data_dir)
    if data_path.name == "eval":
        # data_dir = .../filter/{task}/eval  →  data_root = .../filter
        return data_path.parent.parent
    return data_path


def preprocess_images(input_image_paths: List[str]) -> List[Image.Image]:
    input_images = []
    for path in input_image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            if img.size[0] == 0 or img.size[1] == 0:
                continue
            input_images.append(img)
        except Exception as e:
            print(f"Warning: failed to preprocess image {path}: {e}")
            continue
    return input_images


def run_generation(
    args: argparse.Namespace,
    accelerator: Accelerator,
    pipeline: OmniGen2Pipeline,
    instruction: str,
    negative_prompt: str,
    input_images: List[Image.Image],
    max_input_pixels=None,
):
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if max_input_pixels is not None:
        if isinstance(max_input_pixels, (list, tuple)):
            num_images = len(input_images) if input_images else 0
            if num_images > 0:
                idx = min(num_images - 1, len(max_input_pixels) - 1)
                idx = max(0, idx)
                max_pixels = max_input_pixels[idx]
            else:
                max_pixels = max_input_pixels[0] if len(max_input_pixels) > 0 else 1048576
        else:
            max_pixels = max_input_pixels
    else:
        max_pixels = 1048576

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=args.width,
        height=args.height,
        align_res=not args.disable_align_res,
        num_inference_steps=args.num_inference_step,
        max_sequence_length=32768,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        cfg_range=(args.cfg_range_start, args.cfg_range_end),
        negative_prompt=negative_prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        generator=generator,
        output_type="pil",
        max_pixels=max_pixels,
    )
    return results.images


def create_collage(images: List[torch.Tensor]) -> Image.Image:
    max_height = max(img.shape[-2] for img in images)
    total_width = sum(img.shape[-1] for img in images)
    canvas = torch.zeros((3, max_height, total_width), device=images[0].device)

    current_x = 0
    for img in images:
        h, w = img.shape[-2:]
        canvas[:, :h, current_x:current_x + w] = img * 0.5 + 0.5
        current_x += w

    return to_pil_image(canvas)


def load_samples_from_evaluation(task_type: str, data_dir: str, categories: List[str] = None) -> List[dict]:
    from utils import load_data_for_task, IMAGE_NUM_CATEGORIES

    data_root = _resolve_data_root(data_dir)

    all_samples = []
    categories_to_load = categories if categories else IMAGE_NUM_CATEGORIES

    # For regular tasks, load each category explicitly
    for category in categories_to_load:
        try:
            samples = load_data_for_task(
                task=task_type,
                image_num_category=category,
                data_root=data_root,
            )
            for i, sample in enumerate(samples):
                idx = sample.get("idx", i)
                all_samples.append(
                    {
                        "sample_id": f"{idx:08d}",
                        "category": category,
                        "instruction": sample.get("prompt", sample.get("instruction", "")),
                        "reference_images": sample.get("input_images", []),
                        "target_image": sample.get("output_image", ""),
                        "idx": idx,
                    }
                )
        except Exception as e:
            print(f"Warning: failed to load samples for {task_type}/{category}: {e}")
            continue

    return all_samples


def main(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    max_input_pixels = None
    if args.max_input_image_pixels:
        try:
            max_input_pixels = json.loads(args.max_input_image_pixels)
        except json.JSONDecodeError:
            try:
                max_input_pixels = int(args.max_input_image_pixels)
            except ValueError:
                raise ValueError(f"Cannot parse max_input_image_pixels: {args.max_input_image_pixels}")
    else:
        max_input_pixels = [1048576, 1048576, 589824, 589824, 589824, 262144, 262144, 262144, 262144, 262144]

    accelerator = Accelerator(mixed_precision=args.dtype if args.dtype != "fp32" else "no")

    process_index = accelerator.process_index
    num_processes = accelerator.num_processes
    processed = 0
    skipped = 0
    failed = 0

    if accelerator.is_main_process:
        print(f"Seed: {args.seed}")
        print(f"Task: {args.task_type}")

    try:
        all_samples = load_samples_from_evaluation(args.task_type, args.data_dir, args.categories)
        if accelerator.is_main_process:
            print(f"Found {len(all_samples)} samples")

        # Shard samples by process index
        process_samples = all_samples[process_index::num_processes]
        print(
            f"Process {process_index}: handling {len(process_samples)} samples "
            f"(total processes: {num_processes})"
        )

        weight_dtype = torch.float32
        if args.dtype == "fp16":
            weight_dtype = torch.float16
        elif args.dtype == "bf16":
            weight_dtype = torch.bfloat16

        pipeline = load_pipeline(args, accelerator, weight_dtype)

        if args.result_dir is None:
            args.result_dir = os.path.join(args.data_dir, "outputs", args.model_name)

        if args.checkpoint_step is not None:
            base_dir = os.path.dirname(args.result_dir)
            model_name = os.path.basename(args.result_dir)
            args.result_dir = os.path.join(base_dir, f"{model_name}_step{args.checkpoint_step}")

        os.makedirs(args.result_dir, exist_ok=True)

        with tqdm(
            total=len(process_samples),
            desc=f"Generating (proc {process_index})",
            unit="sample",
            disable=not accelerator.is_main_process,
        ) as pbar:
            for sample in process_samples:
                sample_id = sample["sample_id"]
                save_idx = sample.get("idx", int(sample_id) if sample_id.isdigit() else 0)
                instruction = sample["instruction"]
                input_image_paths = sample["reference_images"]

                output_subdir = args.result_dir
                os.makedirs(output_subdir, exist_ok=True)
                output_image_path = os.path.join(output_subdir, f"{sample_id}.jpg")

                if not args.force_regenerate and os.path.exists(output_image_path):
                    try:
                        with Image.open(output_image_path) as test_img:
                            test_img.verify()
                        with Image.open(output_image_path) as test_img:
                            test_img.load()
                        skipped += 1
                        pbar.update(1)
                        continue
                    except Exception:
                        try:
                            os.remove(output_image_path)
                        except Exception:
                            pass

                valid_input_paths = []
                for img_path in input_image_paths:
                    if isinstance(img_path, (str, Path)) and os.path.exists(str(img_path)):
                        valid_input_paths.append(str(img_path))

                if not valid_input_paths:
                    failed += 1
                    print(f"Error: no valid input images for sample {sample_id}")
                    pbar.update(1)
                    continue

                try:
                    input_images = preprocess_images(valid_input_paths)
                    generated_images = run_generation(
                        args,
                        accelerator,
                        pipeline,
                        instruction,
                        args.negative_prompt,
                        input_images,
                        max_input_pixels=max_input_pixels,
                    )

                    if len(generated_images) == 1:
                        generated_images[0].save(output_image_path)
                    else:
                        for i, img in enumerate(generated_images):
                            image_name, ext = os.path.splitext(output_image_path)
                            img.save(f"{image_name}_{i}{ext}")

                        # Keep a stitched preview for easier inspection
                        vis_images = [to_tensor(image) * 2 - 1 for image in generated_images]
                        preview = create_collage(vis_images)
                        preview.save(output_image_path)

                    from utils import save_sample

                    sample_for_save = {
                        "prompt": instruction,
                        "instruction": instruction,
                        "input_images": valid_input_paths,
                        "output_image": sample.get("target_image", ""),
                        "task": args.task_type,
                        "idx": save_idx,
                    }

                    save_sample(
                        output_dir=Path(output_subdir),
                        idx=save_idx,
                        sample=sample_for_save,
                        output_image_path=output_image_path,
                        target_image_path=sample.get("target_image", ""),
                        seed=None,
                    )

                    processed += 1
                except Exception as e:
                    failed += 1
                    print(f"Error processing sample {sample_id}: {e}")

                pbar.update(1)
                pbar.set_postfix_str(f"processed={processed}, skipped={skipped}, failed={failed}")

    except Exception as e:
        print(f"Process {process_index}: fatal error: {e}")
        failed += 1

    # Aggregate stats across processes
    global_failed = failed
    global_processed = processed
    global_skipped = skipped
    if num_processes > 1:
        try:
            import torch.distributed as dist

            if dist.is_initialized() and dist.is_available():
                failed_tensor = torch.tensor([failed], device=accelerator.device, dtype=torch.int64)
                processed_tensor = torch.tensor([processed], device=accelerator.device, dtype=torch.int64)
                skipped_tensor = torch.tensor([skipped], device=accelerator.device, dtype=torch.int64)
                dist.all_reduce(failed_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(processed_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(skipped_tensor, op=dist.ReduceOp.SUM)
                global_failed = int(failed_tensor.item())
                global_processed = int(processed_tensor.item())
                global_skipped = int(skipped_tensor.item())
        except Exception as e:
            print(f"Process {process_index}: failed to aggregate stats: {e}")

    if accelerator.is_main_process:
        print("\nGeneration completed")
        print(f"  Processed: {global_processed}")
        print(f"  Skipped: {global_skipped}")
        print(f"  Failed: {global_failed}")
        print(f"  Results: {args.result_dir}")

        if global_failed == 0:
            finish_file = Path(args.result_dir) / ".finish"
            try:
                finish_file.touch()
                os.chmod(finish_file, 0o777)
                print(f"  Created finish flag: {finish_file}")
            except Exception as e:
                print(f"  Warning: failed to create finish flag: {e}")

    return 0 if global_failed == 0 else 1


if __name__ == "__main__":
    cli_args = parse_args()
    try:
        exit_code = main(cli_args)
        sys.exit(exit_code)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
