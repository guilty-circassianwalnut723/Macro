#!/usr/bin/env python3
"""
Bagel 推理运行脚本

从 config.yaml 读取配置并运行推理任务
支持选择特定 checkpoint、任务和类别

使用方式:
    python run.py                           # 运行所有配置的 checkpoints
    python run.py --list                    # 列出所有可用的 checkpoints
    python run.py --ckpt bagel_official     # 运行指定的 checkpoint
    python run.py --ckpt bagel_official --task customization --category 1-3
    python run.py --config my_config.yaml   # 使用自定义配置文件
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"

# Supported tasks and categories
SUPPORTED_TASKS = ["customization", "illustration", "spatial", "temporal"]
IMAGE_NUM_CATEGORIES = ["1-3", "4-5", "6-7", ">=8"]

# Minimum file size for ema.safetensors (about 27GB)
EMA_SAFETENSORS_MIN_BYTES = 27 * (1024 ** 3)


def load_config(config_path: Path) -> dict:
    """Load configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def list_checkpoints(config: dict) -> None:
    """List all available checkpoints."""
    checkpoints = config.get('checkpoints', {})
    if not checkpoints:
        print("No checkpoints configured")
        return
    print("\nAvailable checkpoints:")
    print("=" * 60)
    for name, ckpt_config in checkpoints.items():
        path = ckpt_config.get('path', 'N/A')
        is_safetensor = ckpt_config.get('is_safetensor', True)
        tasks = ckpt_config.get('tasks', {})
        print(f"\n  {name}:")
        print(f"    Path: {path}")
        print(f"    Format: {'SafeTensor' if is_safetensor else 'FSDP (needs conversion)'}")
        print(f"    Tasks:")
        for task, categories in tasks.items():
            if isinstance(categories, list):
                cats = ', '.join(categories)
            else:
                cats = categories
            print(f"      - {task}: [{cats}]")
    print("\n" + "=" * 60)


def check_and_convert_checkpoint(ckpt_path: str, is_safetensor: bool,
                                   base_model_path: Optional[str] = None) -> str:
    """Check and convert checkpoint if needed."""
    ckpt_path = Path(ckpt_path)
    if is_safetensor:
        return str(ckpt_path)
    ema_path = ckpt_path / "ema.safetensors"
    if ema_path.exists():
        size_bytes = ema_path.stat().st_size
        if size_bytes < EMA_SAFETENSORS_MIN_BYTES:
            size_gb = size_bytes / (1024 ** 3)
            print(f"  ema.safetensors exists but too small ({size_gb:.2f} GB < 27 GB): {ema_path}")
        else:
            print(f"  ema.safetensors found: {ema_path}")
            return str(ema_path)
    # Convert FSDP checkpoint
    print(f"  Converting FSDP checkpoint: {ckpt_path}")
    convert_script = SCRIPT_DIR / "convert_fsdp_to_safetensors.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"Conversion script not found: {convert_script}")
    cmd = [
        sys.executable, str(convert_script),
        "--checkpoint_dir", str(ckpt_path),
        "--output_file", str(ema_path),
        "--use_ema"
    ]
    if base_model_path:
        cmd.extend(["--base_model_path", base_model_path])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Conversion failed:")
        print(result.stderr)
        raise RuntimeError(f"Failed to convert checkpoint: {ckpt_path}")
    print(f"  Conversion complete: {ema_path}")
    return str(ema_path)


def build_inference_command(model_path: str, task: str, category: str,
                             output_dir: str, global_config: dict,
                             ckpt_config: dict) -> List[str]:
    """Build inference command."""
    vae_transform = ckpt_config.get('vae_transform', global_config.get('default_vae_transform', [768, 512]))
    vit_transform = ckpt_config.get('vit_transform', global_config.get('default_vit_transform', [336, 224]))
    base_model_path = ckpt_config.get('base_model_path', global_config.get('base_model_path'))
    max_mem_per_gpu = global_config.get('max_mem_per_gpu', '40GiB')
    offload_folder = global_config.get('offload_folder', '/tmp/offload')
    num_workers = global_config.get('num_workers', 1)
    max_input_pixels = ckpt_config.get('max_input_pixels', global_config.get('default_max_input_pixels'))
    data_root = ckpt_config.get('data_root', global_config.get('data_root'))

    default_hyper = global_config.get('inference_hyper', {})
    ckpt_hyper = ckpt_config.get('inference_hyper', {})
    inference_hyper = {**default_hyper, **ckpt_hyper}

    inference_script = SCRIPT_DIR / "inference.py"
    cmd = [
        sys.executable, str(inference_script),
        "--model_path", model_path,
        "--task", task,
        "--output_dir", output_dir,
        "--vae_transform", f"{vae_transform[0]},{vae_transform[1]}",
        "--vit_transform", f"{vit_transform[0]},{vit_transform[1]}",
        "--max_mem_per_gpu", max_mem_per_gpu,
        "--offload_folder", offload_folder,
        "--num_workers", str(num_workers),
        "--image_num_category", category,
    ]
    if base_model_path:
        cmd.extend(["--base_model_path", base_model_path])
    if data_root:
        cmd.extend(["--data_root", data_root])
    for key, value in inference_hyper.items():
        if key == 'cfg_interval':
            cmd.extend([f"--{key}", f"{value[0]},{value[1]}"])
        elif key == 'image_shapes':
            if isinstance(value, (list, tuple)) and len(value) == 2:
                cmd.extend([f"--{key}", f"{value[0]},{value[1]}"])
            else:
                cmd.extend([f"--{key}", str(value)])
        else:
            cmd.extend([f"--{key}", str(value)])
    if max_input_pixels is not None:
        if isinstance(max_input_pixels, (list, tuple)):
            import json
            cmd.extend(["--max_input_pixels", json.dumps(max_input_pixels)])
        else:
            cmd.extend(["--max_input_pixels", str(max_input_pixels)])
    return cmd


def run_checkpoint(ckpt_name: str, ckpt_config: dict, global_config: dict,
                   filter_task: Optional[str] = None,
                   filter_category: Optional[str] = None) -> Dict[str, Any]:
    """Run all inference tasks for a single checkpoint."""
    print(f"\n{'='*60}")
    print(f"Processing checkpoint: {ckpt_name}")
    print(f"{'='*60}")

    ckpt_path = ckpt_config.get('path')
    is_safetensor = ckpt_config.get('is_safetensor', True)
    base_model_path = ckpt_config.get('base_model_path', global_config.get('base_model_path'))
    tasks_config = ckpt_config.get('tasks', {})
    output_root = global_config.get('output_root', './outputs')

    results = {'total': 0, 'success': 0, 'failed': 0, 'details': []}

    try:
        model_path = check_and_convert_checkpoint(ckpt_path, is_safetensor, base_model_path)
    except Exception as e:
        print(f"Checkpoint conversion failed: {e}")
        results['details'].append({'task': 'convert', 'category': 'N/A', 'status': 'failed', 'error': str(e)})
        results['failed'] += 1
        return results

    for task, task_config in tasks_config.items():
        if filter_task and task != filter_task:
            continue
        if task not in SUPPORTED_TASKS:
            print(f"Warning: Unknown task {task}, skipping")
            continue

        if isinstance(task_config, str):
            categories = IMAGE_NUM_CATEGORIES if task_config == "all" else [task_config]
        elif isinstance(task_config, list):
            categories = task_config
        else:
            print(f"Warning: Invalid task config for {task}, skipping")
            continue

        for category in categories:
            if filter_category and category != filter_category:
                continue
            output_dir = os.path.join(output_root, ckpt_name, task, category)
            finish_file = Path(output_dir) / ".finish"
            if finish_file.exists():
                print(f"\n--- Task: {task}, Category: {category} ---")
                print(f"  Skipped (already completed, .finish file exists)")
                continue
            results['total'] += 1
            print(f"\n--- Task: {task}, Category: {category} ---")
            try:
                cmd = build_inference_command(
                    model_path=model_path, task=task, category=category,
                    output_dir=output_dir, global_config=global_config, ckpt_config=ckpt_config
                )
                print(f"  Executing: {chr(32).join(cmd)}")
                result = subprocess.run(cmd)
                if result.returncode == 0:
                    results['success'] += 1
                    results['details'].append({'task': task, 'category': category, 'status': 'success'})
                    print(f"  ✓ Done")
                else:
                    results['failed'] += 1
                    results['details'].append({'task': task, 'category': category, 'status': 'failed', 'exit_code': result.returncode})
                    print(f"  ✗ Failed (exit code: {result.returncode})")
            except Exception as e:
                results['failed'] += 1
                results['details'].append({'task': task, 'category': category, 'status': 'failed', 'error': str(e)})
                print(f"  ✗ Exception: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Bagel inference run script")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Config file path")
    parser.add_argument("--list", action="store_true", help="List all available checkpoints")
    parser.add_argument("--ckpt", type=str, default=None, help="Specify checkpoint name to run")
    parser.add_argument("--task", type=str, default=None, choices=SUPPORTED_TASKS, help="Specify task to run")
    parser.add_argument("--category", type=str, default=None, choices=IMAGE_NUM_CATEGORIES + ["all"], help="Specify category to run")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    config = load_config(config_path)
    global_config = config.get('global_config', {})
    checkpoints = config.get('checkpoints', {})

    if args.list:
        list_checkpoints(config)
        return

    ckpts_to_run = {args.ckpt: checkpoints[args.ckpt]} if args.ckpt else checkpoints

    all_results = {}
    total_success = 0
    total_failed = 0
    for ckpt_name, ckpt_config in ckpts_to_run.items():
        results = run_checkpoint(
            ckpt_name=ckpt_name, ckpt_config=ckpt_config,
            global_config=global_config,
            filter_task=args.task,
            filter_category=args.category if args.category != "all" else None,
        )
        all_results[ckpt_name] = results
        total_success += results['success']
        total_failed += results['failed']

    print(f"\n{'='*60}")
    print("Inference Summary")
    print(f"{'='*60}")
    for ckpt_name, results in all_results.items():
        print(f"\n{ckpt_name}:")
        print(f"  Success: {results['success']}/{results['total']}")
        print(f"  Failed: {results['failed']}/{results['total']}")
        if results['failed'] > 0:
            print("  Failures:")
            for detail in results['details']:
                if detail['status'] == 'failed':
                    error = detail.get('error', detail.get('exit_code', 'unknown'))
                    print(f"    - {detail['task']}/{detail['category']}: {error}")
    print(f"\nTotal: Success {total_success}, Failed {total_failed}")
    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
