#!/usr/bin/env python3
"""
Qwen-Image-Edit inference run script

Reads configuration from config.yaml and runs inference tasks

Usage:
    python run.py                           # run all configured checkpoints
    python run.py --list                    # list all available checkpoints
    python run.py --ckpt qwen_official      # run a specific checkpoint
    python run.py --ckpt qwen_official --task customization --category 1-3
    python run.py --config my_config.yaml   # use a custom config file
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

SUPPORTED_TASKS = ["customization", "illustration", "spatial", "temporal"]
IMAGE_NUM_CATEGORIES = ["1-3", "4-5", "6-7", ">=8"]


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
        tasks = ckpt_config.get('tasks', {})
        print(f"\n  {name}:")
        print(f"    Path: {path}")
        print(f"    Tasks:")
        for task, categories in tasks.items():
            cats = ', '.join(categories) if isinstance(categories, list) else categories
            print(f"      - {task}: [{cats}]")
    print("\n" + "=" * 60)


def build_inference_command(task: str, category: str, output_dir: str,
                             global_config: dict, ckpt_config: dict) -> List[str]:
    """Build qwen inference command.

    Derives model_configs / processor_config / tokenizer_config automatically
    from model_base_path + model_id, so the config only needs those two fields.
    """
    num_workers = global_config.get('num_workers', 1)
    inference_script = SCRIPT_DIR / "inference.py"

    # Resolve model_base_path and model_id
    model_base_path = ckpt_config.get('model_base_path', global_config.get('model_base_path', './ckpts'))
    model_id = ckpt_config['model_id']

    # Derive standard component patterns from model_id
    model_configs = [
        {'model_id': model_id, 'origin_file_pattern': 'transformer/diffusion_pytorch_model*.safetensors'},
        {'model_id': model_id, 'origin_file_pattern': 'text_encoder/model*.safetensors'},
        {'model_id': model_id, 'origin_file_pattern': 'vae/diffusion_pytorch_model.safetensors'},
    ]
    processor_config = {'model_id': model_id, 'origin_file_pattern': 'processor/'}
    tokenizer_config = {'model_id': model_id, 'origin_file_pattern': 'tokenizer/'}

    default_hyper = global_config.get('inference_hyper', {})
    ckpt_hyper = ckpt_config.get('inference_hyper', {})
    inference_hyper = {**default_hyper, **ckpt_hyper}

    data_root = ckpt_config.get('data_root', global_config.get('data_root'))
    transformer_path = ckpt_config.get('transformer_path')
    max_input_pixels = ckpt_config.get('max_input_pixels', global_config.get('default_max_input_pixels'))

    cmd = [
        sys.executable, str(inference_script),
        "--task", task,
        "--output_dir", output_dir,
        "--image_num_category", category,
        "--num_workers", str(num_workers),
        "--model_base_path", model_base_path,
    ]

    # Add model configs (transformer / text_encoder / vae)
    for config in model_configs:
        cmd.extend(["--model_configs", f"{config['model_id']}:{config['origin_file_pattern']}"])

    # Add processor and tokenizer configs
    cmd.extend(["--processor_config", f"{processor_config['model_id']}:{processor_config['origin_file_pattern']}"])
    cmd.extend(["--tokenizer_config", f"{tokenizer_config['model_id']}:{tokenizer_config['origin_file_pattern']}"])

    if transformer_path:
        cmd.extend(["--transformer_path", transformer_path])

    if data_root:
        cmd.extend(["--data_root", data_root])

    if max_input_pixels is not None:
        import json as _json
        if isinstance(max_input_pixels, (list, tuple)):
            cmd.extend(["--max_input_pixels", _json.dumps(max_input_pixels)])
        else:
            cmd.extend(["--max_input_pixels", str(max_input_pixels)])

    for key, value in inference_hyper.items():
        cmd.extend([f"--{key}", str(value)])

    return cmd


def run_checkpoint(ckpt_name: str, ckpt_config: dict, global_config: dict,
                   filter_task: Optional[str] = None,
                   filter_category: Optional[str] = None) -> Dict[str, Any]:
    """Run all inference tasks for a single checkpoint."""
    print(f"\n{'='*60}")
    print(f"Processing checkpoint: {ckpt_name}")
    print(f"{'='*60}")

    tasks_config = ckpt_config.get('tasks', {})
    output_root = global_config.get('output_root', './outputs')

    results = {'total': 0, 'success': 0, 'failed': 0, 'details': []}

    for task, task_config in tasks_config.items():
        if filter_task and task != filter_task:
            continue
        if task not in SUPPORTED_TASKS:
            print(f"Warning: Unknown task {task}, skipping")
            continue

        categories = IMAGE_NUM_CATEGORIES if task_config == "all" else (
            task_config if isinstance(task_config, list) else [task_config]
        )

        for category in categories:
            if filter_category and category != filter_category:
                continue
            output_dir = os.path.join(output_root, ckpt_name, task, category)
            finish_file = Path(output_dir) / ".finish"
            if finish_file.exists():
                print(f"\n--- Task: {task}, Category: {category} ---")
                print(f"  Skipped (already completed)")
                continue
            results['total'] += 1
            print(f"\n--- Task: {task}, Category: {category} ---")
            try:
                cmd = build_inference_command(
                    task=task, category=category,
                    output_dir=output_dir,
                    global_config=global_config,
                    ckpt_config=ckpt_config,
                )
                print(f"  Executing: {' '.join(cmd)}")
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
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit inference run script")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--task", type=str, default=None, choices=SUPPORTED_TASKS)
    parser.add_argument("--category", type=str, default=None, choices=IMAGE_NUM_CATEGORIES + ["all"])
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
    print(f"\nTotal: Success {total_success}, Failed {total_failed}")
    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
