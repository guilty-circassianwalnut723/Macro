#!/usr/bin/env python3
"""
OmniGen2 inference run script.

Read config.yaml and run evaluation tasks for selected checkpoints.
Only supports: customization / illustration / spatial / temporal.
"""

import os
import sys
import json
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
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_checkpoints(config: dict) -> None:
    checkpoints = config.get("checkpoints", {})
    if not checkpoints:
        print("No checkpoints configured")
        return

    print("\nAvailable checkpoints:")
    print("=" * 60)
    for name, ckpt_config in checkpoints.items():
        model_path = ckpt_config.get("model_path", "N/A")
        transformer_path = ckpt_config.get("transformer_path", "default")
        is_trained = ckpt_config.get("is_trained", False)
        tasks = ckpt_config.get("tasks", {})

        print(f"\n  {name}:")
        print(f"    model_path: {model_path}")
        print(f"    transformer_path: {transformer_path}")
        print(f"    type: {'trained checkpoint' if is_trained else 'pretrained'}")
        print("    tasks:")
        for task, categories in tasks.items():
            cats = ", ".join(categories) if isinstance(categories, list) else str(categories)
            print(f"      - {task}: [{cats}]")
    print("\n" + "=" * 60)


def check_and_convert_checkpoint(exp_dir: str, checkpoint_step: int, convert_script: str) -> str:
    """Convert training checkpoint to transformer directory if needed."""
    exp_dir = Path(exp_dir)
    checkpoint_dir = exp_dir / f"checkpoint-{checkpoint_step}"
    transformer_dir = checkpoint_dir / "transformer"

    if transformer_dir.exists():
        print(f"  transformer already exists: {transformer_dir}")
        return str(transformer_dir)

    model_bin = checkpoint_dir / "model.bin"
    if not model_bin.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_dir}")

    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"convert script not found: {convert_script}")

    cmd = [
        sys.executable,
        convert_script,
        "--model_checkpoint_dir",
        str(checkpoint_dir),
    ]
    print(f"  converting checkpoint: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"failed to convert checkpoint: {checkpoint_dir}")

    print(f"  checkpoint converted: {transformer_dir}")
    return str(transformer_dir)


def _normalize_inference_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy keys to inference.py argument names."""
    normalized = dict(params)

    # legacy: num_inference_steps -> num_inference_step
    if "num_inference_steps" in normalized and "num_inference_step" not in normalized:
        normalized["num_inference_step"] = normalized.pop("num_inference_steps")

    # legacy: guidance_scale -> text_guidance_scale
    if "guidance_scale" in normalized and "text_guidance_scale" not in normalized:
        normalized["text_guidance_scale"] = normalized.pop("guidance_scale")

    # legacy: max_input_pixels -> max_input_image_pixels
    if "max_input_pixels" in normalized and "max_input_image_pixels" not in normalized:
        normalized["max_input_image_pixels"] = normalized.pop("max_input_pixels")

    return normalized


def run_inference(
    ckpt_name: str,
    task: str,
    categories: List[str],
    output_dir: str,
    global_config: dict,
    ckpt_config: dict,
    num_gpus: int = 1,
) -> int:
    model_path = ckpt_config.get("model_path", global_config.get("model_path"))
    transformer_path = ckpt_config.get("transformer_path")
    data_root = global_config.get("data_root")

    if not model_path:
        raise ValueError(f"model_path is missing for checkpoint: {ckpt_name}")
    if not data_root:
        raise ValueError("global_config.data_root is required")

    default_params = global_config.get("inference_params", global_config.get("inference_hyper", {}))
    ckpt_params = ckpt_config.get("inference_params", ckpt_config.get("inference_hyper", {}))
    inference_params = _normalize_inference_params({**default_params, **ckpt_params})

    # Keep compatibility for old top-level max_input_pixels settings.
    max_input_pixels = ckpt_config.get("max_input_pixels", global_config.get("default_max_input_pixels"))
    if max_input_pixels is not None and "max_input_image_pixels" not in inference_params:
        inference_params["max_input_image_pixels"] = max_input_pixels

    data_dir = os.path.join(data_root, task, "eval")
    inference_script = SCRIPT_DIR / "inference.py"

    cmd = ["accelerate", "launch"]
    if num_gpus > 1:
        cmd.extend(["--num_processes", str(num_gpus)])
    cmd.extend([
        str(inference_script),
        "--model_path",
        model_path,
        "--model_name",
        ckpt_name,
        "--task_type",
        task,
        "--data_dir",
        data_dir,
        "--result_dir",
        output_dir,
    ])

    if categories:
        cmd.extend(["--categories", *categories])

    if transformer_path:
        cmd.extend(["--transformer_path", transformer_path])

    for key, value in inference_params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif isinstance(value, (list, tuple)):
            cmd.extend([f"--{key}", json.dumps(value)])
        else:
            cmd.extend([f"--{key}", str(value)])

    print("\nExecuting inference command:")
    print("  " + " ".join(cmd))
    return subprocess.run(cmd).returncode


def run_checkpoint(
    ckpt_name: str,
    ckpt_config: dict,
    global_config: dict,
    filter_task: Optional[str] = None,
    filter_category: Optional[str] = None,
    num_gpus: int = 1,
    output_root_override: Optional[str] = None,
) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"Processing checkpoint: {ckpt_name}")
    print(f"{'=' * 60}")

    tasks_config = ckpt_config.get("tasks", {})
    output_root = output_root_override or global_config.get("output_root", "./outputs")

    results = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "details": [],
    }

    # Align with mvb_new: support trained checkpoints that need conversion.
    is_trained = ckpt_config.get("is_trained", False)
    if is_trained and not ckpt_config.get("transformer_path"):
        exp_dir = ckpt_config.get("exp_dir")
        checkpoint_step = ckpt_config.get("checkpoint_step")
        convert_script = global_config.get("convert_script", str(SCRIPT_DIR / "convert_checkpoint.py"))
        if not exp_dir or not checkpoint_step:
            err = "trained checkpoint requires exp_dir and checkpoint_step"
            print(f"Error: {err}")
            results["failed"] += 1
            results["details"].append({"task": "convert", "status": "failed", "error": err})
            return results
        try:
            ckpt_config["transformer_path"] = check_and_convert_checkpoint(
                exp_dir=exp_dir,
                checkpoint_step=checkpoint_step,
                convert_script=convert_script,
            )
        except Exception as e:
            results["failed"] += 1
            results["details"].append({"task": "convert", "status": "failed", "error": str(e)})
            print(f"Checkpoint conversion failed: {e}")
            return results

    for task, categories_cfg in tasks_config.items():
        if filter_task and task != filter_task:
            continue

        if task not in SUPPORTED_TASKS:
            # Explicitly ignore unsupported tasks in Macro open-source workflow.
            continue

        if categories_cfg == "all":
            categories = IMAGE_NUM_CATEGORIES
        elif isinstance(categories_cfg, list):
            categories = categories_cfg
        elif isinstance(categories_cfg, str):
            categories = [categories_cfg]
        else:
            print(f"Warning: invalid categories config for {task}, skip")
            continue

        for category in categories:
            if filter_category and category != filter_category:
                continue

            output_dir = os.path.join(output_root, ckpt_name, task, category)
            finish_file = Path(output_dir) / ".finish"
            if finish_file.exists():
                print(f"\n--- Task: {task}, Category: {category} ---")
                print("  Skip (already completed)")
                continue

            results["total"] += 1
            print(f"\n--- Task: {task}, Category: {category} ---")

            try:
                exit_code = run_inference(
                    ckpt_name=ckpt_name,
                    task=task,
                    categories=[category],
                    output_dir=output_dir,
                    global_config=global_config,
                    ckpt_config=ckpt_config,
                    num_gpus=num_gpus,
                )
                if exit_code == 0:
                    results["success"] += 1
                    results["details"].append({"task": task, "category": category, "status": "success"})
                    print("  ✓ Done")
                else:
                    results["failed"] += 1
                    results["details"].append(
                        {"task": task, "category": category, "status": "failed", "exit_code": exit_code}
                    )
                    print(f"  ✗ Failed (exit code: {exit_code})")
            except Exception as e:
                results["failed"] += 1
                results["details"].append(
                    {"task": task, "category": category, "status": "failed", "error": str(e)}
                )
                print(f"  ✗ Exception: {e}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniGen2 inference run script")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="config file path")
    parser.add_argument("--list", action="store_true", help="list available checkpoints")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint name")
    parser.add_argument("--task", type=str, default=None, choices=SUPPORTED_TASKS, help="task filter")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=IMAGE_NUM_CATEGORIES + ["all"],
        help="category filter",
    )
    parser.add_argument("--num_gpus", type=int, default=None, help="override num gpus for accelerate")
    parser.add_argument("--output_root", type=str, default=None, help="override output root")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    global_config = config.get("global_config", {})
    checkpoints = config.get("checkpoints", {})

    if args.list:
        list_checkpoints(config)
        return

    if args.ckpt:
        if args.ckpt not in checkpoints:
            print(f"Checkpoint not found: {args.ckpt}")
            print("Available checkpoints:")
            for name in checkpoints.keys():
                print(f"  - {name}")
            sys.exit(1)
        ckpts_to_run = {args.ckpt: checkpoints[args.ckpt]}
    else:
        ckpts_to_run = checkpoints

    num_gpus = args.num_gpus if args.num_gpus is not None else global_config.get("num_gpus", 1)

    all_results = {}
    total_success = 0
    total_failed = 0

    for ckpt_name, ckpt_config in ckpts_to_run.items():
        results = run_checkpoint(
            ckpt_name=ckpt_name,
            ckpt_config=ckpt_config,
            global_config=global_config,
            filter_task=args.task,
            filter_category=args.category if args.category != "all" else None,
            num_gpus=num_gpus,
            output_root_override=args.output_root,
        )
        all_results[ckpt_name] = results
        total_success += results["success"]
        total_failed += results["failed"]

    print(f"\nTotal: Success {total_success}, Failed {total_failed}")
    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
