#!/usr/bin/env python3
"""
Evaluation runner script

Read configuration from config.yaml and run evaluation tasks
Supports LLM scoring tasks (customization, spatial, illustration, temporal)

Usage:
    python run_eval.py                           # Run all configured evaluations
    python run_eval.py --baseline bagel --exp bagel_official  # Run specified evaluation
    python run_eval.py --config my_config.yaml   # Use a custom config file
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
MACRO_DIR = SCRIPT_DIR.parent
EVAL_DIR = SCRIPT_DIR

# Supported baseline models
SUPPORTED_BASELINES = ["bagel", "omnigen", "qwen"]

# LLM scoring tasks
LLM_TASKS = ["customization", "illustration", "spatial", "temporal"]
IMAGE_NUM_CATEGORIES = ["1-3", "4-5", "6-7", ">=8"]

# Third-party evaluation tasks (not included in this version)
THIRDPARTY_TASKS = []


def load_config(config_path: Path) -> dict:
    """Load config file. Uses FullLoader to support YAML anchors (*) and merge keys (<<)."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def run_llm_evaluation(
    baseline: str,
    exp_name: str,
    task: str,
    category: str,
    output_root: str,
    use_gpt: bool = True,
    use_gemini: bool = True,
    max_samples: Optional[int] = None,
) -> int:
    """
    Run an LLM scoring task
    
    Args:
        baseline: model type (includes api)
        exp_name: experiment name (gpt/seed/nano for api)
        task: task type
        category: image count category
        output_root: output root directory (generation result root; for api, must match run script)
        use_gpt: whether to use GPT scoring
        use_gemini: whether to use Gemini scoring
        max_samples: max samples to evaluate per (task, category); None means all
        
    Returns:
        exit code
    """
    evaluate_script = EVAL_DIR / "evaluate.py"
    
    cmd = [
        sys.executable, str(evaluate_script),
        "--baseline", baseline,
        "--exp_name", exp_name,
        "--task", task,
        "--image_num_category", category,
    ]
    if output_root:
        cmd.extend(["--output_root", output_root])
    if max_samples is not None and max_samples > 0:
        cmd.extend(["--max_samples", str(max_samples)])
    
    if use_gpt:
        cmd.append("--use_gpt")
    else:
        cmd.append("--no_gpt")
    
    if use_gemini:
        cmd.append("--use_gemini")
    else:
        cmd.append("--no_gemini")
    
    print(f"\nRunning LLM scoring: {' '.join(cmd)}")
    
    # Pass current environment variables, including API keys set in main
    env = os.environ.copy()
    result = subprocess.run(cmd, env=env)
    return result.returncode


def run_thirdparty_evaluation(
    baseline: str,
    exp_name: str,
    task: str,
    output_root: str,
    global_config: dict
) -> int:
    """
    Run a third-party evaluation task
    
    Args:
        baseline: model type
        exp_name: experiment name
        task: task type (omnicontext, geneval, dpg)
        output_root: output root directory
        global_config: global configuration
        
    Returns:
        exit code
    """
    output_dir = Path(output_root) / baseline / exp_name / task
    
    if task == "omnicontext":
        # Run omnicontext evaluation
        eval_script = THIRDPARTY_DIR / "omnicontext_eval.py"
        cmd = [
            sys.executable, str(eval_script),
            "--output_dir", str(output_dir),
            "--openai_url", global_config.get("openai_url", os.environ.get("OPENAI_URL", "")),
            "--openai_key", global_config.get("openai_key", os.environ.get("OPENAI_KEY", "")),
            "--max_workers", str(global_config.get("parallel_workers", 4)),
        ]
        
    elif task == "geneval":
        # Run geneval evaluation
        # geneval requires bench directory (contains {idx:05d}/samples/{seed:04d}.png structure)
        bench_dir = output_dir / "bench"
        if not bench_dir.exists():
            print(f"Warning: bench directory not found: {bench_dir}")
            print(f"Please ensure inference has successfully created the bench directory")
            return 1
        
        eval_script = THIRDPARTY_DIR / "geneval_eval.py"
        cmd = [
            sys.executable, str(eval_script),
            "--image_path", str(bench_dir),
        ]
        
    elif task == "dpg":
        # Run dpg evaluation
        # dpg requires bench directory (contains grid image files)
        bench_dir = output_dir / "bench"
        if not bench_dir.exists():
            print(f"Warning: bench directory not found: {bench_dir}")
            print(f"Please ensure inference has successfully created the bench directory")
            return 1
        
        eval_script = THIRDPARTY_DIR / "dpg_eval.py"
        cmd = [
            sys.executable, str(eval_script),
            "--image_path", str(bench_dir),
            "--gpus", "2",
            "--resolution", "512",
            "--padding", "0",
        ]
        
    else:
        print(f"Error: unsupported task type: {task}")
        return 1
    
    print(f"\nRunning third-party evaluation {task}: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def run_evaluation(
    baseline: str,
    exp_name: str,
    tasks_config: dict,
    output_root: str,
    global_config: dict,
    exp_config: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Run all evaluation tasks for a single experiment
    
    Args:
        baseline: model type
        exp_name: experiment name
        tasks_config: task configuration
        output_root: output root directory (from global_config, can be overridden by exp)
        global_config: global configuration
        exp_config: configuration for the current experiment, may include output_root override (for when baseline results are not under global output_root)
        
    Returns:
        run result statistics
    """
    # Allow this experiment to independently specify output_root (consistent with inference write path)
    if exp_config and exp_config.get("output_root"):
        output_root = exp_config["output_root"]
        print(f"  Using experiment-specified output_root: {output_root}")
    print(f"\n{'='*60}")
    print(f"Processing evaluation: {baseline}/{exp_name}")
    print(f"{'='*60}")
    
    results = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'details': []
    }
    
    # Read LLM evaluation count limit from task config (only for customization/illustration/spatial/temporal)
    max_samples = tasks_config.get("max_samples")

    # Process each task
    for task, task_config in tasks_config.items():
        if task == "max_samples":
            continue
        if task in LLM_TASKS:
            # LLM scoring tasks
            if isinstance(task_config, str):
                if task_config == "all":
                    categories = IMAGE_NUM_CATEGORIES
                else:
                    categories = [task_config]
            elif isinstance(task_config, list):
                categories = task_config
            else:
                print(f"Warning: configuration format for task {task} is incorrect, skipping")
                continue
            
            for category in categories:
                results['total'] += 1
                print(f"\n--- LLM scoring task: {task}, category: {category} ---")
                
                try:
                    # Read LLM scoring config from global_config, defaults: use_gpt=False, use_gemini=True
                    use_gpt = global_config.get("use_gpt", False)
                    use_gemini = global_config.get("use_gemini", True)
                    
                    exit_code = run_llm_evaluation(
                        baseline=baseline,
                        exp_name=exp_name,
                        task=task,
                        category=category,
                        output_root=output_root or "",
                        use_gpt=use_gpt,
                        use_gemini=use_gemini,
                        max_samples=max_samples,
                    )
                    
                    if exit_code == 0:
                        results['success'] += 1
                        results['details'].append({
                            'task': task,
                            'category': category,
                            'type': 'llm',
                            'status': 'success'
                        })
                        print(f"  ✓ Done")
                    else:
                        results['failed'] += 1
                        results['details'].append({
                            'task': task,
                            'category': category,
                            'type': 'llm',
                            'status': 'failed',
                            'exit_code': exit_code
                        })
                        print(f"  ✗ Failed (exit code: {exit_code})")
                        
                except Exception as e:
                    results['failed'] += 1
                    results['details'].append({
                        'task': task,
                        'category': category,
                        'type': 'llm',
                        'status': 'failed',
                        'error': str(e)
                    })
                    print(f"  ✗ Exception: {e}")
        
        elif task in THIRDPARTY_TASKS:
            # Third-party evaluation task
            # For omnicontext and dpg, execute whenever the key appears
            # For geneval, true/false indicates whether to use refine prompt, but always execute
            if task_config is False:
                # If explicitly set to False, skip (but geneval still executes)
                if task == "geneval":
                    # geneval executes even when false (just does not use refine prompt)
                    pass
                else:
                    continue
            
            results['total'] += 1
            print(f"\n--- Third-party evaluation task: {task} ---")
            
            try:
                exit_code = run_thirdparty_evaluation(
                    baseline=baseline,
                    exp_name=exp_name,
                    task=task,
                    output_root=output_root,
                    global_config=global_config
                )
                
                if exit_code == 0:
                    results['success'] += 1
                    results['details'].append({
                        'task': task,
                        'type': 'thirdparty',
                        'status': 'success'
                    })
                    print(f"  ✓ Done")
                else:
                    results['failed'] += 1
                    results['details'].append({
                        'task': task,
                        'type': 'thirdparty',
                        'status': 'failed',
                        'exit_code': exit_code
                    })
                    print(f"  ✗ Failed (exit code: {exit_code})")
                    
            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'task': task,
                    'type': 'thirdparty',
                    'status': 'failed',
                    'error': str(e)
                })
                print(f"  ✗ Exception: {e}")
        
        else:
            print(f"Warning: unknown task type: {task}, skipping")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluation runner script")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                       help="config file path")
    parser.add_argument("--baseline", type=str, default=None,
                       choices=SUPPORTED_BASELINES,
                       help="specify the baseline model to run")
    parser.add_argument("--exp", type=str, default=None,
                       help="specify the experiment name to run")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    global_config = config.get('global_config', {})
    
    # Set API environment variables from config if provided
    api_config = global_config.get('api_config', {})
    if api_config:
        openai_config = api_config.get('openai', {})
        if openai_config.get('url'):
            os.environ['OPENAI_URL'] = openai_config['url']
        if openai_config.get('key'):
            os.environ['OPENAI_KEY'] = openai_config['key']
            
        gemini_config = api_config.get('gemini', {})
        if gemini_config.get('api_key'):
            os.environ['GEMINI_API_KEY'] = gemini_config['api_key']
        if gemini_config.get('model_name'):
            os.environ['GEMINI_MODEL_NAME'] = gemini_config['model_name']

    evaluations = config.get('evaluations', {})
    
    if not evaluations:
        print("No evaluation tasks in config file, exiting")
        sys.exit(0)
    
    # Determine evaluations to run
    evaluations_to_run = {}
    if args.baseline and args.exp:
        # Both baseline and exp are specified
        if args.baseline == "any":
            print("--baseline any does not support being specified together with --exp, please use evaluations.any config item to list directories")
            sys.exit(1)
        if args.baseline not in evaluations:
            print(f"Baseline not found: {args.baseline}")
            sys.exit(1)
        if args.exp not in evaluations[args.baseline]:
            print(f"Experiment not found: {args.baseline}/{args.exp}")
            sys.exit(1)
        evaluations_to_run[args.baseline] = {args.exp: evaluations[args.baseline][args.exp]}
    else:
        # Run all configured evaluations
        evaluations_to_run = evaluations
    
    # Run evaluations
    all_results = {}
    total_success = 0
    total_failed = 0
    
    for baseline, baseline_config in evaluations_to_run.items():
        if baseline_config is None:
            continue
        if isinstance(baseline_config, list):
            # evaluations.any: list of { path, tasks }, path is the result root directory
            for item in baseline_config:
                path = item.get('path', '').rstrip('/')
                if not path:
                    continue
                exp_name = os.path.basename(path) or path
                tasks_config = item.get('tasks', {})
                exp_config = dict(item)
                exp_config['output_root'] = os.path.dirname(path)
                results = run_evaluation(
                    baseline=baseline,
                    exp_name=exp_name,
                    tasks_config=tasks_config,
                    output_root=global_config.get('output_root', './outputs'),
                    global_config=global_config,
                    exp_config=exp_config,
                )
                all_results[f"{baseline}/{exp_name}"] = results
                total_success += results['success']
                total_failed += results['failed']
        elif isinstance(baseline_config, dict):
            for exp_name, exp_config in baseline_config.items():
                tasks_config = exp_config.get('tasks', {})

                results = run_evaluation(
                    baseline=baseline,
                    exp_name=exp_name,
                    tasks_config=tasks_config,
                    output_root=global_config.get('output_root', './outputs'),
                    global_config=global_config,
                    exp_config=exp_config,
                )

                all_results[f"{baseline}/{exp_name}"] = results
                total_success += results['success']
                total_failed += results['failed']
    
    # Print summary
    print(f"\n{'='*60}")
    print("Evaluation task summary")
    print(f"{'='*60}")
    
    for key, results in all_results.items():
        print(f"\n{key}:")
        print(f"  Success: {results['success']}/{results['total']}")
        print(f"  Failed: {results['failed']}/{results['total']}")
        
        if results['failed'] > 0:
            print("  Failure details:")
            for detail in results['details']:
                if detail['status'] == 'failed':
                    error = detail.get('error', detail.get('exit_code', 'unknown'))
                    task_info = f"{detail['task']}"
                    if 'category' in detail:
                        task_info += f"/{detail['category']}"
                    print(f"    - {task_info}: {error}")
    
    print(f"\nTotal: success {total_success}, failed {total_failed}")
    
    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
