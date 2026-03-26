#!/usr/bin/env python3
"""
评测运行脚本

从 config.yaml 读取配置并运行评测任务
支持 LLM 评分任务（customization, spatial, illustration, temporal）

使用方式:
    python run_eval.py                           # 运行所有配置的评测
    python run_eval.py --baseline bagel --exp bagel_official  # 运行指定的评测
    python run_eval.py --config my_config.yaml   # 使用自定义配置文件
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

# 支持的 baseline 模型
SUPPORTED_BASELINES = ["bagel", "omnigen", "qwen"]

# LLM 评分任务
LLM_TASKS = ["customization", "illustration", "spatial", "temporal"]
IMAGE_NUM_CATEGORIES = ["1-3", "4-5", "6-7", ">=8"]

# 第三方评测任务（未包含在此版本中）
THIRDPARTY_TASKS = []


def load_config(config_path: Path) -> dict:
    """加载配置文件。使用 FullLoader 以支持 YAML 锚点（*）与合并键（<<）。"""
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
    运行 LLM 评分任务
    
    Args:
        baseline: 模型类型（含 api）
        exp_name: 实验名称（api 时为 gpt/seed/nano）
        task: 任务类型
        category: 图像数量类别
        output_root: 输出根目录（生成结果根目录，api 时必须与 run 脚本一致）
        use_gpt: 是否使用 GPT 评分
        use_gemini: 是否使用 Gemini 评分
        max_samples: 每个 (task, category) 最多评测的样本数，None 表示全部
        
    Returns:
        退出码
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
    
    print(f"\n运行 LLM 评分: {' '.join(cmd)}")
    
    # 传递当前环境变量，包括在 main 中设置的 API keys
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
    运行第三方评测任务
    
    Args:
        baseline: 模型类型
        exp_name: 实验名称
        task: 任务类型 (omnicontext, geneval, dpg)
        output_root: 输出根目录
        global_config: 全局配置
        
    Returns:
        退出码
    """
    output_dir = Path(output_root) / baseline / exp_name / task
    
    if task == "omnicontext":
        # 运行 omnicontext 评测
        eval_script = THIRDPARTY_DIR / "omnicontext_eval.py"
        cmd = [
            sys.executable, str(eval_script),
            "--output_dir", str(output_dir),
            "--openai_url", global_config.get("openai_url", os.environ.get("OPENAI_URL", "")),
            "--openai_key", global_config.get("openai_key", os.environ.get("OPENAI_KEY", "")),
            "--max_workers", str(global_config.get("parallel_workers", 4)),
        ]
        
    elif task == "geneval":
        # 运行 geneval 评测
        # geneval 需要使用 bench 目录（包含 {idx:05d}/samples/{seed:04d}.png 结构）
        bench_dir = output_dir / "bench"
        if not bench_dir.exists():
            print(f"警告: bench 目录不存在: {bench_dir}")
            print(f"请确保 inference 已成功创建 bench 目录")
            return 1
        
        eval_script = THIRDPARTY_DIR / "geneval_eval.py"
        cmd = [
            sys.executable, str(eval_script),
            "--image_path", str(bench_dir),
        ]
        
    elif task == "dpg":
        # 运行 dpg 评测
        # dpg 需要使用 bench 目录（包含 grid 图像文件）
        bench_dir = output_dir / "bench"
        if not bench_dir.exists():
            print(f"警告: bench 目录不存在: {bench_dir}")
            print(f"请确保 inference 已成功创建 bench 目录")
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
        print(f"错误: 不支持的任务类型: {task}")
        return 1
    
    print(f"\n运行第三方评测 {task}: {' '.join(cmd)}")
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
    运行单个实验的所有评测任务
    
    Args:
        baseline: 模型类型
        exp_name: 实验名称
        tasks_config: 任务配置
        output_root: 输出根目录（来自 global_config，可被 exp 覆盖）
        global_config: 全局配置
        exp_config: 当前实验的配置，可含 output_root 覆盖（用于该 baseline 结果不在 global output_root 下的情况）
        
    Returns:
        运行结果统计
    """
    # 允许该实验单独指定 output_root（与 inference 写入路径一致）
    if exp_config and exp_config.get("output_root"):
        output_root = exp_config["output_root"]
        print(f"  使用实验指定 output_root: {output_root}")
    print(f"\n{'='*60}")
    print(f"处理评测: {baseline}/{exp_name}")
    print(f"{'='*60}")
    
    results = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'details': []
    }
    
    # 从任务配置中读取 LLM 评测数量限制（仅对 customization/illustration/spatial/temporal 生效）
    max_samples = tasks_config.get("max_samples")

    # 处理每个任务
    for task, task_config in tasks_config.items():
        if task == "max_samples":
            continue
        if task in LLM_TASKS:
            # LLM 评分任务
            if isinstance(task_config, str):
                if task_config == "all":
                    categories = IMAGE_NUM_CATEGORIES
                else:
                    categories = [task_config]
            elif isinstance(task_config, list):
                categories = task_config
            else:
                print(f"警告: 任务 {task} 的配置格式不正确，跳过")
                continue
            
            for category in categories:
                results['total'] += 1
                print(f"\n--- LLM 评分任务: {task}, 类别: {category} ---")
                
                try:
                    # 从 global_config 读取 LLM 评分配置，默认值：use_gpt=False, use_gemini=True
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
                        print(f"  ✓ 完成")
                    else:
                        results['failed'] += 1
                        results['details'].append({
                            'task': task,
                            'category': category,
                            'type': 'llm',
                            'status': 'failed',
                            'exit_code': exit_code
                        })
                        print(f"  ✗ 失败 (退出码: {exit_code})")
                        
                except Exception as e:
                    results['failed'] += 1
                    results['details'].append({
                        'task': task,
                        'category': category,
                        'type': 'llm',
                        'status': 'failed',
                        'error': str(e)
                    })
                    print(f"  ✗ 异常: {e}")
        
        elif task in THIRDPARTY_TASKS:
            # 第三方评测任务
            # 对于 omnicontext 和 dpg，只要 key 出现就执行
            # 对于 geneval，true/false 表示是否使用 refine prompt，但都执行
            if task_config is False:
                # 如果明确设置为 False，跳过（但 geneval 仍然执行）
                if task == "geneval":
                    # geneval 即使 false 也执行（只是不使用 refine prompt）
                    pass
                else:
                    continue
            
            results['total'] += 1
            print(f"\n--- 第三方评测任务: {task} ---")
            
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
                    print(f"  ✓ 完成")
                else:
                    results['failed'] += 1
                    results['details'].append({
                        'task': task,
                        'type': 'thirdparty',
                        'status': 'failed',
                        'exit_code': exit_code
                    })
                    print(f"  ✗ 失败 (退出码: {exit_code})")
                    
            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'task': task,
                    'type': 'thirdparty',
                    'status': 'failed',
                    'error': str(e)
                })
                print(f"  ✗ 异常: {e}")
        
        else:
            print(f"警告: 未知的任务类型: {task}，跳过")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="评测运行脚本")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                       help="配置文件路径")
    parser.add_argument("--baseline", type=str, default=None,
                       choices=SUPPORTED_BASELINES,
                       help="指定要运行的 baseline 模型")
    parser.add_argument("--exp", type=str, default=None,
                       help="指定要运行的实验名称")
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
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
        print("配置文件中没有评测任务，退出")
        sys.exit(0)
    
    # 确定要运行的评测
    evaluations_to_run = {}
    if args.baseline and args.exp:
        # 指定了 baseline 和 exp
        if args.baseline == "any":
            print("--baseline any 不支持与 --exp 同时指定，请使用 evaluations.any 配置项列出目录")
            sys.exit(1)
        if args.baseline not in evaluations:
            print(f"未找到 baseline: {args.baseline}")
            sys.exit(1)
        if args.exp not in evaluations[args.baseline]:
            print(f"未找到实验: {args.baseline}/{args.exp}")
            sys.exit(1)
        evaluations_to_run[args.baseline] = {args.exp: evaluations[args.baseline][args.exp]}
    else:
        # 运行所有配置的评测
        evaluations_to_run = evaluations
    
    # 运行评测
    all_results = {}
    total_success = 0
    total_failed = 0
    
    for baseline, baseline_config in evaluations_to_run.items():
        if baseline_config is None:
            continue
        if isinstance(baseline_config, list):
            # evaluations.any: list of { path, tasks }，path 为结果根目录
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
    
    # 打印总结
    print(f"\n{'='*60}")
    print("评测任务总结")
    print(f"{'='*60}")
    
    for key, results in all_results.items():
        print(f"\n{key}:")
        print(f"  成功: {results['success']}/{results['total']}")
        print(f"  失败: {results['failed']}/{results['total']}")
        
        if results['failed'] > 0:
            print("  失败详情:")
            for detail in results['details']:
                if detail['status'] == 'failed':
                    error = detail.get('error', detail.get('exit_code', 'unknown'))
                    task_info = f"{detail['task']}"
                    if 'category' in detail:
                        task_info += f"/{detail['category']}"
                    print(f"    - {task_info}: {error}")
    
    print(f"\n总计: 成功 {total_success}, 失败 {total_failed}")
    
    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
