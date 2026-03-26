#!/usr/bin/env python3
"""
评估脚本

从inference的输出目录加载数据，使用LLM（GPT4o和Gemini-3-flash）进行评分
支持多任务并行评估和断点续传
支持多个baseline模型的评估

outputs目录结构：
  outputs/{baseline}/{exp_name}/{task}/{image_num_category}/
  例如：outputs/bagel/exp_001/customization/1-3/
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# ============================================================================
# 配置常量
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
MACRO_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = MACRO_DIR / "outputs"

# Add Macro root to path for importing utils
if str(MACRO_DIR) not in sys.path:
    sys.path.insert(0, str(MACRO_DIR))

# 支持的task列表和image num categories（使用LLM评分的任务）
SUPPORTED_TASKS = ["customization", "illustration", "spatial", "temporal"]
IMAGE_NUM_CATEGORIES = ["1-3", "4-5", "6-7", ">=8"]

# 支持的baseline模型
SUPPORTED_BASELINES = ["bagel", "omnigen", "qwen"]

# LLM配置
GPT_CONFIG = {
    "url": os.environ.get("OPENAI_URL", "https://api.openai.com/v1/chat/completions"),
    "key": os.environ.get("OPENAI_KEY", "")
}

GEMINI_CONFIG = {
    "api_key": os.environ.get("GEMINI_API_KEY", ""),
    "model_name": os.environ.get("GEMINI_MODEL_NAME", "gemini-3.0-flash-preview"),
    "max_try": 100
}

# 重试配置
MAX_RETRIES = 10
RETRY_DELAY = 2
TIMEOUT = 60  # 60秒超时
PARALLEL_WORKERS = 128  # 并行处理样本数


# ============================================================================
# 数据加载函数
# ============================================================================
def load_samples_from_output(
    baseline: str,
    exp_name: str,
    task: str,
    image_num_category: str,
    output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    从output目录加载样本数据
    
    目录结构：{output_root}/{baseline}/{exp_name}/{task}/{image_num_category}/
    当 output_root 为 None 时使用默认 OUTPUT_DIR。
    
    Args:
        baseline: 模型类型（bagel, omnigen, qwen, api）
        exp_name: 实验名称（或 api 名称如 gpt/seed/nano）
        task: 任务类型
        image_num_category: 图像数量类别
        output_root: 输出根目录，None 时用默认 OUTPUT_DIR
        
    Returns:
        样本字典，key为idx（字符串格式），value为样本数据（包含json_file_path字段）
    """
    root = Path(output_root) if output_root is not None else OUTPUT_DIR
    if baseline == "any":
        # 任意目录：output_root 为“结果根”的父目录，exp_name 为结果根目录名
        input_dir = root / exp_name / task / image_num_category
    else:
        input_dir = root / baseline / exp_name / task / image_num_category
    
    if not input_dir.exists():
        return {}
    
    samples = {}
    # 只加载样本 JSON（{idx:08d}.json），跳过 metadata.json 等非样本文件，避免覆盖
    for json_file in sorted(input_dir.glob("*.json")):
        if not json_file.stem.isdigit():
            continue
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                sample = json.load(f)
            
            # 使用 idx 作为 key；部分模型（如 omnigen）只写 base_idx 不写 idx，需兼容
            idx = sample.get("idx")
            if idx is None:
                idx = sample.get("base_idx")
            if idx is None and json_file.stem.isdigit():
                idx = int(json_file.stem)
            if idx is None:
                idx = 0
            # 优先用文件名作为 sample_id，避免多个 JSON 写相同 idx 或 metadata.json 等覆盖
            if json_file.stem.isdigit() and len(json_file.stem) <= 8:
                sample_id = f"{int(json_file.stem):08d}"
            else:
                sample_id = f"{idx:08d}"
            
            # 标准化字段名（prompt -> instruction）
            if 'prompt' in sample and 'instruction' not in sample:
                sample['instruction'] = sample['prompt']
            
            row = {
                'sample_id': sample_id,
                'idx': idx,
                'instruction': sample.get('instruction', ''),
                'input_images': sample.get('input_images', []),
                'output_image': sample.get('output_image', ''),
                'target_image': sample.get('target_image', ''),
                'category': sample.get('category', image_num_category),
                'json_file_path': str(json_file),  # 保存JSON文件路径，用于保存评分文件
                'json_dir': str(json_file.parent)  # 保存JSON文件所在目录
            }
            # 若 JSON 中为相对路径，将 output_image 解析为相对 json_dir 的绝对路径
            if row['output_image'] and not os.path.isabs(row['output_image']):
                row['output_image'] = str(row['output_image'])
            if row['input_images']:
                resolved = []
                for p in row['input_images']:
                    if p and not os.path.isabs(p):
                        p = str(p)
                    resolved.append(p)
                row['input_images'] = resolved
            samples[sample_id] = row
        except Exception as e:
            print(f"警告: 加载JSON文件失败 {json_file}: {e}")
            continue
    
    return samples


# ============================================================================
# 主函数
# ============================================================================
def load_score_file(score_file: Path) -> Dict[str, Any]:
    """加载单个样本的评分文件，用于断点续传"""
    if not score_file.exists():
        return {}
    
    try:
        with open(score_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 加载评分文件失败 {score_file}: {e}")
        return {}


def save_score_file(score_file: Path, scores: Dict[str, Any]):
    """保存单个样本的评分文件"""
    try:
        with open(score_file, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"警告: 保存评分文件失败 {score_file}: {e}")


def run_evaluation(
    baseline: str,
    exp_name: str,
    task: str,
    image_num_category: str = "all",
    output_dir: str = None,
    output_root: str = None,
    use_gpt: bool = True,
    use_gemini: bool = True,
    max_samples: Optional[int] = None,
):
    """
    运行评估（支持GPT和Gemini评分，支持断点续传）
    
    Args:
        baseline: 模型类型（bagel, omnigen, qwen, api）
        exp_name: 实验名称（模型检查点名称，或 api 时为 gpt/seed/nano）
        task: 任务类型 (customization, illustration, spatial, temporal, all)
        image_num_category: 图像数量类别 (1-3, 4-5, 6-7, >=8, all)
        output_dir: 输出目录，默认为评分文件保存在样本所在目录
        output_root: 生成结果根目录，None 时用默认 OUTPUT_DIR（用于 api 等非默认路径）
        use_gpt: 是否使用GPT评分
        use_gemini: 是否使用Gemini评分
        max_samples: 每个 (task, category) 最多评测的样本数，None 表示全部
    """
    # 验证baseline
    if baseline not in SUPPORTED_BASELINES:
        raise ValueError(f"Unsupported baseline: {baseline}. Supported baselines: {SUPPORTED_BASELINES}")
    
    # 确定要处理的tasks
    if task == "all":
        tasks_to_process = SUPPORTED_TASKS
    else:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}. Supported tasks: {SUPPORTED_TASKS}")
        tasks_to_process = [task]
    
    # 确定要处理的categories
    if image_num_category == "all":
        categories_to_process = IMAGE_NUM_CATEGORIES
    else:
        if image_num_category not in IMAGE_NUM_CATEGORIES:
            raise ValueError(f"Unsupported image_num_category: {image_num_category}. Supported: {IMAGE_NUM_CATEGORIES}")
        categories_to_process = [image_num_category]
    
    # 处理每个task和category组合
    for current_task in tasks_to_process:
        for current_category in categories_to_process:
            print("=" * 80)
            print(f"处理任务: {current_task}, 图像数量类别: {current_category}")
            print(f"Baseline: {baseline}, 实验: {exp_name}")
            print("=" * 80)
            
            # 1. 加载样本数据
            samples = load_samples_from_output(
                baseline, exp_name, current_task, current_category,
                output_root=Path(output_root) if output_root else None,
            )
            if not samples:
                print(f"警告: 没有找到样本数据，跳过 {current_task}/{current_category}")
                continue
            
            # 可选：只评测前 max_samples 个样本（按 sample_id 排序）
            if max_samples is not None and max_samples > 0:
                sorted_ids = sorted(samples.keys())[:max_samples]
                samples = {sid: samples[sid] for sid in sorted_ids}
                print(f"限制评测数量: {max_samples}，当前 {len(samples)} 个样本")
            else:
                print(f"找到 {len(samples)} 个样本")
            
            # 确定JSON文件所在目录（从第一个样本获取，所有样本应该在同一目录）
            first_sample = next(iter(samples.values()))
            json_dir = Path(first_sample.get('json_dir', ''))
            if not json_dir.exists():
                print(f"警告: JSON文件目录不存在: {json_dir}，跳过 {current_task}/{current_category}")
                continue
            
            print(f"JSON文件目录: {json_dir}")
            
            # 动态导入对应的评分模块（使用文件路径导入）
            score_module_path = SCRIPT_DIR / "score" / f"{current_task}.py"
            if not score_module_path.exists():
                print(f"错误: 评分模块不存在: {score_module_path}")
                continue
            
            # 使用importlib加载模块
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"score.{current_task}", score_module_path)
            score_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(score_module)
            
            evaluate_with_gpt_func = getattr(score_module, 'evaluate_with_gpt')
            evaluate_with_gemini_func = getattr(score_module, 'evaluate_with_gemini')
            is_score_valid_func = getattr(score_module, 'is_score_valid')
            
            # 3. 初始化Gemini生成器（如果需要）
            gemini_generator = None
            if use_gemini:
                try:
                    from api_generator.text_generator.gemini_api import GeminiAPIGenerator
                    gemini_generator = GeminiAPIGenerator(
                        app_key=GEMINI_CONFIG["api_key"],
                        model_name=GEMINI_CONFIG["model_name"],
                        max_try=MAX_RETRIES,
                        print_log=False,
                        timeout=TIMEOUT
                    )
                except ImportError as e:
                    print(f"警告: 无法导入GeminiAPIGenerator: {e}，跳过Gemini评分")
                    use_gemini = False
            
            # 4. 加载已有评分文件（断点续传）
            existing_scores = {}
            for sample_id in samples.keys():
                score_file = json_dir / f"{sample_id}.score"
                if score_file.exists():
                    existing_scores[sample_id] = load_score_file(score_file)
            
            print(f"已加载 {len(existing_scores)} 个已有评分文件")
            
            # 获取需要处理的样本列表（过滤已完成的）
            sample_items = list(samples.items())
            samples_to_process = []
            for sample_id, sample_data in sample_items:
                existing_score = existing_scores.get(sample_id, {})
                gpt_done = is_score_valid_func(existing_score.get('gpt_scores')) if use_gpt else True
                gemini_done = is_score_valid_func(existing_score.get('gemini_scores')) if use_gemini else True
                if gpt_done and gemini_done:
                    continue
                samples_to_process.append((sample_id, sample_data))
            
            print(f"需要处理 {len(samples_to_process)} 个样本")
            
            # 5. GPT评分（如果启用）
            if use_gpt:
                print("\n" + "-" * 80)
                print("第一步：进行GPT4o评分（并行处理）")
                print("-" * 80)
                
                gpt_completed = 0
                gpt_total = len(samples_to_process)
                gpt_lock = threading.Lock()
                
                def process_gpt_sample(item):
                    nonlocal gpt_completed
                    sample_id, sample_data = item
                    # 检查是否已有GPT评分
                    existing_score = existing_scores.get(sample_id, {})
                    if is_score_valid_func(existing_score.get('gpt_scores')):
                        with gpt_lock:
                            gpt_completed += 1
                        return sample_id, existing_score.get('gpt_scores')
                    
                    gpt_score = None
                    if GPT_CONFIG["key"]:
                        gpt_score = evaluate_with_gpt_func(sample_data, sample_id)
                    
                    with gpt_lock:
                        gpt_completed += 1
                        print(f"[GPT] [{gpt_completed}/{gpt_total}] {sample_id}: 完成")
                    
                    return sample_id, gpt_score
                
                # 并行处理GPT评分
                save_lock = threading.Lock()
                with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                    gpt_futures = {executor.submit(process_gpt_sample, item): item for item in samples_to_process}
                    
                    for future in as_completed(gpt_futures):
                        sample_id, gpt_score = future.result()
                        
                        # 加载或创建评分文件（保存在JSON文件同一目录）
                        score_file = json_dir / f"{sample_id}.score"
                        with save_lock:
                            current_scores = load_score_file(score_file)
                            if not current_scores:
                                current_scores = {}
                            
                            # 更新GPT评分（只有当评分有效时才保存，区分0分和None）
                            if is_score_valid_func(gpt_score):
                                # 评分有效（包括0分），保存评分
                                current_scores['gpt_scores'] = gpt_score
                                # 如果使用Gemini，保留已有的gemini_scores
                                if use_gemini and 'gemini_scores' not in current_scores:
                                    existing_score = existing_scores.get(sample_id, {})
                                    if is_score_valid_func(existing_score.get('gemini_scores')):
                                        current_scores['gemini_scores'] = existing_score.get('gemini_scores')
                                # 立即保存到.score文件
                                save_score_file(score_file, current_scores)
                            elif gpt_score is None and not is_score_valid_func(current_scores.get('gpt_scores')):
                                # 评分失败且当前也没有有效评分，不保存None，保持原状
                                pass
                
                print(f"\nGPT评分结果已保存到对应的.score文件")
            
            # 6. Gemini评分（如果启用）
            if use_gemini and gemini_generator:
                print("\n" + "-" * 80)
                print("第二步：进行Gemini评分（并行处理）")
                print("-" * 80)
                
                gemini_completed = 0
                gemini_total = len(samples_to_process)
                gemini_lock = threading.Lock()
                
                def process_gemini_sample(item):
                    nonlocal gemini_completed
                    sample_id, sample_data = item
                    # 检查是否已有Gemini评分
                    existing_score = existing_scores.get(sample_id, {})
                    if is_score_valid_func(existing_score.get('gemini_scores')):
                        with gemini_lock:
                            gemini_completed += 1
                        return sample_id, existing_score.get('gemini_scores')
                    
                    gemini_score = evaluate_with_gemini_func(sample_data, gemini_generator, sample_id)
                    
                    with gemini_lock:
                        gemini_completed += 1
                        print(f"[Gemini] [{gemini_completed}/{gemini_total}] {sample_id}: 完成")
                    
                    return sample_id, gemini_score
                
                # 并行处理Gemini评分
                save_lock = threading.Lock()
                with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                    gemini_futures = {executor.submit(process_gemini_sample, item): item for item in samples_to_process}
                    
                    for future in as_completed(gemini_futures):
                        sample_id, gemini_score = future.result()
                        
                        # 加载或创建评分文件（保存在JSON文件同一目录）
                        score_file = json_dir / f"{sample_id}.score"
                        with save_lock:
                            current_scores = load_score_file(score_file)
                            if not current_scores:
                                current_scores = {}
                            
                            # 更新Gemini评分（只有当评分有效时才保存，区分0分和None）
                            if is_score_valid_func(gemini_score):
                                # 评分有效（包括0分），保存评分
                                current_scores['gemini_scores'] = gemini_score
                                # 如果使用GPT，保留已有的gpt_scores
                                if use_gpt and 'gpt_scores' not in current_scores:
                                    existing_score = existing_scores.get(sample_id, {})
                                    if is_score_valid_func(existing_score.get('gpt_scores')):
                                        current_scores['gpt_scores'] = existing_score.get('gpt_scores')
                                # 立即保存到.score文件
                                save_score_file(score_file, current_scores)
                            elif gemini_score is None and not is_score_valid_func(current_scores.get('gemini_scores')):
                                # 评分失败且当前也没有有效评分，不保存None，保持原状
                                pass
                
                print(f"\nGemini评分结果已保存到对应的.score文件")
            
            # 7. 统计结果
            total_samples = len(samples)
            gpt_success = 0
            gemini_success = 0
            for sample_id in samples.keys():
                score_file = json_dir / f"{sample_id}.score"
                if score_file.exists():
                    scores = load_score_file(score_file)
                    if use_gpt and is_score_valid_func(scores.get('gpt_scores')):
                        gpt_success += 1
                    if use_gemini and is_score_valid_func(scores.get('gemini_scores')):
                        gemini_success += 1
            print(f"\n统计: 总样本数={total_samples}, GPT成功={gpt_success}, Gemini成功={gemini_success}")
    
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="多Baseline模型评估脚本（支持GPT和Gemini评分，支持断点续传）")
    
    parser.add_argument("--baseline", type=str, required=True,
                       choices=SUPPORTED_BASELINES,
                       help="模型类型（bagel, omnigen, qwen）")
    parser.add_argument("--exp_name", type=str, required=True,
                       help="实验名称（模型检查点名称），对应outputs/{baseline}/下的子目录名")
    parser.add_argument("--task", type=str, default="all",
                       choices=SUPPORTED_TASKS + ["all"],
                       help="任务类型")
    parser.add_argument("--image_num_category", type=str, default="all",
                       choices=IMAGE_NUM_CATEGORIES + ["all"],
                       help="图像数量类别")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录，默认为评分文件保存在样本所在目录")
    parser.add_argument("--output_root", type=str, default=None,
                       help="生成结果根目录，默认为 Macro/outputs；api 评测时需与 run 脚本的 output_root 一致")
    parser.add_argument("--use_gpt", action="store_true", default=True,
                       help="是否使用GPT评分（默认True）")
    parser.add_argument("--no_gpt", dest="use_gpt", action="store_false",
                       help="不使用GPT评分")
    parser.add_argument("--use_gemini", action="store_true", default=True,
                       help="是否使用Gemini评分（默认True）")
    parser.add_argument("--no_gemini", dest="use_gemini", action="store_false",
                       help="不使用Gemini评分")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="每个 (task, image_num_category) 最多评测的样本数，不设则评测全部（用于快速试跑）")
    
    args = parser.parse_args()
    
    # 运行评估
    run_evaluation(
        baseline=args.baseline,
        exp_name=args.exp_name,
        task=args.task,
        image_num_category=args.image_num_category,
        output_dir=args.output_dir,
        output_root=args.output_root,
        use_gpt=args.use_gpt,
        use_gemini=args.use_gemini,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
