import os
#!/usr/bin/env python3
"""
Customization数据筛选脚本

功能：
1. 从final/customization目录读取数据
2. 检查是否有consistency_scores和following_score，如果没有则调用API补全
3. 基于threshold进行过滤筛选
4. 转换为最简格式，只保留: task, idx, prompt, input_images, output_image
5. 保存到filter/customization目录
"""

import json
import random
import shutil
import sys
import threading
import importlib.util
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加utils路径
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from utils.convert_to_minimal import convert_to_minimal

# 添加参考评分脚本路径
SCORE_MODULE_PATH = None  # Set this to your score module path

# ====== 配置参数 ======
FINAL_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "final" / "customization")
FILTER_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "filter" / "customization")
# You can override FILTER_DIR to use a custom path if needed

# 筛选阈值
SKIP_SCORE = False
CONSISTENCY_SCORE_THRESHOLD = 6.0  # consistency_scores列表中每个分数都需要 >= 此阈值
FOLLOWING_SCORE_THRESHOLD = 6.0    # following_score需要 >= 此阈值

# 并行worker数量
MAX_PARALLEL_WORKERS = 256

# 筛选配置：{image_count_category: {train: count, eval: count}}
FILTER_CONFIG = {
    "1-3": {"train": 20000, "eval": 250},
    "4-5": {"train": 20000, "eval": 250},
    "6-7": {"train": 30000, "eval": 250},
    ">=8": {"train": 30000, "eval": 250},
}

# 随机种子
RANDOM_SEED = 42
# ======================


def get_deterministic_seed(seed_str: str) -> int:
    """
    生成确定性的随机种子（使用hashlib确保跨运行的一致性）
    
    Args:
        seed_str: 种子字符串
    
    Returns:
        确定性的整数种子
    """
    # 使用hashlib生成确定性的哈希值
    hash_obj = hashlib.md5(seed_str.encode('utf-8'))
    hash_int = int(hash_obj.hexdigest(), 16)
    # 取模确保在合理范围内
    return hash_int % (2**31)


def get_combination_key_from_sample(sample: Dict[str, Any]) -> Optional[str]:
    """
    从样本的input_images生成combination_key（用于customization任务）
    
    Args:
        sample: 样本数据
    
    Returns:
        combination_key，如果无法生成则返回None
    """
    input_images = sample.get("input_images", [])
    if not isinstance(input_images, list) or len(input_images) == 0:
        return None
    
    # 排序文件列表并生成MD5哈希
    sorted_files = sorted(input_images)
    key_str = "|".join(sorted_files)
    return hashlib.md5(key_str.encode()).hexdigest()


def get_unique_id_from_sample(sample: Dict[str, Any]) -> Optional[str]:
    """
    从样本获取或生成unique_id（用于customization任务）
    
    Args:
        sample: 样本数据
    
    Returns:
        unique_id，如果无法生成则返回None
    """
    # 优先使用已有的unique_id
    unique_id = sample.get("unique_id")
    if unique_id:
        return unique_id
    
    # 如果没有unique_id，从input_images生成combination_key，然后生成unique_id
    combination_key = get_combination_key_from_sample(sample)
    if combination_key:
        # 对于customization，unique_id格式是 customization_{combination_key}
        # 但可能还会经过MD5处理，这里我们直接使用原始格式
        # 如果需要MD5处理，需要检查配置，但为了简单起见，我们使用原始格式
        return f"customization_{combination_key}"
    
    return None


def load_score_module():
    """
    Load the optional reference scoring module.

    Set ``SCORE_MODULE_PATH`` at the top of this file to a valid path
    before calling this function.  When ``SCORE_MODULE_PATH`` is ``None``
    (the default) the function raises ``RuntimeError`` immediately so that
    the caller can fall back to the Gemini-based scorer.

    Returns:
        Loaded score module object.
    """
    if SCORE_MODULE_PATH is None:
        raise RuntimeError(
            "SCORE_MODULE_PATH is not configured. "
            "Set it to the path of your scoring module, or set SKIP_SCORE=True "
            "to skip reference-based scoring."
        )
    score_module_path = Path(SCORE_MODULE_PATH)
    if not score_module_path.exists():
        raise FileNotFoundError(f"Score module not found: {score_module_path}")

    # Ensure the module's parent directory is importable
    module_dir = str(score_module_path.parent)
    # Clean up any previously cached utils modules to avoid import conflicts
    modules_to_remove = [key for key in sys.modules if key.startswith('utils.')]
    for module_name in modules_to_remove:
        del sys.modules[module_name]
    if 'utils' in sys.modules:
        del sys.modules['utils']
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location("score.customization", score_module_path)
    score_module = importlib.util.module_from_spec(spec)
    # Set __file__ so that relative paths inside the module resolve correctly
    score_module.__file__ = str(score_module_path.resolve())
    spec.loader.exec_module(score_module)

    return score_module


def has_score_fields(sample: Dict[str, Any]) -> bool:
    """
    检查样本是否有评分字段
    
    Args:
        sample: 样本数据
    
    Returns:
        是否有consistency_scores和following_score
    """
    consistency_scores = sample.get("consistency_scores")
    following_score = sample.get("following_score")
    
    # 检查consistency_scores是否为有效列表
    if not isinstance(consistency_scores, list) or len(consistency_scores) == 0:
        return False
    
    # 检查following_score是否为有效数值
    if following_score is None or not isinstance(following_score, (int, float)):
        return False
    
    return True


def evaluate_sample(json_path: Path, score_module, gemini_generator) -> Optional[Dict[str, Any]]:
    """
    对单个样本进行评分（使用Gemini-3-flash）
    
    Args:
        json_path: JSON文件路径
        score_module: 评分模块
        gemini_generator: Gemini生成器实例
    
    Returns:
        评分结果字典，包含consistency_scores, following_score, overall_reasoning
        如果失败返回None
    """
    try:
        # 读取样本数据
        with open(json_path, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        
        sample_id = json_path.stem
        
        # 只使用Gemini进行评分
        if gemini_generator is None:
            print(f"  错误: Gemini生成器未初始化")
            return None
        
        evaluate_with_gemini = getattr(score_module, 'evaluate_with_gemini')
        result = evaluate_with_gemini(sample_data, gemini_generator, sample_id)
        
        return result
    except Exception as e:
        print(f"  错误: 评分失败 {json_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_score_to_json(json_path: Path, score_result: Dict[str, Any]) -> bool:
    """
    将评分结果保存到JSON文件中（不破坏已有字段）
    
    Args:
        json_path: JSON文件路径
        score_result: 评分结果字典
    
    Returns:
        是否保存成功
    """
    try:
        # 读取现有数据
        with open(json_path, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        
        # 更新评分字段
        sample_data["consistency_scores"] = score_result.get("consistency_scores", [])
        sample_data["following_score"] = score_result.get("following_score")
        if "overall_reasoning" in score_result:
            sample_data["overall_reasoning"] = score_result.get("overall_reasoning")
        
        # 保存回文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"  错误: 保存评分失败 {json_path}: {e}")
        return False


def check_and_complete_scores(samples: List[Tuple[int, Path, Dict[str, Any]]], 
                               score_module, gemini_generator) -> List[Tuple[int, Path, Dict[str, Any]]]:
    """
    检查并补全缺失的评分（并行处理）
    
    Args:
        samples: 样本列表，每个元素为(idx, json_path, sample_data)
        score_module: 评分模块
        gemini_generator: Gemini生成器实例
    
    Returns:
        更新后的样本列表（已补全评分并保存）
    """
    # 找出需要评分的样本
    samples_to_evaluate = []
    for idx, json_path, sample_data in samples:
        if not has_score_fields(sample_data):
            samples_to_evaluate.append((idx, json_path, sample_data))
    
    if not samples_to_evaluate:
        print(f"  所有样本已有评分，无需补全")
        return samples
    
    print(f"  需要补全 {len(samples_to_evaluate)} 个样本的评分...")
    
    # 并行处理
    completed = 0
    total = len(samples_to_evaluate)
    lock = threading.Lock()
    save_lock = threading.Lock()
    
    def process_sample(item):
        nonlocal completed
        idx, json_path, sample_data = item
        
        # 进行评分
        score_result = evaluate_sample(json_path, score_module, gemini_generator)
        
        if score_result is not None:
            # 保存评分到JSON文件
            with save_lock:
                if save_score_to_json(json_path, score_result):
                    # 重新读取更新后的数据（确保数据是最新的）
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            sample_data = json.load(f)
                    except Exception as e:
                        print(f"  警告: 重新读取失败 {json_path}: {e}")
        
        with lock:
            completed += 1
            status = "成功" if score_result is not None else "失败"
            print(f"  [{completed}/{total}] {json_path.name}: {status}")
        
        return (idx, json_path, sample_data)
    
    # 使用ThreadPoolExecutor并行处理
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        futures = {executor.submit(process_sample, item): item for item in samples_to_evaluate}
        
        updated_samples_map = {}
        for future in as_completed(futures):
            idx, json_path, sample_data = future.result()
            updated_samples_map[(idx, json_path)] = (idx, json_path, sample_data)
    
    # 更新样本列表：只更新被修改过的样本，其他样本保持内存中的数据
    updated_samples = []
    for idx, json_path, sample_data in samples:
        if (idx, json_path) in updated_samples_map:
            # 如果样本被更新过，使用更新后的数据（已从文件重新读取）
            updated_samples.append(updated_samples_map[(idx, json_path)])
        else:
            # 如果样本没有被更新，直接使用内存中的数据，避免不必要的文件 I/O
            updated_samples.append((idx, json_path, sample_data))
    
    return updated_samples


def filter_samples(samples: List[Tuple[int, Path, Dict[str, Any]]]) -> List[Tuple[int, Path, Dict[str, Any]]]:
    """
    基于阈值筛选样本
    
    Args:
        samples: 样本列表，每个元素为(idx, json_path, sample_data)
    
    Returns:
        筛选后的样本列表，保持 (idx, json_path, sample_data) 格式
    """
    filtered = []
    
    for idx, json_path, sample in samples:
        # 检查评分字段
        if not has_score_fields(sample):
            # 如果没有评分字段，跳过（应该在之前已补全）
            continue
        
        consistency_scores = sample.get("consistency_scores", [])
        following_score = sample.get("following_score")
        
        # 检查consistency_scores：列表中每个分数都需要 >= threshold
        all_consistency_ok = True
        if isinstance(consistency_scores, list):
            for score in consistency_scores:
                if not isinstance(score, (int, float)) or score < CONSISTENCY_SCORE_THRESHOLD:
                    all_consistency_ok = False
                    break
        else:
            all_consistency_ok = False
        
        # 检查following_score
        following_ok = isinstance(following_score, (int, float)) and following_score >= FOLLOWING_SCORE_THRESHOLD
        
        input_num = len(sample.get("input_images", []))
        within_threshold = True
        if input_num > 10:
            print(f"More than 10 input images: {input_num}")
            within_threshold = False
        
        # 两个条件都满足才保留
        if all_consistency_ok and following_ok and within_threshold:
            filtered.append((idx, json_path, sample))
    
    return filtered


def main():
    """主函数"""
    print("=" * 80)
    print("Customization数据筛选脚本")
    print("=" * 80)
    print(f"Final目录: {FINAL_DIR}")
    print(f"Filter目录: {FILTER_DIR}")
    print(f"Consistency Score阈值: {CONSISTENCY_SCORE_THRESHOLD}")
    print(f"Following Score阈值: {FOLLOWING_SCORE_THRESHOLD}")
    print(f"筛选配置: {FILTER_CONFIG}")
    print("=" * 80)
    
    if not FINAL_DIR.exists():
        print(f"错误: Final目录不存在: {FINAL_DIR}")
        return
    
    # 加载评分模块和初始化Gemini生成器
    print("\n正在加载评分模块...")
    try:
        score_module = load_score_module()
        
        # 初始化Gemini生成器（使用gemini-3-flash-preview进行评分）
        # 注意：gemini_generator 会被多个线程并发使用，需要确保 GeminiAPIGenerator 是线程安全的
        # 如果遇到并发问题，可以考虑为每个线程创建独立的实例
        GEMINI_CONFIG = {
            "api_key": os.environ.get("GEMINI_API_KEY", ""),
            "model_name": "gemini-3-flash-preview",
            "max_try": 10,
            "timeout": 60
        }
        from api_generator.text_generator.gemini_api import GeminiAPIGenerator
        gemini_generator = GeminiAPIGenerator(
            app_key=GEMINI_CONFIG["api_key"],
            model_name=GEMINI_CONFIG["model_name"],
            max_try=GEMINI_CONFIG["max_try"],
            print_log=False,
            timeout=GEMINI_CONFIG["timeout"]
        )
        print("评分模块加载完成")
    except Exception as e:
        print(f"错误: 加载评分模块失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 确保 FILTER_DIR 存在
    FILTER_DIR.mkdir(parents=True, exist_ok=True)
    
    # 处理train和eval数据
    # 判断config中是否配置eval数据，如果配置则处理eval数据，否则只处理train数据
    for split_type in ["train", "eval"]:
        print(f"\n处理 {split_type} 数据...")
        
        # 对于train，清除整个train目录后重新构建
        if split_type == "train":
            train_dir = FILTER_DIR / "train"
            if train_dir.exists():
                print(f"清除 Train 目录: {train_dir}")
                shutil.rmtree(train_dir)
            train_dir.mkdir(parents=True, exist_ok=True)
        
        # 遍历所有image_count_category目录
        for category_dir in (FINAL_DIR / split_type).glob("*"):
            if not category_dir.is_dir():
                continue
            
            image_count_category = category_dir.name
            
            # 检查是否有配置，如果没有配置则跳过数量控制
            if image_count_category not in FILTER_CONFIG:
                print(f"\n处理类别: {image_count_category} (无数量限制)")
                target_count = None
            else:
                target_count = FILTER_CONFIG[image_count_category].get(split_type)
                if target_count is None:
                    print(f"\n处理类别: {image_count_category} (无数量限制)")
                    target_count = None
                else:
                    print(f"\n处理类别: {image_count_category} (目标数量: {target_count})")
            
            if split_type == "eval" and target_count is None:
                continue
            
            # 对于eval，检查现有数据数量
            output_dir = FILTER_DIR / split_type / image_count_category
            existing_count = 0
            if split_type == "eval" and output_dir.exists():
                existing_files = list(output_dir.glob("*.json"))
                existing_count = len(existing_files)
                if target_count is not None and existing_count >= target_count:
                    print(f"  Eval数据已满足目标数量 ({existing_count} >= {target_count})，跳过")
                    continue
                elif existing_count > 0:
                    print(f"  现有Eval数据: {existing_count} 个，目标: {target_count}，需要补足 {target_count - existing_count} 个")
            
            # 读取JSON文件
            json_dir = category_dir / "json"
            if not json_dir.exists():
                print(f"  跳过: JSON目录不存在")
                continue
            
            samples = []
            for json_file in sorted(json_dir.glob("*.json")):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        sample = json.load(f)
                        # 提取idx
                        idx = int(json_file.stem)
                        samples.append((idx, json_file, sample))
                except Exception as e:
                    print(f"  警告: 读取JSON文件失败 {json_file}: {e}")
                    continue
            
            print(f"  加载了 {len(samples)} 个样本")
            
            # 检查和补全评分
            if not SKIP_SCORE:
                print(f"  正在检查并补全评分...")
                samples = check_and_complete_scores(samples, score_module, gemini_generator)
            else:
                print(f"  跳过评分检查和补全")
            
            # 筛选样本（基于阈值），直接返回 (idx, json_path, sample) 元组列表
            # 注意：不对eval数据使用filter，直接使用原始samples
            if split_type == "eval":
                filtered_with_idx = samples
                print(f"  Eval数据跳过筛选，使用原始样本: {len(filtered_with_idx)} 个样本")
            else:
                filtered_with_idx = filter_samples(samples)
                print(f"  筛选后: {len(filtered_with_idx)} 个样本")
            
            # 对于eval，需要排除已存在的样本（基于unique_id）
            if split_type == "eval" and existing_count > 0:
                # 读取现有文件的unique_id
                existing_identifiers = set()
                for existing_file in output_dir.glob("*.json"):
                    try:
                        with open(existing_file, 'r', encoding='utf-8') as f:
                            existing_sample = json.load(f)
                            # 对于customization，使用unique_id作为唯一标识
                            # 如果没有unique_id，从input_images生成combination_key并生成unique_id
                            unique_id = get_unique_id_from_sample(existing_sample)
                            if unique_id:
                                existing_identifiers.add(unique_id)
                    except Exception as e:
                        print(f"  警告: 读取现有文件失败 {existing_file}: {e}")
                        continue
                
                # 过滤掉已存在的样本
                original_count = len(filtered_with_idx)
                filtered_with_idx = [
                    (idx, json_path, sample) for idx, json_path, sample in filtered_with_idx
                    if get_unique_id_from_sample(sample) not in existing_identifiers
                ]
                print(f"  排除已存在样本后: {len(filtered_with_idx)} 个样本（移除了 {original_count - len(filtered_with_idx)} 个）")
            
            # 如果配置了目标数量且筛选后样本数量多于目标数量，打乱并取前n个
            if target_count is not None:
                if split_type == "eval":
                    # eval需要补足的数量
                    needed_count = target_count - existing_count
                    if needed_count > 0 and len(filtered_with_idx) > needed_count:
                        # 设置随机种子以确保可重现性（基于split_type和image_count_category）
                        seed_str = f"{RANDOM_SEED}_{split_type}_{image_count_category}"
                        seed_hash = get_deterministic_seed(seed_str)
                        random.seed(seed_hash)
                        random.shuffle(filtered_with_idx)
                        # 取前needed_count个
                        filtered_with_idx = filtered_with_idx[:needed_count]
                        print(f"  打乱后取前 {needed_count} 个样本用于补足")
                else:
                    # train保持原有逻辑
                    if len(filtered_with_idx) > target_count:
                        # 设置随机种子以确保可重现性（基于split_type和image_count_category）
                        seed_str = f"{RANDOM_SEED}_{split_type}_{image_count_category}"
                        seed_hash = get_deterministic_seed(seed_str)
                        random.seed(seed_hash)
                        random.shuffle(filtered_with_idx)
                        # 取前target_count个
                        filtered_with_idx = filtered_with_idx[:target_count]
                        print(f"  打乱后取前 {target_count} 个样本")
            
            # 转换为最简格式并保存
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if split_type == "eval" and existing_count > 0:
                # eval：找到当前最大编号，从下一个编号开始
                existing_indices = []
                for existing_file in output_dir.glob("*.json"):
                    try:
                        idx = int(existing_file.stem)
                        existing_indices.append(idx)
                    except ValueError:
                        continue
                start_idx = max(existing_indices, default=0) + 1
            else:
                # train：重新编号，从1开始
                # 注意：train 目录已在第 410-415 行被完全删除并重建，此处无需再次清空文件
                start_idx = 1
            
            # 保存样本
            for i, (original_idx, json_path, sample) in enumerate(filtered_with_idx, start=start_idx):
                minimal = convert_to_minimal(sample, "customization", i)
                output_file = output_dir / f"{i:08d}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(minimal, f, ensure_ascii=False, indent=2)
            
            final_count = existing_count + len(filtered_with_idx) if split_type == "eval" and existing_count > 0 else len(filtered_with_idx)
            print(f"  已保存到: {output_dir}（共 {final_count} 个样本）")
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()

