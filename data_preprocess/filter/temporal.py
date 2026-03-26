#!/usr/bin/env python3
"""
Temporal数据筛选脚本

功能：
1. 从final/temporal目录读取数据
2. 筛选分数高于阈值的样本
3. 转换为最简格式，只保留: task, idx, prompt, input_images, output_image
4. 保存到filter/temporal目录
"""

import json
import random
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List

# 添加utils路径
import sys
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from utils.convert_to_minimal import convert_to_minimal

# ====== 配置参数 ======
FINAL_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "final" / "temporal")
FILTER_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "filter" / "temporal")
# You can override FILTER_DIR to use a custom path if needed

# 筛选阈值（需要根据实际需求调整）
TEMPORAL_SCORE_THRESHOLD = 6

# 筛选配置：{image_count_category: {train: count, eval: count}}
FILTER_CONFIG = {
    "1-3": {"train": 25000, "eval": 250},
    "4-5": {"train": 25000, "eval": 250},
    "6-7": {"train": 25000, "eval": 250},
    ">=8": {"train": 25000, "eval": 250},
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


def filter_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    筛选样本
    
    Args:
        samples: 样本列表
    
    Returns:
        筛选后的样本列表
    """
    filtered = []
    
    for sample in samples:
        # 检查分数（需要根据实际字段名调整）
        temporal_score = sample.get("temporal_score", sample.get("score", 0))
        
        # 筛选分数高于阈值的样本
        if temporal_score >= TEMPORAL_SCORE_THRESHOLD:
            input_num = len(sample.get("input_images", []))
            if input_num > 10:
                # print(f"More than 10 input images: {input_num}")
                continue
            filtered.append(sample)
    
    return filtered


def main():
    """主函数"""
    print("=" * 80)
    print("Temporal数据筛选脚本")
    print("=" * 80)
    print(f"Final目录: {FINAL_DIR}")
    print(f"Filter目录: {FILTER_DIR}")
    print(f"Temporal Score阈值: {TEMPORAL_SCORE_THRESHOLD}")
    print(f"筛选配置: {FILTER_CONFIG}")
    print("=" * 80)
    
    if not FINAL_DIR.exists():
        print(f"错误: Final目录不存在: {FINAL_DIR}")
        return
    
    # 确保 FILTER_DIR 存在
    FILTER_DIR.mkdir(parents=True, exist_ok=True)
    
    # 对于train，清除整个train目录后重新构建
    if (FILTER_DIR / "train").exists():
        print(f"清除 Train 目录: {FILTER_DIR / 'train'}")
        shutil.rmtree(FILTER_DIR / "train")
    (FILTER_DIR / "train").mkdir(parents=True, exist_ok=True)
    
    # 处理train和eval数据
    for split_type in ["train", "eval"]:
        print(f"\n处理 {split_type} 数据...")
        
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
            existing_identifiers = set()
            if split_type == "eval" and output_dir.exists():
                existing_files = list(output_dir.glob("*.json"))
                existing_count = len(existing_files)
                if target_count is not None and existing_count >= target_count:
                    print(f"  Eval数据已满足目标数量 ({existing_count} >= {target_count})，跳过")
                    continue
                elif existing_count > 0:
                    print(f"  现有Eval数据: {existing_count} 个，目标: {target_count}，需要补足 {target_count - existing_count} 个")
                    # 读取现有文件的unique_id或其他唯一标识
                    for existing_file in output_dir.glob("*.json"):
                        try:
                            with open(existing_file, 'r', encoding='utf-8') as f:
                                existing_sample = json.load(f)
                                # 尝试使用unique_id，如果没有则使用其他唯一标识
                                unique_id = existing_sample.get("unique_id")
                                if unique_id:
                                    existing_identifiers.add(unique_id)
                                else:
                                    # 使用source_file+source_line+true_index作为备选
                                    source_file = existing_sample.get("source_file", "")
                                    source_line = existing_sample.get("source_line", -1)
                                    true_index = existing_sample.get("true_index", -1)
                                    existing_identifiers.add((source_file, source_line, true_index))
                        except Exception as e:
                            print(f"  警告: 读取现有文件失败 {existing_file}: {e}")
                            continue
            
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
                        samples.append((idx, sample))
                except Exception as e:
                    print(f"  警告: 读取JSON文件失败 {json_file}: {e}")
                    continue
            
            print(f"  加载了 {len(samples)} 个样本")
            
            # 筛选样本
            filtered_samples = filter_samples([s[1] for s in samples])
            print(f"  筛选后: {len(filtered_samples)} 个样本")
            
            # 创建筛选后的样本及其索引的列表
            filtered_with_idx = [(idx, sample) for idx, sample in samples if sample in filtered_samples]
            
            # 对于eval，需要排除已存在的样本
            if split_type == "eval" and existing_count > 0:
                original_count = len(filtered_with_idx)
                filtered_with_idx = [
                    (idx, sample) for idx, sample in filtered_with_idx
                    if not (
                        sample.get("unique_id") in existing_identifiers or
                        (sample.get("source_file", ""), sample.get("source_line", -1), sample.get("true_index", -1)) in existing_identifiers
                    )
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
                start_idx = 1
                # 清空已存在的文件（仅对train）
                for existing_file in output_dir.glob("*.json"):
                    existing_file.unlink()
            
            # 保存样本
            for i, (original_idx, sample) in enumerate(filtered_with_idx, start=start_idx):
                minimal = convert_to_minimal(sample, "temporal", i)
                output_file = output_dir / f"{i:08d}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(minimal, f, ensure_ascii=False, indent=2)
            
            final_count = existing_count + len(filtered_with_idx) if split_type == "eval" and existing_count > 0 else len(filtered_with_idx)
            print(f"  已保存到: {output_dir}（共 {final_count} 个样本）")
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()

