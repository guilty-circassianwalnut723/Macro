#!/usr/bin/env python3
"""
Spatial数据筛选脚本

功能：
1. 统一控制outdoor、indoor、object三个子类型的筛选配置和数量
2. 从final/spatial目录读取数据（按子类型组织）
3. 先加载所有三类数据，合并后一起打乱采样，取前n项
4. 转换为最简格式，只保留: task, idx, prompt, input_images, output_image
5. 重新排序编号保存到filter/spatial目录（不包含子类型路径）
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
FINAL_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "final" / "spatial")
FILTER_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "filter" / "spatial")
# You can override FILTER_DIR to use a custom path if needed

# 子类型选择：outdoor、indoor或object
# 设置为 None 表示处理所有子类型，设置为具体子类型字符串则只处理该子类型
SUB_TYPE = None  # 修改此值来选择要处理的子类型，None表示处理所有子类型

# 筛选配置：{sub_type: {image_count_category: {train: count, eval: count}}}
FILTER_CONFIG = {
    "object": {
        "1-3": {"train": 10000, "eval": 90},
        "4-5": {"train": 10000, "eval": 90},
        "6-7": {"train": 10000, "eval": 90},
        ">=8": {"train": 10000, "eval": 90},
    },
    "outdoor": {
        "1-3": {"train": 7500, "eval": 80},
        "4-5": {"train": 7500, "eval": 80},
        "6-7": {"train": 7500, "eval": 80},
        ">=8": {"train": 7500, "eval": 80},
    },
    "indoor": {
        "1-3": {"train": 7500, "eval": 80},
        "4-5": {"train": 7500, "eval": 80},
        "6-7": {"train": 7500, "eval": 80},
        ">=8": {"train": 7500, "eval": 80},
    },
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
    筛选样本（对于spatial，目前全部保留）
    
    Args:
        samples: 样本列表
    
    Returns:
        筛选后的样本列表
    """
    # 对于spatial，目前全部保留

    filtered = []
    for sample in samples:
        input_num = len(sample.get("input_images", []))
        if input_num > 10:
            print(f"More than 10 input images: {input_num}")
            continue
        filtered.append(sample)
    return filtered


def main():
    """主函数"""
    # 确定要处理的子类型列表
    if SUB_TYPE is None:
        # 处理所有子类型
        sub_types_to_process = list(FILTER_CONFIG.keys())
        print("=" * 80)
        print("Spatial 数据筛选脚本 - 处理所有子类型（合并后统一采样）")
        print("=" * 80)
    else:
        # 只处理指定的子类型
        if SUB_TYPE not in FILTER_CONFIG:
            raise ValueError(f"不支持的子类型: {SUB_TYPE}，支持的类型: {list(FILTER_CONFIG.keys())}")
        sub_types_to_process = [SUB_TYPE]
        print("=" * 80)
        print(f"Spatial {SUB_TYPE.upper()}数据筛选脚本")
        print("=" * 80)
    
    print(f"Final目录: {FINAL_DIR}")
    print(f"Filter目录: {FILTER_DIR}")
    print(f"将处理的子类型: {sub_types_to_process}")
    print(f"筛选配置: {FILTER_CONFIG}")
    print("=" * 80)
    
    if not FINAL_DIR.exists():
        print(f"错误: Final目录不存在: {FINAL_DIR}")
        return
    
    # 确保 FILTER_DIR 存在
    FILTER_DIR.mkdir(parents=True, exist_ok=True)
    
    # 对于train，清除整个train目录后重新构建
    train_dir = FILTER_DIR / "train"
    if train_dir.exists():
        print(f"清除 Train 目录: {train_dir}")
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有image_count_category（从所有子类型配置中收集）
    all_categories = set()
    for sub_type_config in FILTER_CONFIG.values():
        all_categories.update(sub_type_config.keys())
    all_categories = sorted(list(all_categories))
    
    # 处理train和eval数据
    for split_type in ["train", "eval"]:
        print(f"\n{'=' * 80}")
        print(f"处理 {split_type} 数据")
        print(f"{'=' * 80}")
        
        # 对于每个image_count_category，合并所有子类型的数据后统一处理
        for image_count_category in all_categories:
            print(f"\n处理类别: {image_count_category}")
            
            # 收集所有子类型的数据
            all_samples = []
            category_total_count = 0
            sub_type_counts = {}  # 记录每个子类型的配置数量
            
            for current_sub_type in sub_types_to_process:
                # 计算该子类型和类别的目标数量
                filter_config = FILTER_CONFIG[current_sub_type]
                if image_count_category not in filter_config:
                    target_count = None
                else:
                    target_count = filter_config[image_count_category].get(split_type)

                if split_type == "eval" and target_count is None:
                    continue
                
                if target_count is not None:
                    category_total_count += target_count
                    sub_type_counts[current_sub_type] = target_count
                
                # 读取该子类型和类别的样本
                sub_type_dir = FINAL_DIR / split_type / current_sub_type
                json_dir = sub_type_dir / image_count_category / "json"
                
                if not json_dir.exists():
                    print(f"  跳过 {current_sub_type}: JSON目录不存在")
                    continue
                
                samples = []
                for json_file in sorted(json_dir.glob("*.json")):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            sample = json.load(f)
                            # 保存子类型信息和原始idx
                            sample['_sub_type'] = current_sub_type
                            sample['_original_idx'] = int(json_file.stem)
                            samples.append(sample)
                    except Exception as e:
                        print(f"  警告: 读取JSON文件失败 {json_file}: {e}")
                        continue
                
                count_info = f"加载了 {len(samples)} 个样本"
                if current_sub_type in sub_type_counts:
                    count_info += f" (配置数量: {sub_type_counts[current_sub_type]})"
                print(f"  {current_sub_type}: {count_info}")
                all_samples.extend(samples)
            
            if not all_samples:
                print(f"  跳过: 没有找到任何样本")
                continue
            
            print(f"  总计加载了 {len(all_samples)} 个样本（来自所有子类型）")
            
            # 筛选样本
            filtered_samples = filter_samples(all_samples)
            print(f"  筛选后: {len(filtered_samples)} 个样本")
            
            # 计算目标总数（所有子类型配置的总和）
            target_count = category_total_count if category_total_count > 0 else None
            
            if target_count is not None:
                print(f"  目标总数: {target_count}（所有子类型配置的总和）")
            
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
            
            # 对于eval，需要排除已存在的样本
            if split_type == "eval" and existing_count > 0:
                original_count = len(filtered_samples)
                filtered_samples = [
                    sample for sample in filtered_samples
                    if not (
                        sample.get("unique_id") in existing_identifiers or
                        (sample.get("source_file", ""), sample.get("source_line", -1), sample.get("true_index", -1)) in existing_identifiers
                    )
                ]
                print(f"  排除已存在样本后: {len(filtered_samples)} 个样本（移除了 {original_count - len(filtered_samples)} 个）")
            
            # 如果配置了目标数量且筛选后样本数量多于目标数量，打乱并取前n个
            if target_count is not None:
                if split_type == "eval":
                    # eval需要补足的数量
                    needed_count = target_count - existing_count
                    if needed_count > 0 and len(filtered_samples) > needed_count:
                        # 设置随机种子以确保可重现性（基于split_type和image_count_category）
                        seed_str = f"{RANDOM_SEED}_{split_type}_{image_count_category}"
                        seed_hash = get_deterministic_seed(seed_str)
                        random.seed(seed_hash)
                        random.shuffle(filtered_samples)
                        # 取前needed_count个
                        filtered_samples = filtered_samples[:needed_count]
                        print(f"  打乱后取前 {needed_count} 个样本用于补足")
                    elif needed_count > 0:
                        print(f"  样本数量 {len(filtered_samples)} <= 需要补足数量 {needed_count}，保留所有样本用于补足")
                else:
                    # train保持原有逻辑
                    if len(filtered_samples) > target_count:
                        # 设置随机种子以确保可重现性（基于split_type和image_count_category）
                        seed_str = f"{RANDOM_SEED}_{split_type}_{image_count_category}"
                        seed_hash = get_deterministic_seed(seed_str)
                        random.seed(seed_hash)
                        random.shuffle(filtered_samples)
                        # 取前target_count个
                        filtered_samples = filtered_samples[:target_count]
                        print(f"  打乱后取前 {target_count} 个样本")
                    else:
                        print(f"  样本数量 {len(filtered_samples)} <= 目标数量 {target_count}，保留所有样本")
            else:
                print(f"  无数量限制，保留所有 {len(filtered_samples)} 个样本")
            
            # 转换为最简格式并按序编号保存
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
            
            # 重新排序编号保存
            saved_count = 0
            for i, sample in enumerate(filtered_samples, start=start_idx):
                # 移除临时字段
                sub_type = sample.pop('_sub_type', 'unknown')
                original_idx = sample.pop('_original_idx', -1)
                
                minimal = convert_to_minimal(sample, "spatial", i)
                output_file = output_dir / f"{i:08d}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(minimal, f, ensure_ascii=False, indent=2)
                saved_count += 1
            
            final_count = existing_count + saved_count if split_type == "eval" and existing_count > 0 else saved_count
            print(f"  已保存到: {output_dir}（共 {final_count} 个样本，编号: {start_idx}-{start_idx + saved_count - 1 if saved_count > 0 else start_idx - 1}）")
    
    print(f"\n{'=' * 80}")
    print("所有数据处理完成！")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

