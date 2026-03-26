#!/usr/bin/env python3
"""
Illustration数据筛选脚本

功能：
1. 从final/illustration目录读取数据
2. 筛选分数高于阈值的样本
3. 筛去image_contributions均为false的样本
4. 转换为最简格式，只保留: task, idx, prompt, input_images, output_image
5. 保存到filter/illustration目录
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


def get_image_count_category(image_count: int) -> str:
    """根据图像数量返回类别目录名"""
    if image_count <= 3:
        return "1-3"
    elif image_count <= 5:
        return "4-5"
    elif image_count <= 7:
        return "6-7"
    else:
        return ">=8"

# ====== 配置参数 ======
FINAL_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "final" / "illustration")
FILTER_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "filter" / "illustration")
# You can override FILTER_DIR to use a custom path if needed

# 筛选阈值
TRAINING_SCORE_THRESHOLD = 6
GUIDANCE_SCORE_THRESHOLD = 6

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
        # 检查image_contributions并计算effective_image_count（有效的image_contributions为true的数量）
        image_contributions = sample.get("image_contributions", [])
        input_images = sample.get("input_images", [])
        if input_images == [] or image_contributions == []:
            continue
        assert len(input_images) == len(image_contributions), f"input_images和image_contributions长度不一致: {len(input_images)} != {len(image_contributions)}, sample: {sample.get('idx')}"
        
        filtered_input_images = [input_images[i] for i in range(len(input_images)) if image_contributions[i]]
        if len(filtered_input_images) == 0 or len(filtered_input_images) > 10:
            continue

        if isinstance(image_contributions, list):
            effective_image_count = sum(1 for x in image_contributions if x is True)
        else:
            effective_image_count = sample.get("effective_image_count", 0)
        assert effective_image_count == len(filtered_input_images), f"effective_image_count和filtered_input_images长度不一致: {effective_image_count} != {len(filtered_input_images)}, sample: {sample.get('idx')}"

        sample['input_images'] = filtered_input_images
        image_count = sample.get("image_count", 0)
        
        # 筛去effective_image_count为0或image_count为0的样本
        if effective_image_count == 0 or image_count == 0:
            continue

        is_invalid = sample.get("is_invalid", False)
        
        # 检查分数
        training_score = sample.get("suitable", sample.get("training_score", 0))
        guidance_score = sample.get("guidance_score", 0)
        
        # 筛选分数高于阈值的样本
        if training_score >= TRAINING_SCORE_THRESHOLD and guidance_score >= GUIDANCE_SCORE_THRESHOLD \
            and effective_image_count <= 10:
            if not is_invalid:
                filtered.append(sample)
            else:
                # print(f"  筛去无效样本: {sample.get('unique_id')}")
                continue
    
    return filtered


def get_actual_image_count(sample: Dict[str, Any]) -> int:
    """
    获取样本实际输入的图像数量（不包括输出图像）
    
    Args:
        sample: 样本数据
    
    Returns:
        实际输入图像数量（仅计算input_images的数量）
    """
    input_images = sample.get("input_images", [])
    
    # 只计算输入图像的数量，不包括输出图像
    count = len(input_images) if isinstance(input_images, list) else 0
    
    return count


def main():
    """主函数"""
    print("=" * 80)
    print("Illustration数据筛选脚本")
    print("=" * 80)
    print(f"Final目录: {FINAL_DIR}")
    print(f"Filter目录: {FILTER_DIR}")
    print(f"Training Score阈值: {TRAINING_SCORE_THRESHOLD}")
    print(f"Guidance Score阈值: {GUIDANCE_SCORE_THRESHOLD}")
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
    
    # 处理train和eval数据
    for split_type in ["train", "eval"]:
        print(f"\n处理 {split_type} 数据...")
        
        # 一次性读取所有final目录下的数据
        print("  正在读取所有数据...")
        all_samples = []
        split_dir = FINAL_DIR / split_type
        if not split_dir.exists():
            print(f"  跳过: {split_type} 目录不存在")
            continue
        
        # 遍历所有image_count_category目录
        for category_dir in split_dir.glob("*"):
            if not category_dir.is_dir():
                continue
            
            json_dir = category_dir / "json"
            if not json_dir.exists():
                continue
            
            # 读取该目录下的所有JSON文件
            for json_file in sorted(json_dir.glob("*.json")):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        sample = json.load(f)
                        # 提取idx
                        idx = int(json_file.stem)
                        all_samples.append((idx, sample))
                except Exception as e:
                    print(f"  警告: 读取JSON文件失败 {json_file}: {e}")
                    continue
        
        print(f"  共加载了 {len(all_samples)} 个样本")
        
        # 筛选样本
        filtered_samples = filter_samples([s[1] for s in all_samples])
        print(f"  筛选后: {len(filtered_samples)} 个样本")
        
        # 创建筛选后的样本及其索引的列表
        filtered_with_idx = [(idx, sample) for idx, sample in all_samples if sample in filtered_samples]
        
        # 使用 unique_id 去重，避免不同 category 目录下相同 idx 的样本重复处理
        print("  正在去重（基于 unique_id 或 source_file+source_line+true_index）...")
        seen_unique_ids = set()
        seen_source_keys = set()  # 用于没有 unique_id 的样本
        deduplicated_samples = []
        duplicate_count = 0
        
        for idx, sample in filtered_with_idx:
            unique_id = sample.get("unique_id")
            if unique_id:
                # 使用 unique_id 去重
                if unique_id in seen_unique_ids:
                    duplicate_count += 1
                    continue
                seen_unique_ids.add(unique_id)
            else:
                # 对于没有 unique_id 的样本，使用 source_file+source_line+true_index 组合去重
                source_file = sample.get("source_file", "")
                source_line = sample.get("source_line", -1)
                true_index = sample.get("true_index", -1)
                source_key = (source_file, source_line, true_index)
                if source_key in seen_source_keys:
                    duplicate_count += 1
                    continue
                seen_source_keys.add(source_key)
            
            deduplicated_samples.append((idx, sample))
        
        if duplicate_count > 0:
            print(f"  去重: 移除了 {duplicate_count} 个重复样本")
        print(f"  去重后: {len(deduplicated_samples)} 个样本")
        
        # 根据实际图像数量重新分类
        print("  正在根据实际图像数量重新分类...")
        samples_by_category = {}  # {image_count_category: [(idx, sample), ...]}
        
        for idx, sample in deduplicated_samples:
            # 计算实际图像数量
            actual_count = get_actual_image_count(sample)
            
            # 根据实际图像数量计算新的类别
            new_category = get_image_count_category(actual_count)
            
            if new_category not in samples_by_category:
                samples_by_category[new_category] = []
            samples_by_category[new_category].append((idx, sample))
        
        print(f"  重新分类后，共 {len(samples_by_category)} 个类别:")
        for category, samples_list in sorted(samples_by_category.items()):
            print(f"    {category}: {len(samples_list)} 个样本")
        
        # 按类别保存
        for image_count_category, samples_list in sorted(samples_by_category.items()):
            print(f"\n  处理类别: {image_count_category} ({len(samples_list)} 个样本)")
            
            # 检查是否有配置，如果没有配置则跳过数量控制
            if image_count_category not in FILTER_CONFIG:
                print(f"    (无数量限制)")
                target_count = None
            else:
                target_count = FILTER_CONFIG[image_count_category].get(split_type)
                if target_count is None:
                    print(f"    (无数量限制)")
                    target_count = None
                else:
                    print(f"    (目标数量: {target_count})")

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
                    print(f"    Eval数据已满足目标数量 ({existing_count} >= {target_count})，跳过")
                    continue
                elif existing_count > 0:
                    print(f"    现有Eval数据: {existing_count} 个，目标: {target_count}，需要补足 {target_count - existing_count} 个")
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
                            print(f"    警告: 读取现有文件失败 {existing_file}: {e}")
                            continue
            
            # 对于eval，需要排除已存在的样本
            if split_type == "eval" and existing_count > 0:
                original_count = len(samples_list)
                samples_list = [
                    (idx, sample) for idx, sample in samples_list
                    if not (
                        sample.get("unique_id") in existing_identifiers or
                        (sample.get("source_file", ""), sample.get("source_line", -1), sample.get("true_index", -1)) in existing_identifiers
                    )
                ]
                print(f"    排除已存在样本后: {len(samples_list)} 个样本（移除了 {original_count - len(samples_list)} 个）")
            
            # 如果配置了目标数量且筛选后样本数量多于目标数量，打乱并取前n个
            if target_count is not None:
                if split_type == "eval":
                    # eval需要补足的数量
                    needed_count = target_count - existing_count
                    if needed_count > 0 and len(samples_list) > needed_count:
                        # 设置随机种子以确保可重现性（基于split_type和image_count_category）
                        seed_str = f"{RANDOM_SEED}_{split_type}_{image_count_category}"
                        seed_hash = get_deterministic_seed(seed_str)
                        random.seed(seed_hash)
                        random.shuffle(samples_list)
                        # 取前needed_count个
                        samples_list = samples_list[:needed_count]
                        print(f"    打乱后取前 {needed_count} 个样本用于补足")
                else:
                    # train保持原有逻辑
                    if len(samples_list) > target_count:
                        # 设置随机种子以确保可重现性（基于split_type和image_count_category）
                        seed_str = f"{RANDOM_SEED}_{split_type}_{image_count_category}"
                        seed_hash = get_deterministic_seed(seed_str)
                        random.seed(seed_hash)
                        random.shuffle(samples_list)
                        # 取前target_count个
                        samples_list = samples_list[:target_count]
                        print(f"    打乱后取前 {target_count} 个样本")
            
            # 转换为最简格式并保存
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if split_type == "eval" and existing_count > 0:
                if target_count is not None and existing_count >= target_count:
                    print(f"    Eval数据已满足目标数量 ({existing_count} >= {target_count})，跳过")
                    continue

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
            for i, (original_idx, sample) in enumerate(samples_list, start=start_idx):
                minimal = convert_to_minimal(sample, "illustration", i)
                output_file = output_dir / f"{i:08d}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(minimal, f, ensure_ascii=False, indent=2)
            
            final_count = existing_count + len(samples_list) if split_type == "eval" and existing_count > 0 else len(samples_list)
            print(f"    已保存到: {output_dir}（共 {final_count} 个样本）")
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()

