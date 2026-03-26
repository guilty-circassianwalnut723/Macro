#!/usr/bin/env python3
"""
Illustration数据拆分脚本：从processed/illustration中采样并分离训练测试集

功能：
1. 读取processed/illustration目录下所有jsonl文件
2. 按照类别标签和图像数量分组
3. 根据配置的eval数量进行采样
4. 其余数据全部用于train
5. 生成train和eval数据集，保存到data_hl02/split/illustration目录
6. 保存为json格式（使用list），控制保存内容数量，避免大量无效占用
7. 记录topic category和image num category统计信息
8. 如果数据量大，按文件拆分为多个子文件保存
"""

import json
import random
from collections import defaultdict
from pathlib import Path

MACRO_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = MACRO_DIR / "data"
from typing import Any, Dict, List, Optional, Set, Tuple
from tqdm import tqdm

# ====== 配置参数 ======
PROCESSED_DIR = DATA_DIR / "processed" / "illustration"
SOURCE_DIR = DATA_DIR / "source" / "illustration"
OUTPUT_DIR = DATA_DIR / "split" / "illustration"

# Eval样本数量配置：{image_group: eval_count}
EVAL_COUNTS = {
    "1-3": 500,
    "4-5": 500,
    "6-7": 500,
    ">=8": 500,
}

# 最多读取的jsonl文件数量，None表示读取所有文件
MAX_FILES: Optional[int] = None

# 每个json文件的最大样本数（用于拆分大文件）
MAX_SAMPLES_PER_FILE = 10000

# 随机种子
RANDOM_SEED = 42
# ======================


def find_true_indices(image_information_flow: List[bool]) -> List[int]:
    """找到所有为true的索引"""
    return [i for i, val in enumerate(image_information_flow) if val]


def get_image_group(image_count: int) -> str:
    """根据图像数量确定分组"""
    if image_count <= 3:
        return "1-3"
    elif image_count <= 5:
        return "4-5"
    elif image_count <= 7:
        return "6-7"
    else:
        return ">=8"


def load_jsonl_files(data_dir: Path, max_files: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载jsonl文件中的数据"""
    all_samples = []
    all_jsonl_files = list(data_dir.glob("*.jsonl"))
    jsonl_files = sorted(all_jsonl_files)
    
    if max_files is not None and max_files > 0:
        jsonl_files = jsonl_files[:max_files]
        print(f"找到 {len(all_jsonl_files)} 个jsonl文件，将读取前 {len(jsonl_files)} 个")
    else:
        print(f"找到 {len(jsonl_files)} 个jsonl文件")
    
    for jsonl_file in tqdm(jsonl_files, desc="加载文件"):
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    if sample.get("state") == "success":
                        all_samples.append(sample)
                except json.JSONDecodeError:
                    continue
    
    return all_samples


def load_original_sample(source_dir: Path, source_file: str, source_line: int) -> Optional[Dict[str, Any]]:
    """从源文件中加载原始样本以获取图像数量信息"""
    file_path = source_dir / source_file
    if not file_path.exists():
        return None
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                if line_num == source_line:
                    line = line.strip()
                    if not line:
                        return None
                    return json.loads(line)
    except Exception:
        return None
    return None


def get_true_indices_and_image_count_from_sample(
    sample: Dict[str, Any], source_dir: Path
) -> Optional[Dict[str, Any]]:
    """从样本中获取true索引和图像数量信息"""
    source_file = sample.get("source_file")
    source_line = sample.get("source_line")
    if source_file and source_line:
        original_sample = load_original_sample(source_dir, source_file, source_line)
        if original_sample:
            scores = original_sample.get("scores", {})
            if scores and isinstance(scores, dict):
                image_information_flow = scores.get("image_information_flow", [])
                if isinstance(image_information_flow, list):
                    true_indices = find_true_indices(image_information_flow)
                    meta = original_sample.get("meta", {})
                    image_count = meta.get("image_count")
                    if image_count is None:
                        image_count = len(image_information_flow)
                    
                    return {
                        "true_indices": true_indices,
                        "image_count": image_count,
                        "original_sample": original_sample,
                    }
    
    return None


def create_minimal_sample(sample: Dict[str, Any], true_index: int, image_group: str) -> Dict[str, Any]:
    """创建最小化的样本，只保留必要字段，避免大量无效占用"""
    minimal = {
        "source_file": sample.get("source_file"),
        "source_line": sample.get("source_line"),
        "category": sample.get("category"),
        "true_index": true_index,
        "image_count": true_index + 1,
        "actual_image_count": true_index,
        "image_num_category": image_group,  # 添加image_num_category字段
    }
    
    # 只保留必要的字段，避免保存大量无效数据
    # 如果需要其他字段，可以在gen阶段从source重新加载
    return minimal


def organize_samples_by_category_and_image_group(
    samples: List[Dict[str, Any]],
    source_dir: Path,
) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    """按类别和图像数量分组组织样本"""
    organized: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    
    print("正在组织样本...")
    for sample in tqdm(samples, desc="分组"):
        category = sample.get("category")
        if category is None:
            continue
        
        info = get_true_indices_and_image_count_from_sample(sample, source_dir)
        if info is None:
            continue
        
        true_indices = info["true_indices"]
        
        if not true_indices:
            continue
        
        # 为每个true索引创建一个独立的样本
        for true_index in true_indices:
            image_group = get_image_group(true_index)
            minimal_sample = create_minimal_sample(sample, true_index, image_group)
            organized[category][image_group].append(minimal_sample)
    
    return organized


def sample_eval_data_by_image_group(
    organized_samples: Dict[int, Dict[str, List[Dict[str, Any]]]],
    eval_limits: Dict[str, int],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Set[Tuple[str, int]]]:
    """按image_group采样评估数据，优先保证从不同topic category中均匀采样"""
    rng = random.Random(seed)
    all_selected_samples = []
    used_samples: Set[Tuple[str, int]] = set()
    
    # 按image_group处理
    all_image_groups = set()
    for category_data in organized_samples.values():
        all_image_groups.update(category_data.keys())
    
    for img_grp in sorted(all_image_groups):
        eval_limit = eval_limits.get(img_grp, 0)
        if eval_limit <= 0:
            continue
        
        # 收集该image_group下所有category的可用样本
        available_by_category: Dict[int, List[Dict[str, Any]]] = {}
        for category_id, category_data in organized_samples.items():
            if img_grp not in category_data:
                continue
            samples_list = category_data[img_grp]
            available = [
                s for s in samples_list
                if (s.get("source_file"), s.get("source_line")) not in used_samples
            ]
            if available:
                available_by_category[category_id] = available
        
        if not available_by_category:
            continue
        
        # 按sample_id分组，为每个category建立sample_id索引
        category_samples_by_id: Dict[int, Dict[Tuple[str, int], List[Dict[str, Any]]]] = {}
        category_total_counts: Dict[int, int] = {}  # 记录每个category的总样本数
        for category_id, samples in available_by_category.items():
            samples_by_id: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
            for sample in samples:
                sample_id = (sample.get("source_file"), sample.get("source_line"))
                if sample_id[0] and sample_id[1] is not None:
                    samples_by_id[sample_id].append(sample)
            category_samples_by_id[category_id] = samples_by_id
            category_total_counts[category_id] = len(samples_by_id)  # 记录unique sample_id数量
        
        # 优先从每个category均匀采样
        num_categories = len(available_by_category)
        if num_categories == 0:
            continue
        
        # 计算每个category的目标采样数
        samples_per_category = eval_limit // num_categories
        remaining_needed = eval_limit
        
        selected_samples = []
        category_ids = list(available_by_category.keys())
        rng.shuffle(category_ids)  # 随机化category处理顺序
        
        # 跟踪每个category已分配的eval样本数
        category_eval_counts: Dict[int, int] = {cat_id: 0 for cat_id in category_ids}
        
        # 第一阶段：尝试从每个category均匀采样
        for category_id in category_ids:
            if remaining_needed <= 0:
                break
            
            samples_by_id = category_samples_by_id[category_id]
            unique_sample_ids = list(samples_by_id.keys())
            rng.shuffle(unique_sample_ids)
            
            # 计算该category的最大可分配eval数量：确保训练数据 >= eval数据
            # 如果总共有N条，最多分配floor(N/2)条给eval
            total_count = category_total_counts[category_id]
            max_eval_for_category = total_count // 2  # 最多分配一半给eval
            
            # 从该category采样，最多samples_per_category个，但不超过剩余需求和最大限制
            target_count = min(samples_per_category, len(unique_sample_ids), remaining_needed, max_eval_for_category)
            
            for sample_id in unique_sample_ids[:target_count]:
                if remaining_needed <= 0:
                    break
                # 再次检查：确保分配后训练数据不会少于eval数据
                current_eval_count = category_eval_counts[category_id]
                # 如果分配后eval数量会超过最大限制，跳过
                if current_eval_count >= max_eval_for_category:
                    break
                # 检查分配后是否满足约束：train >= eval
                # 分配后：eval = current_eval_count + 1, train = total_count - (current_eval_count + 1)
                # 需要确保：train >= eval，即 total_count >= 2 * (current_eval_count + 1)
                if total_count < 2 * (current_eval_count + 1):
                    break
                
                samples_for_id = samples_by_id[sample_id]
                if samples_for_id:
                    selected_sample = rng.choice(samples_for_id)
                    selected_samples.append(selected_sample)
                    used_samples.add(sample_id)
                    category_eval_counts[category_id] += 1
                    remaining_needed -= 1
        
        # 第二阶段：如果还有剩余需求，从所有category中随机采样补齐
        if remaining_needed > 0:
            # 收集所有未使用的sample_id，并检查是否满足约束
            all_remaining_samples_by_id: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
            sample_id_to_category: Dict[Tuple[str, int], int] = {}  # 记录sample_id属于哪个category
            
            for category_id, samples_by_id in category_samples_by_id.items():
                current_eval_count = category_eval_counts[category_id]
                total_count = category_total_counts[category_id]
                
                # 该category最多能分配的eval数量（确保train >= eval）
                max_eval_for_category = total_count // 2
                
                # 如果已经达到最大限制，跳过该category
                if current_eval_count >= max_eval_for_category:
                    continue
                
                for sample_id, samples in samples_by_id.items():
                    if sample_id not in used_samples:
                        all_remaining_samples_by_id[sample_id].extend(samples)
                        sample_id_to_category[sample_id] = category_id
            
            unique_sample_ids = list(all_remaining_samples_by_id.keys())
            rng.shuffle(unique_sample_ids)
            
            for sample_id in unique_sample_ids[:remaining_needed]:
                if sample_id not in sample_id_to_category:
                    continue
                
                sample_category_id = sample_id_to_category[sample_id]
                current_eval_count = category_eval_counts[sample_category_id]
                total_count = category_total_counts[sample_category_id]
                max_eval_for_category = total_count // 2
                
                # 检查是否还能分配（不能超过最大限制，且分配后train >= eval）
                if current_eval_count >= max_eval_for_category:
                    continue
                
                # 分配后：eval = current_eval_count + 1, train = total_count - (current_eval_count + 1)
                # 需要确保：train >= eval，即 total_count - (current_eval_count + 1) >= current_eval_count + 1
                # 即：total_count >= 2 * (current_eval_count + 1)
                if total_count < 2 * (current_eval_count + 1):
                    continue
                
                samples_for_id = all_remaining_samples_by_id[sample_id]
                if samples_for_id:
                    selected_sample = rng.choice(samples_for_id)
                    selected_samples.append(selected_sample)
                    used_samples.add(sample_id)
                    category_eval_counts[sample_category_id] += 1
        
        all_selected_samples.extend(selected_samples)
    
    return all_selected_samples, used_samples


def save_json(samples: List[dict], output_path: Path):
    """保存样本到json文件（使用list格式）"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def save_samples_in_chunks(
    samples: List[dict],
    output_dir: Path,
    split_type: str,
    max_samples_per_file: int = 10000,
):
    """保存样本到json文件（使用list格式），如果数据量大则拆分为多个文件"""
    if not samples:
        return
    
    if len(samples) <= max_samples_per_file:
        # 单个文件
        output_file = output_dir / f"{split_type}.json"
        save_json(samples, output_file)
        print(f"已保存 {len(samples)} 个{split_type}样本到: {output_file}")
    else:
        # 多个文件
        num_files = (len(samples) + max_samples_per_file - 1) // max_samples_per_file
        for file_idx in range(num_files):
            start_idx = file_idx * max_samples_per_file
            end_idx = min(start_idx + max_samples_per_file, len(samples))
            chunk = samples[start_idx:end_idx]
            
            output_file = output_dir / f"{split_type}_{file_idx:04d}.json"
            save_json(chunk, output_file)
            print(f"已保存 {len(chunk)} 个{split_type}样本到: {output_file} (文件 {file_idx + 1}/{num_files})")


def main():
    """主函数"""
    print("=" * 80)
    print("Illustration数据拆分脚本")
    print("=" * 80)
    print(f"处理目录: {PROCESSED_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"最多读取文件数: {MAX_FILES if MAX_FILES else '全部'}")
    print(f"每个文件最大样本数: {MAX_SAMPLES_PER_FILE}")
    print(f"\nEval样本数配置:")
    for img_grp, count in EVAL_COUNTS.items():
        print(f"  {img_grp}: {count}")
    print("=" * 80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载所有样本
    print("\n步骤1: 加载样本...")
    all_samples = load_jsonl_files(PROCESSED_DIR, max_files=MAX_FILES)
    print(f"共加载 {len(all_samples)} 个成功样本")
    
    # 组织样本
    print("\n步骤2: 按类别和图像数量分组...")
    organized_samples = organize_samples_by_category_and_image_group(all_samples, SOURCE_DIR)
    
    print(f"\n找到 {len(organized_samples)} 个类别")
    for category_id in sorted(organized_samples.keys()):
        category_data = organized_samples[category_id]
        total = sum(len(samples) for samples in category_data.values())
        print(f"  类别 {category_id}: {total} 个样本")
        for img_grp, samples in category_data.items():
            print(f"    {img_grp}: {len(samples)} 个样本")
    
    # 采样eval数据
    print("\n步骤3: 采样eval数据...")
    random.seed(RANDOM_SEED)
    all_eval_samples, used_samples = sample_eval_data_by_image_group(
        organized_samples, EVAL_COUNTS, RANDOM_SEED
    )
    
    # 组织train数据（所有未用于eval的样本）
    print("\n步骤4: 组织train数据...")
    all_train_samples = []
    for category_id, category_data in organized_samples.items():
        for img_grp, samples in category_data.items():
            for sample in samples:
                sample_id = (sample.get("source_file"), sample.get("source_line"))
                if sample_id not in used_samples:
                    all_train_samples.append(sample)
    
    # 计算统计信息：{topic_category: {image_num_category: count}}
    category_stats = {}  # {topic_category: {image_num_category: {train: count, eval: count}}}
    
    for category_id in organized_samples.keys():
        category_stats[category_id] = {
            "1-3": {"train": 0, "eval": 0},
            "4-5": {"train": 0, "eval": 0},
            "6-7": {"train": 0, "eval": 0},
            ">=8": {"train": 0, "eval": 0},
        }
    
    # 统计train数据
    for sample in all_train_samples:
        category_id = sample.get("category")
        image_group = sample.get("image_num_category")
        if category_id is not None and image_group:
            if category_id in category_stats and image_group in category_stats[category_id]:
                category_stats[category_id][image_group]["train"] += 1
    
    # 统计eval数据
    for sample in all_eval_samples:
        category_id = sample.get("category")
        image_group = sample.get("image_num_category")
        if category_id is not None and image_group:
            if category_id in category_stats and image_group in category_stats[category_id]:
                category_stats[category_id][image_group]["eval"] += 1
    
    # 保存train和eval数据
    print("\n步骤5: 保存数据...")
    
    # 保存train数据（按文件拆分）
    if all_train_samples:
        save_samples_in_chunks(
            all_train_samples,
            OUTPUT_DIR,
            "train",
            MAX_SAMPLES_PER_FILE
        )
    
    # 保存eval数据（按文件拆分）
    if all_eval_samples:
        save_samples_in_chunks(
            all_eval_samples,
            OUTPUT_DIR,
            "eval",
            MAX_SAMPLES_PER_FILE
        )
    
    # 保存统计信息（包含topic category和image num category统计）
    stats_data = {
        'topic_category_statistics': category_stats,  # {topic_category: {image_num_category: {train: count, eval: count}}}
        'summary': {
            'total_train': len(all_train_samples),
            'total_eval': len(all_eval_samples),
            'total_all': len(all_train_samples) + len(all_eval_samples),
        },
    }
    
    stats_file = OUTPUT_DIR / "statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)
    print(f"\n已保存统计信息到: {stats_file}")
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("拆分统计信息")
    print("=" * 80)
    print(f"总Train样本数: {len(all_train_samples)}")
    print(f"总Eval样本数: {len(all_eval_samples)}")
    print(f"总样本数: {len(all_train_samples) + len(all_eval_samples)}")
    print("\n按topic category和image num category统计:")
    for category_id in sorted(category_stats.keys()):
        print(f"  类别 {category_id}:")
        for img_grp in ["1-3", "4-5", "6-7", ">=8"]:
            train_count = category_stats[category_id][img_grp]["train"]
            eval_count = category_stats[category_id][img_grp]["eval"]
            if train_count > 0 or eval_count > 0:
                print(f"    {img_grp}: Train={train_count}, Eval={eval_count}, Total={train_count + eval_count}")
    print("=" * 80)
    
    print("\n拆分完成！")


if __name__ == "__main__":
    main()
