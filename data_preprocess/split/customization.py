#!/usr/bin/env python3
"""
Customization数据拆分脚本：将每个类别的数据拆分为train和eval

功能：
1. 读取processed/customization目录下每个类别的jsonl文件
2. 根据配置的eval样本数量，将数据拆分为train和eval
3. 对于scene类别，特殊处理：100个multi-frame场景 + 100个single-frame场景
4. 支持从已有数据中扩展eval：如果需要的eval数量超过已有的，从final_old中未被使用的数据源添加到新的eval中
5. 保存到data_hl02/split/customization目录，格式为json（使用list）

注意：
- 用户只需指定eval数量，train数量自动计算为（总数据量 - eval数量）
- 具体数据量由gen决定，split只负责划分
"""

import json
import random
import hashlib
from pathlib import Path

MACRO_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = MACRO_DIR / "data"
from tqdm import tqdm
from typing import Set, Dict, List

# ====== 配置参数 ======
DATA_DIR = DATA_DIR / "processed" / "customization"
OUTPUT_DIR = DATA_DIR / "split" / "customization"
FINAL_DIR = DATA_DIR / "final" / "customization"
FINAL_OLD_DIR = DATA_DIR / "final_old" / "customization"

# 配置每类eval样本数量（None表示不拆分，全部用于train）
# 注意：用户只需指定eval数量，train数量自动计算为（总数据量 - eval数量）
EVAL_COUNTS = {
    'human': 500,
    'cloth': 300,
    'object': 500,
    'scene': 200,  # 200个场景：100个多帧 + 100个单帧
    'style': 300,  # 从200扩展到300
}

# 随机种子
RANDOM_SEED = 42
# ======================


def load_jsonl(jsonl_path: Path) -> list:
    """加载jsonl文件"""
    samples = []
    if not jsonl_path.exists():
        print(f"警告: 文件不存在: {jsonl_path}")
        return samples
    
    print(f"正在加载 {jsonl_path}...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                samples.append(data)
            except json.JSONDecodeError as e:
                print(f"警告: 无法解析JSON行: {e}")
                continue
    
    print(f"加载了 {len(samples)} 个样本")
    return samples


def split_samples(samples: list, eval_count: int, seed: int = 42) -> tuple:
    """将样本拆分为train和eval"""
    if eval_count is None or eval_count <= 0:
        return samples, []
    
    if len(samples) <= eval_count:
        print(f"警告: 样本总数 {len(samples)} 小于等于eval数量 {eval_count}，所有样本用于eval")
        return [], samples
    
    random.seed(seed)
    shuffled_samples = samples.copy()
    random.shuffle(shuffled_samples)
    
    eval_samples = shuffled_samples[:eval_count]
    train_samples = shuffled_samples[eval_count:]
    
    return train_samples, eval_samples


def split_scene_samples(samples: list, eval_scene_count: int, seed: int = 42) -> tuple:
    """
    特殊处理scene场景：将样本拆分为train和eval
    根据文件名识别多帧场景和单帧场景：
    - 多帧场景：文件名格式为 {scene_idx}_{frame_idx}.jpg，例如 00000001_1.jpg
    - 单帧场景：文件名格式为 {scene_idx}.jpg，例如 00006379.jpg
    
    - 100个多帧场景，每个场景的所有帧都加入到eval（约500个样本）
    - 100个单帧场景，每个场景1帧（100个样本）
    - 总共约600个eval样本
    """
    if eval_scene_count is None or eval_scene_count <= 0:
        return samples, []
    
    random.seed(seed)
    
    # 根据文件名识别多帧场景和单帧场景
    multi_frame_groups = {}  # key: scene_idx, value: list of samples (该场景的所有帧)
    single_frame_samples = []  # 单帧场景的样本列表
    
    for sample in samples:
        filename = sample.get('filename', '')
        if not filename:
            filepath = sample.get('filepath', '')
            if filepath:
                filename = Path(filepath).name
            else:
                single_frame_samples.append(sample)
                continue
        
        # 检查是否为多帧场景：文件名格式为 {scene_idx}_{frame_idx}.jpg
        if '_' in filename and filename.endswith('.jpg'):
            parts = filename.rsplit('_', 1)
            if len(parts) == 2:
                scene_idx = parts[0]
                frame_part = parts[1].replace('.jpg', '')
                if frame_part.isdigit():
                    if scene_idx not in multi_frame_groups:
                        multi_frame_groups[scene_idx] = []
                    multi_frame_groups[scene_idx].append(sample)
                    continue
        
        # 单帧场景
        single_frame_samples.append(sample)
    
    print(f"识别到 {len(multi_frame_groups)} 个多帧场景")
    print(f"识别到 {len(single_frame_samples)} 个单帧场景")
    
    # 需要100个多帧场景 + 100个单帧场景
    multi_frame_count = 100
    single_frame_count = 100
    
    # 打乱多帧场景组
    multi_frame_scene_indices = list(multi_frame_groups.keys())
    random.shuffle(multi_frame_scene_indices)
    
    # 打乱单帧场景
    random.shuffle(single_frame_samples)
    
    # 选择多帧场景（前multi_frame_count个）
    selected_multi_frame_indices = multi_frame_scene_indices[:multi_frame_count]
    
    # 选择单帧场景（前single_frame_count个）
    selected_single_frame_samples = single_frame_samples[:single_frame_count]
    
    # 收集eval样本
    eval_samples = []
    
    # 添加多帧场景的所有帧
    for scene_idx in selected_multi_frame_indices:
        frames = multi_frame_groups[scene_idx]
        eval_samples.extend(frames)
        print(f"场景 {scene_idx}: {len(frames)} 帧")
    
    # 添加单帧场景
    eval_samples.extend(selected_single_frame_samples)
    
    # 收集train样本
    train_samples = []
    
    # 剩余的多帧场景
    remaining_multi_frame_indices = multi_frame_scene_indices[multi_frame_count:]
    for scene_idx in remaining_multi_frame_indices:
        train_samples.extend(multi_frame_groups[scene_idx])
    
    # 剩余的单帧场景
    remaining_single_frame_samples = single_frame_samples[single_frame_count:]
    train_samples.extend(remaining_single_frame_samples)
    
    print(f"多帧场景数: {len(selected_multi_frame_indices)} (共 {sum(len(multi_frame_groups[idx]) for idx in selected_multi_frame_indices)} 个样本)")
    print(f"单帧场景数: {len(selected_single_frame_samples)} (共 {len(selected_single_frame_samples)} 个样本)")
    print(f"Eval总样本数: {len(eval_samples)}")
    print(f"Train总样本数: {len(train_samples)}")
    
    return train_samples, eval_samples


def get_combination_key(files: List[str]) -> str:
    """生成图像组合的唯一键（用于去重）"""
    sorted_files = sorted(files)
    key_str = "|".join(sorted_files)
    return hashlib.md5(key_str.encode()).hexdigest()


def load_existing_eval_data(final_old_dir: Path, class_name: str) -> Dict[str, dict]:
    """
    从final_old目录加载已有的eval数据
    
    Args:
        final_old_dir: final_old目录路径
        class_name: 类别名称
    
    Returns:
        {combination_key: sample_data} 字典
    """
    existing_eval = {}
    
    if not final_old_dir.exists():
        return existing_eval
    
    # 遍历所有image_count_category目录
    for category_dir in final_old_dir.glob("*"):
        if not category_dir.is_dir():
            continue
        
        eval_json_dir = category_dir / "eval" / "json"
        if not eval_json_dir.exists():
            continue
        
        # 读取所有JSON文件
        for json_file in eval_json_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 检查是否属于当前类别
                    sample_class = data.get('class', '')
                    if sample_class != class_name:
                        continue
                    
                    # 提取combination_key
                    input_images = data.get('input_images', [])
                    if input_images:
                        # 提取文件名作为组合键
                        files = [Path(img).name for img in input_images]
                        combination_key = get_combination_key(files)
                        existing_eval[combination_key] = data
            except Exception as e:
                print(f"警告: 读取JSON文件失败 {json_file}: {e}")
                continue
    
    return existing_eval


def load_processed_samples_by_combination(processed_dir: Path, class_name: str) -> Dict[str, dict]:
    """
    从processed目录加载样本，按combination_key索引
    
    Args:
        processed_dir: processed目录路径
        class_name: 类别名称
    
    Returns:
        {combination_key: sample_data} 字典
    """
    samples_by_key = {}
    jsonl_path = processed_dir / f"{class_name}.jsonl"
    
    if not jsonl_path.exists():
        return samples_by_key
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 提取文件列表
                files = []
                if 'files' in data:
                    files = [f if isinstance(f, str) else f.get('filepath', '') for f in data['files']]
                elif 'filepath' in data:
                    files = [data['filepath']]
                
                if files:
                    # 提取文件名
                    file_names = [Path(f).name for f in files]
                    combination_key = get_combination_key(file_names)
                    samples_by_key[combination_key] = data
            except Exception as e:
                print(f"警告: 解析JSON行失败: {e}")
                continue
    
    return samples_by_key


def extend_eval_from_existing(
    eval_samples: List[dict],
    target_eval_count: int,
    processed_dir: Path,
    final_old_dir: Path,
    class_name: str,
    seed: int = 42
) -> List[dict]:
    """
    从已有数据中扩展eval样本
    
    如果需要的eval数量超过已有的，从final_old中未被使用的数据源添加到新的eval中
    
    Args:
        eval_samples: 当前eval样本列表
        target_eval_count: 目标eval数量
        processed_dir: processed目录路径
        final_old_dir: final_old目录路径
        class_name: 类别名称
        seed: 随机种子
    
    Returns:
        扩展后的eval样本列表
    """
    current_eval_count = len(eval_samples)
    
    if current_eval_count >= target_eval_count:
        return eval_samples
    
    print(f"\n需要扩展eval: 当前 {current_eval_count} 个，目标 {target_eval_count} 个")
    print(f"从已有数据中查找可用的数据源...")
    
    # 加载已有的eval数据（从final_old）
    existing_eval = load_existing_eval_data(final_old_dir, class_name)
    print(f"从final_old加载了 {len(existing_eval)} 个已有eval样本")
    
    # 加载processed中的所有样本，按combination_key索引
    processed_samples = load_processed_samples_by_combination(processed_dir, class_name)
    print(f"从processed加载了 {len(processed_samples)} 个样本")
    
    # 获取当前eval样本的combination_key集合
    current_eval_keys: Set[str] = set()
    for sample in eval_samples:
        files = []
        if 'files' in sample:
            files = [f if isinstance(f, str) else f.get('filepath', '') for f in sample['files']]
        elif 'filepath' in sample:
            files = [sample['filepath']]
        
        if files:
            file_names = [Path(f).name for f in files]
            combination_key = get_combination_key(file_names)
            current_eval_keys.add(combination_key)
    
    # 从已有eval中找到未被使用的样本
    available_eval_keys = []
    for combo_key, eval_data in existing_eval.items():
        if combo_key not in current_eval_keys:
            # 检查是否在processed中存在对应的数据源
            if combo_key in processed_samples:
                available_eval_keys.append(combo_key)
    
    print(f"找到 {len(available_eval_keys)} 个可用的已有eval样本")
    
    # 随机选择需要补充的样本
    need_count = target_eval_count - current_eval_count
    if need_count > 0 and available_eval_keys:
        random.seed(seed)
        selected_keys = random.sample(available_eval_keys, min(need_count, len(available_eval_keys)))
        
        # 从processed中获取对应的样本数据
        for combo_key in selected_keys:
            if combo_key in processed_samples:
                eval_samples.append(processed_samples[combo_key])
        
        print(f"已添加 {len(selected_keys)} 个样本到eval")
    
    return eval_samples


def save_json(samples: list, output_path: Path):
    """保存样本到json文件（使用list格式）"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def load_used_style_from_final(final_dir: Path) -> Set[str]:
    """
    从final/customization目录加载已使用的style数据（train数据）
    
    Args:
        final_dir: final目录路径
    
    Returns:
        已使用的style文件路径集合（标准化路径）
    """
    used_style_paths = set()
    
    if not final_dir.exists():
        return used_style_paths
    
    # 遍历所有image_count_category目录
    for category_dir in final_dir.glob("*"):
        if not category_dir.is_dir():
            continue
        
        train_json_dir = category_dir / "train" / "json"
        if not train_json_dir.exists():
            continue
        
        # 读取所有JSON文件
        for json_file in train_json_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 检查是否包含style类别
                    categories = data.get('category', [])
                    if 'style' not in categories:
                        continue
                    
                    # 提取style图像路径
                    input_images = data.get('input_images', [])
                    for img_path in input_images:
                        img_path_str = str(img_path)
                        # 检查路径中是否包含style目录
                        if '/style/' in img_path_str or '/customization/style' in img_path_str:
                            # 标准化路径：提取相对于processed/customization/style的路径
                            if '/style/' in img_path_str:
                                # 提取文件名或相对路径
                                style_path = img_path_str.split('/style/')[-1]
                                # 构建完整路径
                                full_path = str(DATA_DIR / "style" / style_path)
                                used_style_paths.add(full_path)
                            else:
                                used_style_paths.add(img_path_str)
            except Exception as e:
                print(f"警告: 读取JSON文件失败 {json_file}: {e}")
                continue
    
    return used_style_paths


def extend_style_eval_from_unused(
    eval_samples: List[dict],
    target_eval_count: int,
    processed_dir: Path,
    final_dir: Path,
    seed: int = 42
) -> List[dict]:
    """
    从未使用的style数据中扩展eval样本
    
    Args:
        eval_samples: 当前eval样本列表
        target_eval_count: 目标eval数量
        processed_dir: processed目录路径
        final_dir: final目录路径
        seed: 随机种子
    
    Returns:
        扩展后的eval样本列表
    """
    current_eval_count = len(eval_samples)
    
    if current_eval_count >= target_eval_count:
        return eval_samples
    
    print(f"\n需要扩展style eval: 当前 {current_eval_count} 个，目标 {target_eval_count} 个")
    
    # 加载已使用的style数据（从final/customization的train数据）
    used_style_paths = load_used_style_from_final(final_dir)
    print(f"从final/customization加载了 {len(used_style_paths)} 个已使用的style数据")
    
    # 加载所有style样本
    style_jsonl_path = processed_dir / "style.jsonl"
    all_style_samples = load_jsonl(style_jsonl_path)
    print(f"从processed加载了 {len(all_style_samples)} 个style样本")
    
    # 标准化路径：转换为绝对路径并规范化
    def normalize_path(path_str: str) -> str:
        """标准化路径"""
        if not path_str:
            return ""
        path = Path(path_str)
        if path.is_absolute():
            return str(path.resolve())
        else:
            return str((processed_dir / path_str).resolve())
    
    # 获取当前eval样本的filepath集合（标准化）
    current_eval_paths = set()
    for sample in eval_samples:
        filepath = sample.get('filepath', '')
        if filepath:
            normalized = normalize_path(filepath)
            current_eval_paths.add(normalized)
    
    # 标准化已使用的路径
    normalized_used_paths = {normalize_path(p) for p in used_style_paths}
    
    # 找出未使用的style样本（既不在eval中，也不在final/train中）
    unused_samples = []
    for sample in all_style_samples:
        filepath = sample.get('filepath', '')
        if filepath:
            normalized = normalize_path(filepath)
            if normalized not in current_eval_paths and normalized not in normalized_used_paths:
                unused_samples.append(sample)
    
    print(f"找到 {len(unused_samples)} 个未使用的style样本")
    
    # 随机选择需要补充的样本
    need_count = target_eval_count - current_eval_count
    if need_count > 0 and unused_samples:
        random.seed(seed)
        selected_samples = random.sample(unused_samples, min(need_count, len(unused_samples)))
        eval_samples.extend(selected_samples)
        print(f"已添加 {len(selected_samples)} 个样本到eval")
    elif need_count > 0:
        print(f"警告: 未找到足够的未使用style样本（需要 {need_count} 个，但只有 {len(unused_samples)} 个可用）")
    
    return eval_samples


def main():
    """主函数"""
    print("=" * 80)
    print("Customization数据拆分脚本")
    print("=" * 80)
    print(f"\nEval样本数配置: {EVAL_COUNTS}")
    print(f"随机种子: {RANDOM_SEED}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    all_statistics = {}
    
    # 处理每个类别
    for class_name, eval_count in EVAL_COUNTS.items():
        print("\n" + "=" * 80)
        if eval_count is None or eval_count <= 0:
            print(f"处理类别: {class_name} (eval样本数: 0，全部用于train)")
        else:
            print(f"处理类别: {class_name} (eval样本数: {eval_count})")
        print("=" * 80)
        
        # 对于style类别，从processed/customization继承现有的eval和train分组
        if class_name == 'style':
            # 加载现有的train和eval数据
            train_jsonl_path = DATA_DIR / f"{class_name}_train.jsonl"
            eval_jsonl_path = DATA_DIR / f"{class_name}_eval.jsonl"
            
            train_samples = load_jsonl(train_jsonl_path) if train_jsonl_path.exists() else []
            eval_samples = load_jsonl(eval_jsonl_path) if eval_jsonl_path.exists() else []
            
            print(f"从processed/customization继承: train={len(train_samples)}, eval={len(eval_samples)}")
            
            # 如果eval数量不足，从未使用的数据中扩展
            if len(eval_samples) < eval_count:
                eval_samples = extend_style_eval_from_unused(
                    eval_samples,
                    eval_count,
                    DATA_DIR,
                    FINAL_DIR,
                    RANDOM_SEED
                )
            
            # 确保train_samples中不包含eval_samples中的样本
            eval_filepaths = {s.get('filepath', '') for s in eval_samples if s.get('filepath')}
            filtered_train_samples = [
                s for s in train_samples 
                if s.get('filepath', '') not in eval_filepaths
            ]
            train_samples = filtered_train_samples
        else:
            # 加载jsonl文件
            jsonl_path = DATA_DIR / f"{class_name}.jsonl"
            samples = load_jsonl(jsonl_path)
            
            if not samples:
                print(f"警告: 类别 {class_name} 没有数据，跳过")
                continue
            
            # 对于scene类别，使用特殊处理
            if class_name == 'scene':
                train_samples, eval_samples = split_scene_samples(samples, eval_count, RANDOM_SEED)
            else:
                train_samples, eval_samples = split_samples(samples, eval_count, RANDOM_SEED)
            
            # 如果eval数量不足，从已有数据中扩展
            if eval_count and eval_count > 0 and len(eval_samples) < eval_count:
                eval_samples = extend_eval_from_existing(
                    eval_samples,
                    eval_count,
                    DATA_DIR,
                    FINAL_OLD_DIR,
                    class_name,
                    RANDOM_SEED
                )
            
            # Train数量自动计算为（总数据量 - eval数量）
            # 注意：这里需要确保eval_samples中的样本不会出现在train_samples中
            eval_combination_keys = set()
            for sample in eval_samples:
                files = []
                if 'files' in sample:
                    files = [f if isinstance(f, str) else f.get('filepath', '') for f in sample['files']]
                elif 'filepath' in sample:
                    files = [sample['filepath']]
                
                if files:
                    file_names = [Path(f).name for f in files]
                    combination_key = get_combination_key(file_names)
                    eval_combination_keys.add(combination_key)
            
            # 从train_samples中移除已在eval中的样本
            filtered_train_samples = []
            for sample in train_samples:
                files = []
                if 'files' in sample:
                    files = [f if isinstance(f, str) else f.get('filepath', '') for f in sample['files']]
                elif 'filepath' in sample:
                    files = [sample['filepath']]
                
                if files:
                    file_names = [Path(f).name for f in files]
                    combination_key = get_combination_key(file_names)
                    if combination_key not in eval_combination_keys:
                        filtered_train_samples.append(sample)
                else:
                    filtered_train_samples.append(sample)
            
            train_samples = filtered_train_samples
        
        print(f"Train样本数: {len(train_samples)}")
        print(f"Eval样本数: {len(eval_samples)}")
        
        # 保存train.json
        if train_samples:
            train_path = OUTPUT_DIR / f"{class_name}_train.json"
            save_json(train_samples, train_path)
            print(f"已保存train数据到: {train_path}")
        
        # 保存eval.json
        if eval_samples:
            eval_path = OUTPUT_DIR / f"{class_name}_eval.json"
            save_json(eval_samples, eval_path)
            print(f"已保存eval数据到: {eval_path}")
        
        # 记录统计信息
        all_statistics[class_name] = {
            'total': len(train_samples) + len(eval_samples),
            'train': len(train_samples),
            'eval': len(eval_samples)
        }
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("拆分统计信息")
    print("=" * 80)
    total_train = 0
    total_eval = 0
    total_all = 0
    
    for class_name, stats in sorted(all_statistics.items()):
        print(f"\n类别: {class_name}")
        print(f"  总样本数: {stats['total']}")
        print(f"  Train样本数: {stats['train']}")
        print(f"  Eval样本数: {stats['eval']}")
        total_train += stats['train']
        total_eval += stats['eval']
        total_all += stats['total']
    
    print("\n" + "=" * 80)
    print(f"总计样本数: {total_all}")
    print(f"总计Train样本数: {total_train}")
    print(f"总计Eval样本数: {total_eval}")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("所有类别处理完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

