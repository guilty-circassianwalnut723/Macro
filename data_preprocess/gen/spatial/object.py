#!/usr/bin/env python3
"""
Spatial Object数据生成脚本

功能：
1. 从split/spatial目录读取train/eval数据（object子类型）
2. 从多视角物体图像中采样数据
3. 保存到final/spatial/{train/eval}/{image_count_category}/data和json目录
4. 支持唯一识别编号，避免重复生成
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from tqdm import tqdm
from PIL import Image

# 添加utils路径
CURRENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CURRENT_DIR))

from utils.common import (
    get_image_count_category,
    load_generated_ids,
    save_sample_data,
    generate_unique_id
)

# ====== 配置参数 ======
SPLIT_DIR = (Path(__file__).resolve().parent.parent.parent.parent / "data" / "split" / "spatial")
FINAL_DIR = (Path(__file__).resolve().parent.parent.parent.parent / "data" / "final" / "spatial")

# 生成配置：{image_count_category: {train: count, eval: count}}
GEN_CONFIG = {
    "1-3": {"train": 10000, "eval": 0},
    "4-5": {"train": 0, "eval": 0},
    "6-7": {"train": 0, "eval": 0},
    ">=8": {"train": 0, "eval": 0},
}

# 视角方向定义
VIEW_DIRECTIONS = ['front', 'back', 'left', 'right', 'top', 'bottom', 
                   'front_left', 'front_right', 'back_left', 'back_right']

# 视角名称映射（用于生成prompt）
VIEW_NAMES = {
    'front': 'front',
    'front_left': 'front-left',
    'left': 'left',
    'back_left': 'back-left',
    'back': 'back',
    'back_right': 'back-right',
    'right': 'right',
    'front_right': 'front-right',
    'top': 'top',
    'bottom': 'bottom',
}

# 高层视角偏移（相对于front，mod 24）
HIGH_LAYER_OFFSETS = {
    'front': 0,
    'front_right': 3,
    'right': 6,
    'back_right': 9,
    'back': 12,
    'back_left': 15,
    'left': 18,
    'front_left': 21,
}

# 随机种子
RANDOM_SEED = 42
# ======================


def get_frame_indices_for_view(front_frame: int, view: str) -> List[int]:
    """
    给定正面帧和视角，返回可能的帧索引列表（可能有高层和低层两个选择）
    
    Args:
        front_frame: 正面视角的帧号（0-23，高层）
        view: 视角名称
    
    Returns:
        可能的帧索引列表
    """
    if view == 'top':
        return [25]
    elif view == 'bottom':
        return [26]
    
    if view not in HIGH_LAYER_OFFSETS:
        return []
    
    # 高层帧计算
    high_offset = HIGH_LAYER_OFFSETS[view]
    high_frame = (front_frame + high_offset) % 24
    
    # 检查是否有对应的低层帧
    # 高层 +2n 对应低层 +n
    # 低层帧范围 27-39（共13帧，27和39相同）
    # 只有front, right, back, left有对应低层帧（即高层帧号为偶数时）
    
    candidates = [high_frame]
    
    total_high_offset = high_frame
    if total_high_offset % 2 == 0:  # 存在对应的低层帧
        total_low_offset = total_high_offset // 2
        low_frame = 27 + (total_low_offset % 12)
        assert low_frame <= 39, f"低层帧号超出范围: {low_frame}"
        candidates.append(low_frame)
    
    return candidates


def sample_frame_for_view(front_frame: int, view: str) -> int:
    """
    给定正面帧和视角，随机采样一个帧
    
    Args:
        front_frame: 正面视角的帧号（0-23，高层）
        view: 视角名称
    
    Returns:
        帧索引
    """
    candidates = get_frame_indices_for_view(front_frame, view)
    if not candidates:
        return front_frame  # 默认返回正面帧
    
    return random.choice(candidates)


def get_adjacent_views(view: str) -> List[str]:
    """
    获取指定视角的临近视角列表
    
    Args:
        view: 视角名称
    
    Returns:
        临近视角列表
    """
    adjacent_map = {
        'front': ['front_left', 'front_right'],
        'back': ['back_left', 'back_right'],
        'left': ['front_left', 'back_left'],
        'right': ['front_right', 'back_right'],
        'front_left': ['front', 'left'],
        'front_right': ['front', 'right'],
        'back_left': ['back', 'left'],
        'back_right': ['back', 'right'],
    }
    return adjacent_map.get(view, [])


def generate_prompt(input_views: List[str], output_view: str) -> str:
    """
    生成描述输入和输出视角的prompt
    """
    view_descriptions = []
    for i, view in enumerate(input_views, 1):
        view_name = VIEW_NAMES[view]
        view_descriptions.append(f"<image {i}> is the {view_name} view")
    
    output_name = VIEW_NAMES[output_view]
    prompt = f"For a fixed object, {', '.join(view_descriptions)}. From above, the sequence Front -> Left -> Back -> Right follows a clockwise order. Generate the {output_name} view."
    
    return prompt


def get_all_objects(source_dir: Path) -> List[Path]:
    """
    获取所有物体目录
    
    Args:
        source_dir: 源目录
    
    Returns:
        物体目录列表
    """
    objects = []
    if not source_dir.exists():
        return objects
    
    for category_dir in source_dir.iterdir():
        if category_dir.is_dir():
            for obj_dir in category_dir.iterdir():
                if obj_dir.is_dir() and (obj_dir / 'rgb').exists():
                    objects.append(obj_dir)
    
    return objects


def sample_one(
    obj_dir: Path,
    num_input_views: int,
    sampling_config: Optional[Dict] = None
) -> Optional[Dict]:
    """
    从一个物体目录采样一个样本
    
    Args:
        obj_dir: 物体目录路径
        num_input_views: 输入视角数量
    
    Returns:
        样本数据或None
    """
    rgb_dir = obj_dir / 'rgb'
    if not rgb_dir.exists():
        return None
    
    # 检查是否有足够的图像文件
    image_files = sorted(list(rgb_dir.glob('*.jpg')) + list(rgb_dir.glob('*.png')))
    if len(image_files) < 24:  # 至少需要24个高层帧
        return None
    
    # 获取配置参数
    if sampling_config is None:
        sampling_config = {}
    front_frame_range = sampling_config.get("front_frame_range", list(range(24)))
    view_constraint_mode = sampling_config.get("view_constraint_mode", 1)  # 默认使用约束方式1
    
    # 随机选择正面帧（front_frame_range 应该是一个列表，包含所有可选的 front frame 值，如 [0, 1, 2, ..., 23] 或 [0, 6, 12, 18]）
    if isinstance(front_frame_range, list) and len(front_frame_range) > 0:
        front_frame = random.choice(front_frame_range)
    else:
        # 默认使用 0-23
        front_frame = random.randint(0, 23)
    
    # 可用的视角方向
    available_views = [v for v in VIEW_DIRECTIONS if v != 'front']
    
    # 需要约束的视角（前后左右及其组合视角）
    constrained_views = ['front', 'back', 'left', 'right', 
                         'front_left', 'front_right', 'back_left', 'back_right']
    
    # 随机选择输出视角
    if len(available_views) < 1:
        return None
    
    # 尝试多次采样，确保满足约束条件
    max_sampling_attempts = 100
    input_views = None
    output_view = None
    
    for attempt in range(max_sampling_attempts):
        # 随机选择输出视角
        output_view = random.choice(available_views)
        
        # 根据约束方式选择不同的采样逻辑
        if view_constraint_mode == 1:
            # 约束方式1：如果output_view是前后左右等，需要包含临近视角
            if output_view in constrained_views:
                adjacent_views = get_adjacent_views(output_view)
                # 从可用视角中找出可用的临近视角（排除output_view本身）
                available_adjacent = [v for v in adjacent_views if v in available_views and v != output_view]
                
                # 如果没有可用的临近视角，跳过这个output_view
                if not available_adjacent:
                    continue
                
                # 剩余的可用视角（排除output_view）
                remaining_views = [v for v in available_views if v != output_view]
                
                # 如果剩余视角数量不足，跳过
                if len(remaining_views) < num_input_views:
                    continue
                
                # 确保至少选择一个临近视角
                # 从临近视角中至少选一个
                num_adjacent_to_include = random.randint(1, min(len(available_adjacent), num_input_views))
                selected_adjacent = random.sample(available_adjacent, num_adjacent_to_include)
                
                # 剩余的输入视角从其他可用视角中选择
                remaining_needed = num_input_views - num_adjacent_to_include
                if remaining_needed > 0:
                    other_views = [v for v in remaining_views if v not in selected_adjacent]
                    if len(other_views) < remaining_needed:
                        continue
                    selected_other = random.sample(other_views, remaining_needed)
                    input_views = selected_adjacent + selected_other
                else:
                    input_views = selected_adjacent
            else:
                # 输出视角不需要约束，正常选择
                remaining_views = [v for v in available_views if v != output_view]
                if len(remaining_views) < num_input_views:
                    continue
                input_views = random.sample(remaining_views, num_input_views)
        
        elif view_constraint_mode == 2:
            # 约束方式2：无论目标视角是什么，一定要有前后左右、前左、前右、后左、后右中的一个视角
            remaining_views = [v for v in available_views if v != output_view]
            
            # 检查是否有足够的视角
            if len(remaining_views) < num_input_views:
                continue
            
            # 从constrained_views中找出可用的视角（排除output_view本身）
            available_constrained = [v for v in constrained_views if v in available_views and v != output_view]
            
            # 如果没有可用的约束视角，跳过
            if not available_constrained:
                continue
            
            # 确保至少选择一个约束视角
            num_constrained_to_include = random.randint(1, min(len(available_constrained), num_input_views))
            selected_constrained = random.sample(available_constrained, num_constrained_to_include)
            
            # 剩余的输入视角从其他可用视角中选择
            remaining_needed = num_input_views - num_constrained_to_include
            if remaining_needed > 0:
                other_views = [v for v in remaining_views if v not in selected_constrained]
                if len(other_views) < remaining_needed:
                    continue
                selected_other = random.sample(other_views, remaining_needed)
                input_views = selected_constrained + selected_other
            else:
                input_views = selected_constrained
        
        elif view_constraint_mode == 3:
            # 约束方式3：无约束，正常随机选择
            remaining_views = [v for v in available_views if v != output_view]
            if len(remaining_views) < num_input_views:
                continue
            input_views = random.sample(remaining_views, num_input_views)
        
        else:
            # 未知的约束方式，使用默认方式（无约束）
            remaining_views = [v for v in available_views if v != output_view]
            if len(remaining_views) < num_input_views:
                continue
            input_views = random.sample(remaining_views, num_input_views)
        
        # 如果成功选择了视角，跳出循环
        if input_views is not None:
            break
    
    # 如果经过多次尝试仍无法满足约束，返回None
    if input_views is None or output_view is None:
        return None
    
    # 加载输入图像
    input_images = []
    
    for view in input_views:
        frame_idx = sample_frame_for_view(front_frame, view)
        # 尝试找到对应的图像文件（可能命名格式不同）
        frame_file = None
        for img_file in image_files:
            # 尝试从文件名中提取帧号
            try:
                frame_num = int(img_file.stem.split('_')[-1] or img_file.stem.split('.')[0])
                if frame_num == frame_idx:
                    frame_file = img_file
                    break
            except:
                continue
        
        if frame_file is None:
            # 如果找不到，尝试使用索引
            if frame_idx < len(image_files):
                frame_file = image_files[frame_idx]
            else:
                continue
        
        try:
            img = Image.open(frame_file)
            input_images.append(img)
        except Exception as e:
            print(f"加载图像失败 {frame_file}: {e}")
            continue
    
    if len(input_images) != num_input_views:
        return None
    
    # 加载输出图像
    output_frame_idx = sample_frame_for_view(front_frame, output_view)
    output_frame_file = None
    for img_file in image_files:
        try:
            frame_num = int(img_file.stem.split('_')[-1] or img_file.stem.split('.')[0])
            if frame_num == output_frame_idx:
                output_frame_file = img_file
                break
        except:
            continue
    
    if output_frame_file is None:
        if output_frame_idx < len(image_files):
            output_frame_file = image_files[output_frame_idx]
        else:
            return None
    
    try:
        output_image = Image.open(output_frame_file)
    except Exception as e:
        print(f"加载输出图像失败 {output_frame_file}: {e}")
        return None
    
    return {
        'obj_dir': obj_dir,
        'front_frame': front_frame,
        'input_images': input_images,
        'input_views': input_views,  # 返回原始视角键（下划线格式）
        'output_image': output_image,
        'output_view': output_view  # 返回原始视角键（下划线格式）
    }


def load_split_data(split_dir: Path, split_type: str, sub_type: str) -> List[Dict]:
    """
    从split目录加载数据
    
    Args:
        split_dir: split目录
        split_type: "train" 或 "eval"
        sub_type: 子类型（"object"）
    
    Returns:
        样本列表
    """
    json_file = split_dir / f"{sub_type}_{split_type}.json"
    
    if not json_file.exists():
        # 兼容旧格式：尝试读取txt文件
        txt_file = split_dir / f"{sub_type}_{split_type}.txt"
        if txt_file.exists():
            samples = []
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append({'obj_dir': Path(line)})
            return samples
        return []
    
    samples = []
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        sources = data.get('sources', [])
        for source in sources:
            samples.append({'obj_dir': Path(source)})
    
    return samples


def process_split_data(
    split_dir: Path,
    final_dir: Path,
    split_type: str,
    image_count_category: str,
    target_count: int,
    generated_ids: Set[str],
    sub_type: str = "object",
    sampling_config: Optional[Dict] = None
) -> None:
    """
    处理split数据并生成最终数据
    
    Args:
        split_dir: split目录
        final_dir: final目录
        split_type: "train" 或 "eval"
        image_count_category: 图像数量类别
        target_count: 目标生成数量
        generated_ids: 已生成的唯一ID集合
        sub_type: 子类型（"object"）
    """
    # 加载split数据
    samples = load_split_data(split_dir, split_type, sub_type)
    
    if not samples:
        print(f"未找到 {split_type}/{sub_type} 数据")
        return
    
    # 设置随机种子
    random.seed(RANDOM_SEED)
    
    # 处理样本
    completed = len(generated_ids)  # 从已生成数量开始计数
    current_idx = len(generated_ids)
    max_attempts = target_count * 10  # 最多尝试次数
    
    # 统计信息
    total_attempts = 0
    successful_attempts = 0
    
    with tqdm(
        total=target_count, 
        desc=f"{split_type}/{image_count_category}",
        unit="sample",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    ) as pbar:
        pbar.update(len(generated_ids))
        
        attempt = 0
        while completed < target_count and attempt < max_attempts:
            attempt += 1
            total_attempts += 1
            
            # 根据image_count_category为每个样本随机确定输入视角数量
            if image_count_category == "1-3":
                num_input_views = random.randint(1, 3)
            elif image_count_category == "4-5":
                num_input_views = random.randint(4, 5)
            elif image_count_category == "6-7":
                num_input_views = random.randint(6, 7)
            else:  # >=8
                num_input_views = random.randint(8, 9)  # 最多9个输入视角
            
            # 随机选择一个物体
            sample = random.choice(samples)
            obj_dir = sample['obj_dir']
            
            # 采样一个样本
            sampled = sample_one(obj_dir, num_input_views, sampling_config)
            if not sampled:
                continue
            
            # 生成唯一ID
            # 构建 view_config 字符串，包含视角配置信息（使用转换后的名称）
            input_view_names = [VIEW_NAMES[v] for v in sampled['input_views']]
            output_view_name = VIEW_NAMES[sampled['output_view']]
            view_config = f"f{sampled['front_frame']:02d}_i{','.join(sorted(input_view_names))}_o{output_view_name}"
            # 如果 SAVE_ORIGINAL_STRING=True，需要获取原始字符串以便保存
            from utils.common import SAVE_ORIGINAL_STRING
            unique_id_result = generate_unique_id(
                "spatial",
                return_original=SAVE_ORIGINAL_STRING,
                sub_type=sub_type,
                source=str(obj_dir),
                view_config=view_config
            )
            
            # 提取用于检查的唯一ID（如果是元组，使用 MD5 哈希；如果是字符串，直接使用）
            unique_id = unique_id_result[0] if isinstance(unique_id_result, tuple) else unique_id_result
            
            if unique_id in generated_ids:
                continue
            
            # 生成prompt（使用原始视角键）
            prompt = generate_prompt(sampled['input_views'], sampled['output_view'])
            
            # 准备图像文件
            image_files = {}
            for i, img in enumerate(sampled['input_images']):
                image_files[f"image_{i+1}.jpg"] = img
            image_files["image_output.jpg"] = sampled['output_image']
            
            # 构建预期的图像路径（save_sample_data会更新为实际路径）
            if sub_type:
                data_dir = final_dir / split_type / sub_type / image_count_category / "data" / f"{current_idx:08d}"
            else:
                data_dir = final_dir / split_type / image_count_category / "data" / f"{current_idx:08d}"
            
            input_image_paths = [str(data_dir / f"image_{i+1}.jpg") for i in range(len(sampled['input_images']))]
            output_image_path = str(data_dir / "image_output.jpg")
            
            # 构建JSON数据（使用转换后的名称）
            json_data = {
                'obj_dir': str(obj_dir),
                'front_frame': sampled['front_frame'],
                'num_input_views': num_input_views,
                'input_views': input_view_names,
                'output_view': output_view_name,
                'prompt': prompt,
                'input_images': input_image_paths,
                'output_image': output_image_path
            }
            
            # 保存数据（对于object，需要按子类型分类保存）
            from utils.common import save_sample_data
            # 对于object，路径为 final/spatial/{train/eval}/object/{image_count_category}/...
            # 注意：这里 sub_type 已经是 "object"，所以路径会自动包含它
            success = save_sample_data(
                final_dir,
                split_type,
                image_count_category,
                current_idx,
                unique_id_result,  # 可能是字符串或 (md5_hash, original_string) 元组
                json_data,
                image_files,
                sub_type=sub_type  # 传递子类型以支持分类保存
            )
            
            if success:
                generated_ids.add(unique_id)
                completed += 1
                successful_attempts += 1
                current_idx += 1
                pbar.update(1)
                
                # 更新进度条描述，显示详细统计信息
                success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
                pbar.set_description(
                    f"{split_type}/{image_count_category} | "
                    f"完成:{completed}/{target_count} | "
                    f"尝试:{total_attempts} | "
                    f"成功率:{success_rate:.1f}%"
                )
    
    print(f"\n{split_type}/{image_count_category} 完成: {completed}/{target_count}")


def main():
    """主函数"""
    print("=" * 80)
    print("Spatial Object数据生成脚本")
    print("=" * 80)
    print(f"Split目录: {SPLIT_DIR}")
    print(f"Final目录: {FINAL_DIR}")
    print(f"生成配置: {GEN_CONFIG}")
    print("=" * 80)
    
    # 创建final目录
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 处理train和eval数据
    for split_type in ["train", "eval"]:
        print(f"\n处理 {split_type} 数据...")
        
        for image_count_category, config in GEN_CONFIG.items():
            target_count = config.get(split_type, 0)
            
            if target_count <= 0:
                print(f"跳过 {split_type}/{image_count_category} 数据生成（目标数量为0）")
                continue
            
            # 加载已生成的唯一识别编号
            generated_ids = load_generated_ids(FINAL_DIR, split_type, image_count_category)
            print(f"已加载 {len(generated_ids)} 个已生成的样本ID")
            
            # 处理数据
            process_split_data(
                split_dir=SPLIT_DIR,
                final_dir=FINAL_DIR,
                split_type=split_type,
                image_count_category=image_count_category,
                target_count=target_count,
                generated_ids=generated_ids,
                sub_type="object"
            )
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()

