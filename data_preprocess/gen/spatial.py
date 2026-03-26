#!/usr/bin/env python3
"""
Spatial数据生成统一脚本

功能：
1. 统一控制outdoor、indoor、object三个子类型的生成配置和数量
2. 从split/spatial目录读取train/eval数据
3. 保存到final/spatial/{train/eval}/{image_count_category}/data和json目录
4. 支持唯一识别编号，避免重复生成
"""

import json
import sys
from pathlib import Path

# 添加utils路径
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

# ====== 配置参数 ======
SPLIT_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "split" / "spatial")
FINAL_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "final" / "spatial")

# 子类型选择：outdoor、indoor或object
# 设置为 None 或 [] 表示处理所有子类型，设置为具体子类型字符串则只处理该子类型
SUB_TYPE = None  # 修改此值来选择要处理的子类型，None表示处理所有子类型

# 生成配置：{sub_type: {image_count_category: {train: count, eval: count}}}
GEN_CONFIG = {
    "object": {
        "1-3": {"train": 12000, "eval": 90},
        "4-5": {"train": 12000, "eval": 90},
        "6-7": {"train": 12000, "eval": 90},
        ">=8": {"train": 12000, "eval": 90},
    },
    "outdoor": {
        "1-3": {"train": 9000, "eval": 80},
        "4-5": {"train": 9000, "eval": 80},
        "6-7": {"train": 9000, "eval": 80},
        ">=8": {"train": 9000, "eval": 80},
    },
    "indoor": {
        "1-3": {"train": 9000, "eval": 80},
        "4-5": {"train": 9000, "eval": 80},
        "6-7": {"train": 9000, "eval": 80},
        ">=8": {"train": 9000, "eval": 80},
    },
}

# 采样参数配置：{sub_type: {param_name: value}}
# 参考 runner/spatial/gen 目录下的脚本参数
SAMPLING_CONFIG = {
    "outdoor": {
        "min_overlap": 0.3,      # 最小重叠比例
        "max_overlap": 0.8,      # 最大重叠比例
        "min_fov": 90.0,         # 最小视野角度（度）
        "max_fov": 90.0,         # 最大视野角度（度）
        "image_size": [1024, 1024],  # 输出图像尺寸 [height, width]
        "add_noise": False,      # 是否添加视角扰动
        "noise_scale": 10.0,     # 视角扰动幅度（度）
        "base_pitch_range": [-10, 10],  # base_pitch的随机范围（度），使用 random.uniform(base_pitch_range[0], base_pitch_range[1])
    },
    "indoor": {
        "min_overlap": 0.3,      # 最小重叠比例
        "max_overlap": 0.8,      # 最大重叠比例
        "min_fov": 90.0,         # 最小视野角度（度）
        "max_fov": 90.0,         # 最大视野角度（度）
        "image_size": [1024, 1024],  # 输出图像尺寸 [height, width]
        "add_noise": False,      # 是否添加视角扰动
        "noise_scale": 10.0,     # 视角扰动幅度（度）
        "base_pitch_range": [-10, 10],  # base_pitch的随机范围（度），使用 random.uniform(base_pitch_range[0], base_pitch_range[1])
    },
    "object": {
        "front_frame_range": list(range(24)),  # front_frame的可选值列表，所有可选的 front frame 值（0-23）
        "view_constraint_mode": 1,  # 视角约束方式：1=当前约束（如果output_view是前后左右等，需要包含临近视角），2=必须包含前后左右等视角之一，3=无约束
    },
}
# ======================


def main():
    """主函数"""
    # 确定要处理的子类型列表
    if SUB_TYPE is None:
        # 处理所有子类型
        sub_types_to_process = list(GEN_CONFIG.keys())
        print("=" * 80)
        print("Spatial 数据生成脚本 - 处理所有子类型")
        print("=" * 80)
    else:
        # 只处理指定的子类型
        if SUB_TYPE not in GEN_CONFIG:
            raise ValueError(f"不支持的子类型: {SUB_TYPE}，支持的类型: {list(GEN_CONFIG.keys())}")
        sub_types_to_process = [SUB_TYPE]
        print("=" * 80)
        print(f"Spatial {SUB_TYPE.upper()}数据生成脚本")
        print("=" * 80)
    
    print(f"Split目录: {SPLIT_DIR}")
    print(f"Final目录: {FINAL_DIR}")
    print(f"将处理的子类型: {sub_types_to_process}")
    print("=" * 80)
    
    # 创建final目录
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 遍历处理每个子类型
    for current_sub_type in sub_types_to_process:
        print(f"\n{'=' * 80}")
        print(f"开始处理子类型: {current_sub_type.upper()}")
        print(f"{'=' * 80}")
        
        gen_config = GEN_CONFIG[current_sub_type]
        sampling_config = SAMPLING_CONFIG.get(current_sub_type, {})
        
        print(f"生成配置: {gen_config}")
        print(f"采样配置: {sampling_config}")
        
        # 导入对应的处理模块
        # 注意：spatial.py在gen目录下，outdoor/indoor/object.py在gen/spatial目录下
        if current_sub_type == "outdoor":
            from spatial.outdoor import process_split_data
            from utils.common import load_generated_ids
        elif current_sub_type == "indoor":
            from spatial.indoor import process_split_data
            from utils.common import load_generated_ids
        elif current_sub_type == "object":
            from spatial.object import process_split_data
            from utils.common import load_generated_ids
        else:
            raise ValueError(f"不支持的子类型: {current_sub_type}")
        
        # 处理train和eval数据
        for split_type in ["train", "eval"]:
            print(f"\n处理 {current_sub_type}/{split_type} 数据...")
            
            for image_count_category, config in gen_config.items():
                target_count = config.get(split_type, 0)
                
                if target_count <= 0:
                    print(f"跳过 {current_sub_type}/{split_type}/{image_count_category} 数据生成（目标数量为0）")
                    continue
                
                # 加载已生成的唯一识别编号（对于spatial，需要传递sub_type）
                generated_ids = load_generated_ids(FINAL_DIR, split_type, image_count_category, sub_type=current_sub_type)
                print(f"已加载 {len(generated_ids)} 个已生成的样本ID")
                
                # 处理数据
                process_split_data(
                    split_dir=SPLIT_DIR,
                    final_dir=FINAL_DIR,
                    split_type=split_type,
                    image_count_category=image_count_category,
                    target_count=target_count,
                    generated_ids=generated_ids,
                    sub_type=current_sub_type,
                    sampling_config=sampling_config
                )
        
        print(f"\n子类型 {current_sub_type.upper()} 处理完成！")
    
    print(f"\n{'=' * 80}")
    print("所有子类型处理完成！")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

