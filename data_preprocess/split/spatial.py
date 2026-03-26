#!/usr/bin/env python3
"""
Spatial数据拆分脚本：将object/indoor/outdoor数据拆分为train和eval

功能：
1. 对于object：按物体目录拆分，确保同一物体的样本不会同时出现在train和eval中
2. 对于indoor/outdoor：按源全景图拆分，确保同一全景图的样本不会同时出现在train和eval中
3. 生成train和eval列表文件，保存到data_hl02/split/spatial目录
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# ====== 配置参数 ======
# DATA_ROOT: root directory of raw data (adjust to your data directory)
_SCRIPT_DIR = Path(__file__).resolve().parent.parent  # data_preprocess/
DATA_ROOT = _SCRIPT_DIR / "data"  # data_preprocess/data/
SOURCE_DIR = DATA_ROOT / 'source'
OUTPUT_DIR = DATA_ROOT / 'split' / 'spatial'

# Eval数量配置：{sub_type: eval_count}
# sub_type可以是 'object', 'indoor', 'outdoor'
EVAL_COUNTS = {
    'object': 500,   # object类别的eval物体数量
    'indoor': 500,   # indoor类别的eval全景图数量
    'outdoor': 500,  # outdoor类别的eval全景图数量
}

# 随机种子
RANDOM_SEED = 42
# ======================


def get_object_dirs(source_dir: Path) -> list:
    """获取所有物体目录"""
    objects = []
    for category_dir in source_dir.iterdir():
        if category_dir.is_dir():
            for obj_dir in category_dir.iterdir():
                if obj_dir.is_dir() and (obj_dir / 'rgb').exists():
                    objects.append(str(obj_dir))
    return objects


def get_panorama_images(source_dir: Path) -> list:
    """获取所有全景图像路径"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    images = []
    for ext in image_extensions:
        images.extend([str(p) for p in source_dir.rglob(f'*{ext}')])
    return images


def split_sources(sources: list, eval_count: int, seed: int = 42) -> tuple:
    """将源数据拆分为train和eval"""
    if eval_count is None or eval_count <= 0:
        return sources, []
    
    if len(sources) <= eval_count:
        print(f"警告: 源数据总数 {len(sources)} 小于等于eval数量 {eval_count}，所有数据用于eval")
        return [], sources
    
    random.seed(seed)
    shuffled = sources.copy()
    random.shuffle(shuffled)
    
    eval_sources = shuffled[:eval_count]
    train_sources = shuffled[eval_count:]
    
    return train_sources, eval_sources


def save_json(data: dict, output_path: Path):
    """保存数据到JSON文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def check_split_exists(output_dir: Path, category: str) -> bool:
    """检查指定类别的train和eval文件是否都已存在"""
    train_path = output_dir / f'{category}_train.json'
    eval_path = output_dir / f'{category}_eval.json'
    return train_path.exists() and eval_path.exists()


def main():
    """主函数"""
    print("=" * 80)
    print("Spatial数据拆分脚本")
    print("=" * 80)
    print(f"数据根目录: {DATA_ROOT}")
    print(f"随机种子: {RANDOM_SEED}")
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    statistics = {}
    
    # 1. 处理object
    print("=" * 80)
    print(f"处理 object (eval物体数: {EVAL_COUNTS.get('object', 0)})")
    print("=" * 80)
    
    if check_split_exists(OUTPUT_DIR, 'object'):
        print(f"跳过: object 的 train 和 eval 文件已存在")
        train_path = OUTPUT_DIR / 'object_train.json'
        eval_path = OUTPUT_DIR / 'object_eval.json'
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            train_objects = train_data.get('sources', [])
        with open(eval_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
            eval_objects = eval_data.get('sources', [])
        statistics['object'] = {
            'total': len(train_objects) + len(eval_objects),
            'train': len(train_objects),
            'eval': len(eval_objects)
        }
        print(f"已存在 - Train物体数: {len(train_objects)}")
        print(f"已存在 - Eval物体数: {len(eval_objects)}")
    else:
        object_source_dir = SOURCE_DIR / 'object'
        if object_source_dir.exists():
            object_dirs = get_object_dirs(object_source_dir)
            print(f"找到 {len(object_dirs)} 个物体目录")
            
            train_objects, eval_objects = split_sources(
                object_dirs, EVAL_COUNTS.get('object', 0), RANDOM_SEED
            )
            
            save_json({'sources': train_objects}, OUTPUT_DIR / 'object_train.json')
            save_json({'sources': eval_objects}, OUTPUT_DIR / 'object_eval.json')
            
            print(f"Train物体数: {len(train_objects)}")
            print(f"Eval物体数: {len(eval_objects)}")
            
            statistics['object'] = {
                'total': len(object_dirs),
                'train': len(train_objects),
                'eval': len(eval_objects)
            }
        else:
            print(f"警告: object源目录不存在: {object_source_dir}")
    
    # 2. 处理indoor
    print("\n" + "=" * 80)
    print(f"处理 indoor (eval全景图数: {EVAL_COUNTS.get('indoor', 0)})")
    print("=" * 80)
    
    if check_split_exists(OUTPUT_DIR, 'indoor'):
        print(f"跳过: indoor 的 train 和 eval 文件已存在")
        train_path = OUTPUT_DIR / 'indoor_train.json'
        eval_path = OUTPUT_DIR / 'indoor_eval.json'
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            train_indoor = train_data.get('sources', [])
        with open(eval_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
            eval_indoor = eval_data.get('sources', [])
        statistics['indoor'] = {
            'total': len(train_indoor) + len(eval_indoor),
            'train': len(train_indoor),
            'eval': len(eval_indoor)
        }
        print(f"已存在 - Train全景图数: {len(train_indoor)}")
        print(f"已存在 - Eval全景图数: {len(eval_indoor)}")
    else:
        indoor_source_dir = SOURCE_DIR / 'panorama' / 'indoor'
        if indoor_source_dir.exists():
            indoor_images = get_panorama_images(indoor_source_dir)
            print(f"找到 {len(indoor_images)} 张全景图")
            
            train_indoor, eval_indoor = split_sources(
                indoor_images, EVAL_COUNTS.get('indoor', 0), RANDOM_SEED
            )
            
            save_json({'sources': train_indoor}, OUTPUT_DIR / 'indoor_train.json')
            save_json({'sources': eval_indoor}, OUTPUT_DIR / 'indoor_eval.json')
            
            print(f"Train全景图数: {len(train_indoor)}")
            print(f"Eval全景图数: {len(eval_indoor)}")
            
            statistics['indoor'] = {
                'total': len(indoor_images),
                'train': len(train_indoor),
                'eval': len(eval_indoor)
            }
        else:
            print(f"警告: indoor源目录不存在: {indoor_source_dir}")
    
    # 3. 处理outdoor
    print("\n" + "=" * 80)
    print(f"处理 outdoor (eval全景图数: {EVAL_COUNTS.get('outdoor', 0)})")
    print("=" * 80)
    
    if check_split_exists(OUTPUT_DIR, 'outdoor'):
        print(f"跳过: outdoor 的 train 和 eval 文件已存在")
        train_path = OUTPUT_DIR / 'outdoor_train.json'
        eval_path = OUTPUT_DIR / 'outdoor_eval.json'
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            train_outdoor = train_data.get('sources', [])
        with open(eval_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
            eval_outdoor = eval_data.get('sources', [])
        statistics['outdoor'] = {
            'total': len(train_outdoor) + len(eval_outdoor),
            'train': len(train_outdoor),
            'eval': len(eval_outdoor)
        }
        print(f"已存在 - Train全景图数: {len(train_outdoor)}")
        print(f"已存在 - Eval全景图数: {len(eval_outdoor)}")
    else:
        outdoor_source_dir = SOURCE_DIR / 'panorama' / 'outdoor'
        if outdoor_source_dir.exists():
            outdoor_images = get_panorama_images(outdoor_source_dir)
            print(f"找到 {len(outdoor_images)} 张全景图")
            
            train_outdoor, eval_outdoor = split_sources(
                outdoor_images, EVAL_COUNTS.get('outdoor', 0), RANDOM_SEED
            )
            
            save_json({'sources': train_outdoor}, OUTPUT_DIR / 'outdoor_train.json')
            save_json({'sources': eval_outdoor}, OUTPUT_DIR / 'outdoor_eval.json')
            
            print(f"Train全景图数: {len(train_outdoor)}")
            print(f"Eval全景图数: {len(eval_outdoor)}")
            
            statistics['outdoor'] = {
                'total': len(outdoor_images),
                'train': len(train_outdoor),
                'eval': len(eval_outdoor)
            }
        else:
            print(f"警告: outdoor源目录不存在: {outdoor_source_dir}")
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("拆分统计信息")
    print("=" * 80)
    
    total_all = 0
    total_train = 0
    total_eval = 0
    
    for category, stats in statistics.items():
        print(f"\n类别: {category}")
        print(f"  总数量: {stats['total']}")
        print(f"  Train数量: {stats['train']}")
        print(f"  Eval数量: {stats['eval']}")
        total_all += stats['total']
        total_train += stats['train']
        total_eval += stats['eval']
    
    print("\n" + "=" * 80)
    print(f"总计源数据数: {total_all}")
    print(f"总计Train源数据数: {total_train}")
    print(f"总计Eval源数据数: {total_eval}")
    print("=" * 80)
    
    # 保存统计信息
    stats_path = OUTPUT_DIR / 'statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    print(f"\n统计信息已保存到: {stats_path}")
    
    print("\n拆分完成！")
    print(f"分割文件已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

