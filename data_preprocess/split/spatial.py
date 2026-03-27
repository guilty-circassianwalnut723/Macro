#!/usr/bin/env python3
"""
Spatial data split script: split object/indoor/outdoor data into train and eval

Features:
1. For object: split by object directory, ensuring samples from the same object are not in both train and eval
2. For indoor/outdoor: split by source panoramic image, ensuring samples from the same image are not in both train and eval
3. Generate train and eval list files, save to data_hl02/split/spatial directory
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# ====== Configuration parameters ======
# DATA_ROOT: root directory of raw data (adjust to your data directory)
_SCRIPT_DIR = Path(__file__).resolve().parent.parent  # data_preprocess/
DATA_ROOT = _SCRIPT_DIR / "data"  # data_preprocess/data/
SOURCE_DIR = DATA_ROOT / 'source'
OUTPUT_DIR = DATA_ROOT / 'split' / 'spatial'

# Eval count config: {sub_type: eval_count}
# sub_type can be 'object', 'indoor', 'outdoor'
EVAL_COUNTS = {
    'object': 500,   # eval object count for object category
    'indoor': 500,   # eval panoramic image count for indoor category
    'outdoor': 500,  # eval panoramic image count for outdoor category
}

# Random seed
RANDOM_SEED = 42
# ======================


def get_object_dirs(source_dir: Path) -> list:
    """Get all object directories"""
    objects = []
    for category_dir in source_dir.iterdir():
        if category_dir.is_dir():
            for obj_dir in category_dir.iterdir():
                if obj_dir.is_dir() and (obj_dir / 'rgb').exists():
                    objects.append(str(obj_dir))
    return objects


def get_panorama_images(source_dir: Path) -> list:
    """Get all panoramic image paths"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    images = []
    for ext in image_extensions:
        images.extend([str(p) for p in source_dir.rglob(f'*{ext}')])
    return images


def split_sources(sources: list, eval_count: int, seed: int = 42) -> tuple:
    """Split source data into train and eval"""
    if eval_count is None or eval_count <= 0:
        return sources, []
    
    if len(sources) <= eval_count:
        print(f"Warning: total source count {len(sources)} is <= eval count {eval_count}, all data used for eval")
        return [], sources
    
    random.seed(seed)
    shuffled = sources.copy()
    random.shuffle(shuffled)
    
    eval_sources = shuffled[:eval_count]
    train_sources = shuffled[eval_count:]
    
    return train_sources, eval_sources


def save_json(data: dict, output_path: Path):
    """Save data to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def check_split_exists(output_dir: Path, category: str) -> bool:
    """Check whether both train and eval files exist for the specified category"""
    train_path = output_dir / f'{category}_train.json'
    eval_path = output_dir / f'{category}_eval.json'
    return train_path.exists() and eval_path.exists()


def main():
    """Main function"""
    print("=" * 80)
    print("Spatial data split script")
    print("=" * 80)
    print(f"Data root directory: {DATA_ROOT}")
    print(f"Random seed: {RANDOM_SEED}")
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    statistics = {}
    
    # 1. Process object
    print("=" * 80)
    print(f"Processing object (eval object count: {EVAL_COUNTS.get('object', 0)})")
    print("=" * 80)
    
    if check_split_exists(OUTPUT_DIR, 'object'):
        print(f"Skipping: train and eval files for object already exist")
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
        print(f"Existing - Train object count: {len(train_objects)}")
        print(f"Existing - Eval object count: {len(eval_objects)}")
    else:
        object_source_dir = SOURCE_DIR / 'object'
        if object_source_dir.exists():
            object_dirs = get_object_dirs(object_source_dir)
            print(f"Found {len(object_dirs)} object directories")
            
            train_objects, eval_objects = split_sources(
                object_dirs, EVAL_COUNTS.get('object', 0), RANDOM_SEED
            )
            
            save_json({'sources': train_objects}, OUTPUT_DIR / 'object_train.json')
            save_json({'sources': eval_objects}, OUTPUT_DIR / 'object_eval.json')
            
            print(f"Train object count: {len(train_objects)}")
            print(f"Eval object count: {len(eval_objects)}")
            
            statistics['object'] = {
                'total': len(object_dirs),
                'train': len(train_objects),
                'eval': len(eval_objects)
            }
        else:
            print(f"Warning: object source directory not found: {object_source_dir}")
    
    # 2. Process indoor
    print("\n" + "=" * 80)
    print(f"Processing indoor (eval panoramic image count: {EVAL_COUNTS.get('indoor', 0)})")
    print("=" * 80)
    
    if check_split_exists(OUTPUT_DIR, 'indoor'):
        print(f"Skipping: train and eval files for indoor already exist")
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
        print(f"Existing - Train panoramic image count: {len(train_indoor)}")
        print(f"Existing - Eval panoramic image count: {len(eval_indoor)}")
    else:
        indoor_source_dir = SOURCE_DIR / 'panorama' / 'indoor'
        if indoor_source_dir.exists():
            indoor_images = get_panorama_images(indoor_source_dir)
            print(f"Found {len(indoor_images)} panoramic images")
            
            train_indoor, eval_indoor = split_sources(
                indoor_images, EVAL_COUNTS.get('indoor', 0), RANDOM_SEED
            )
            
            save_json({'sources': train_indoor}, OUTPUT_DIR / 'indoor_train.json')
            save_json({'sources': eval_indoor}, OUTPUT_DIR / 'indoor_eval.json')
            
            print(f"Train panoramic image count: {len(train_indoor)}")
            print(f"Eval panoramic image count: {len(eval_indoor)}")
            
            statistics['indoor'] = {
                'total': len(indoor_images),
                'train': len(train_indoor),
                'eval': len(eval_indoor)
            }
        else:
            print(f"Warning: indoor source directory not found: {indoor_source_dir}")
    
    # 3. Process outdoor
    print("\n" + "=" * 80)
    print(f"Processing outdoor (eval panoramic image count: {EVAL_COUNTS.get('outdoor', 0)})")
    print("=" * 80)
    
    if check_split_exists(OUTPUT_DIR, 'outdoor'):
        print(f"Skipping: train and eval files for outdoor already exist")
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
        print(f"Existing - Train panoramic image count: {len(train_outdoor)}")
        print(f"Existing - Eval panoramic image count: {len(eval_outdoor)}")
    else:
        outdoor_source_dir = SOURCE_DIR / 'panorama' / 'outdoor'
        if outdoor_source_dir.exists():
            outdoor_images = get_panorama_images(outdoor_source_dir)
            print(f"Found {len(outdoor_images)} panoramic images")
            
            train_outdoor, eval_outdoor = split_sources(
                outdoor_images, EVAL_COUNTS.get('outdoor', 0), RANDOM_SEED
            )
            
            save_json({'sources': train_outdoor}, OUTPUT_DIR / 'outdoor_train.json')
            save_json({'sources': eval_outdoor}, OUTPUT_DIR / 'outdoor_eval.json')
            
            print(f"Train panoramic image count: {len(train_outdoor)}")
            print(f"Eval panoramic image count: {len(eval_outdoor)}")
            
            statistics['outdoor'] = {
                'total': len(outdoor_images),
                'train': len(train_outdoor),
                'eval': len(eval_outdoor)
            }
        else:
            print(f"Warning: outdoor source directory not found: {outdoor_source_dir}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Split statistics")
    print("=" * 80)
    
    total_all = 0
    total_train = 0
    total_eval = 0
    
    for category, stats in statistics.items():
        print(f"\nCategory: {category}")
        print(f"  Total count: {stats['total']}")
        print(f"  Train count: {stats['train']}")
        print(f"  Eval count: {stats['eval']}")
        total_all += stats['total']
        total_train += stats['train']
        total_eval += stats['eval']
    
    print("\n" + "=" * 80)
    print(f"Total source data count: {total_all}")
    print(f"Total Train source data count: {total_train}")
    print(f"Total Eval source data count: {total_eval}")
    print("=" * 80)
    
    # Save statistics
    stats_path = OUTPUT_DIR / 'statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    print(f"\nStatistics saved to: {stats_path}")
    
    print("\nSplit complete!")
    print(f"Split files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

