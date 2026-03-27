#!/usr/bin/env python3
"""
Customization data split script: split each category's data into train and eval

Features:
1. Read jsonl files for each category from processed/customization directory
2. Split data into train and eval based on configured eval sample count
3. Special handling for the scene category: 100 multi-frame scenes + 100 single-frame scenes
4. Support extending eval from existing data: if needed eval count exceeds existing, add unused sources from final_old
5. Save to data_hl02/split/customization directory, format is json (using list)

Notes:
- Users only need to specify eval count; train count is automatically (total - eval count)
- The actual data volume is determined by gen; split only handles the division
"""

import json
import random
import hashlib
from pathlib import Path

MACRO_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = MACRO_DIR / "data"
from tqdm import tqdm
from typing import Set, Dict, List

# ====== Configuration parameters ======
DATA_DIR = DATA_DIR / "processed" / "customization"
OUTPUT_DIR = DATA_DIR / "split" / "customization"
FINAL_DIR = DATA_DIR / "final" / "customization"
FINAL_OLD_DIR = DATA_DIR / "final_old" / "customization"

# Configure eval sample count per category (None means no split, all for train)
# Note: users only need to specify eval count; train count is automatically (total - eval count)
EVAL_COUNTS = {
    'human': 500,
    'cloth': 300,
    'object': 500,
    'scene': 200,  # 200 scenes: 100 multi-frame + 100 single-frame
    'style': 300,  # Expanded from 200 to 300
}

# Random seed
RANDOM_SEED = 42
# ======================


def load_jsonl(jsonl_path: Path) -> list:
    """Load jsonl file"""
    samples = []
    if not jsonl_path.exists():
        print(f"Warning: file not found: {jsonl_path}")
        return samples
    
    print(f"Loading {jsonl_path}...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                samples.append(data)
            except json.JSONDecodeError as e:
                print(f"Warning: cannot parse JSON line: {e}")
                continue
    
    print(f"Loaded {len(samples)} samples")
    return samples


def split_samples(samples: list, eval_count: int, seed: int = 42) -> tuple:
    """Split samples into train and eval"""
    if eval_count is None or eval_count <= 0:
        return samples, []
    
    if len(samples) <= eval_count:
        print(f"Warning: total sample count {len(samples)} is <= eval count {eval_count}, all samples used for eval")
        return [], samples
    
    random.seed(seed)
    shuffled_samples = samples.copy()
    random.shuffle(shuffled_samples)
    
    eval_samples = shuffled_samples[:eval_count]
    train_samples = shuffled_samples[eval_count:]
    
    return train_samples, eval_samples


def split_scene_samples(samples: list, eval_scene_count: int, seed: int = 42) -> tuple:
    """
    Special handling for scene samples: split into train and eval
    Identify multi-frame and single-frame scenes from filenames:
    - Multi-frame scenes: filename format {scene_idx}_{frame_idx}.jpg, e.g. 00000001_1.jpg
    - Single-frame scenes: filename format {scene_idx}.jpg, e.g. 00006379.jpg
    
    - 100 multi-frame scenes, all frames added to eval (~500 samples)
    - 100 single-frame scenes, 1 frame each (100 samples)
    - Total ~600 eval samples
    """
    if eval_scene_count is None or eval_scene_count <= 0:
        return samples, []
    
    random.seed(seed)
    
    # Identify multi-frame and single-frame scenes from filenames
    multi_frame_groups = {}  # key: scene_idx, value: list of samples (all frames of this scene)
    single_frame_samples = []  # List of single-frame scene samples
    
    for sample in samples:
        filename = sample.get('filename', '')
        if not filename:
            filepath = sample.get('filepath', '')
            if filepath:
                filename = Path(filepath).name
            else:
                single_frame_samples.append(sample)
                continue
        
        # Check if it is a multi-frame scene: filename format {scene_idx}_{frame_idx}.jpg
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
        
        # Single-frame scene
        single_frame_samples.append(sample)
    
    print(f"Identified {len(multi_frame_groups)} multi-frame scenes")
    print(f"Identified {len(single_frame_samples)} single-frame scenes")
    
    # Need 100 multi-frame scenes + 100 single-frame scenes
    multi_frame_count = 100
    single_frame_count = 100
    
    # Shuffle multi-frame scene groups
    multi_frame_scene_indices = list(multi_frame_groups.keys())
    random.shuffle(multi_frame_scene_indices)
    
    # Shuffle single-frame scenes
    random.shuffle(single_frame_samples)
    
    # Select multi-frame scenes (first multi_frame_count)
    selected_multi_frame_indices = multi_frame_scene_indices[:multi_frame_count]
    
    # Select single-frame scenes (first single_frame_count)
    selected_single_frame_samples = single_frame_samples[:single_frame_count]
    
    # Collect eval samples
    eval_samples = []
    
    # Add all frames of multi-frame scenes
    for scene_idx in selected_multi_frame_indices:
        frames = multi_frame_groups[scene_idx]
        eval_samples.extend(frames)
        print(f"Scene {scene_idx}: {len(frames)} frames")
    
    # Add single-frame scenes
    eval_samples.extend(selected_single_frame_samples)
    
    # Collect train samples
    train_samples = []
    
    # Remaining multi-frame scenes
    remaining_multi_frame_indices = multi_frame_scene_indices[multi_frame_count:]
    for scene_idx in remaining_multi_frame_indices:
        train_samples.extend(multi_frame_groups[scene_idx])
    
    # Remaining single-frame scenes
    remaining_single_frame_samples = single_frame_samples[single_frame_count:]
    train_samples.extend(remaining_single_frame_samples)
    
    print(f"Multi-frame scene count: {len(selected_multi_frame_indices)} (total {sum(len(multi_frame_groups[idx]) for idx in selected_multi_frame_indices)} samples)")
    print(f"Single-frame scene count: {len(selected_single_frame_samples)} (total {len(selected_single_frame_samples)} samples)")
    print(f"Total Eval samples: {len(eval_samples)}")
    print(f"Total Train samples: {len(train_samples)}")
    
    return train_samples, eval_samples


def get_combination_key(files: List[str]) -> str:
    """Generate a unique key for an image combination (for deduplication)"""
    sorted_files = sorted(files)
    key_str = "|".join(sorted_files)
    return hashlib.md5(key_str.encode()).hexdigest()


def load_existing_eval_data(final_old_dir: Path, class_name: str) -> Dict[str, dict]:
    """
    Load existing eval data from the final_old directory
    
    Args:
        final_old_dir: final_old directory path
        class_name: category name
    
    Returns:
        {combination_key: sample_data} dict
    """
    existing_eval = {}
    
    if not final_old_dir.exists():
        return existing_eval
    
    # Iterate over all image_count_category directories
    for category_dir in final_old_dir.glob("*"):
        if not category_dir.is_dir():
            continue
        
        eval_json_dir = category_dir / "eval" / "json"
        if not eval_json_dir.exists():
            continue
        
        # Read all JSON files
        for json_file in eval_json_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Check whether it belongs to the current category
                    sample_class = data.get('class', '')
                    if sample_class != class_name:
                        continue
                    
                    # Extract combination_key
                    input_images = data.get('input_images', [])
                    if input_images:
                        # Extract filenames as combination key
                        files = [Path(img).name for img in input_images]
                        combination_key = get_combination_key(files)
                        existing_eval[combination_key] = data
            except Exception as e:
                print(f"Warning: failed to read JSON file {json_file}: {e}")
                continue
    
    return existing_eval


def load_processed_samples_by_combination(processed_dir: Path, class_name: str) -> Dict[str, dict]:
    """
    Load samples from the processed directory, indexed by combination_key
    
    Args:
        processed_dir: processed directory path
        class_name: category name
    
    Returns:
        {combination_key: sample_data} dict
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
                # Extract file list
                files = []
                if 'files' in data:
                    files = [f if isinstance(f, str) else f.get('filepath', '') for f in data['files']]
                elif 'filepath' in data:
                    files = [data['filepath']]
                
                if files:
                    # Extract filenames
                    file_names = [Path(f).name for f in files]
                    combination_key = get_combination_key(file_names)
                    samples_by_key[combination_key] = data
            except Exception as e:
                print(f"Warning: failed to parse JSON line: {e}")
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
    Extend eval samples from existing data
    
    If the needed eval count exceeds existing, add unused sources from final_old
    
    Args:
        eval_samples: current eval sample list
        target_eval_count: target eval count
        processed_dir: processed directory path
        final_old_dir: final_old directory path
        class_name: category name
        seed: random seed
    
    Returns:
        extended eval sample list
    """
    current_eval_count = len(eval_samples)
    
    if current_eval_count >= target_eval_count:
        return eval_samples
    
    print(f"\nNeed to extend eval: current {current_eval_count}, target {target_eval_count}")
    print(f"Looking for available sources from existing data...")
    
    # Load existing eval data (from final_old)
    existing_eval = load_existing_eval_data(final_old_dir, class_name)
    print(f"Loaded {len(existing_eval)} existing eval samples from final_old")
    
    # Load all samples from processed, indexed by combination_key
    processed_samples = load_processed_samples_by_combination(processed_dir, class_name)
    print(f"Loaded {len(processed_samples)} samples from processed")
    
    # Get the combination_key set of current eval samples
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
    
    # Find unused samples from existing eval
    available_eval_keys = []
    for combo_key, eval_data in existing_eval.items():
        if combo_key not in current_eval_keys:
            # Check whether a corresponding source exists in processed
            if combo_key in processed_samples:
                available_eval_keys.append(combo_key)
    
    print(f"Found {len(available_eval_keys)} available existing eval samples")
    
    # Randomly select samples to supplement
    need_count = target_eval_count - current_eval_count
    if need_count > 0 and available_eval_keys:
        random.seed(seed)
        selected_keys = random.sample(available_eval_keys, min(need_count, len(available_eval_keys)))
        
        # Get corresponding sample data from processed
        for combo_key in selected_keys:
            if combo_key in processed_samples:
                eval_samples.append(processed_samples[combo_key])
        
        print(f"Added {len(selected_keys)} samples to eval")
    
    return eval_samples


def save_json(samples: list, output_path: Path):
    """Save samples to json file (using list format)"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def load_used_style_from_final(final_dir: Path) -> Set[str]:
    """
    Load already-used style data from final/customization directory (train data)
    
    Args:
        final_dir: final directory path
    
    Returns:
        set of already-used style file paths (normalized)
    """
    used_style_paths = set()
    
    if not final_dir.exists():
        return used_style_paths
    
    # Iterate over all image_count_category directories
    for category_dir in final_dir.glob("*"):
        if not category_dir.is_dir():
            continue
        
        train_json_dir = category_dir / "train" / "json"
        if not train_json_dir.exists():
            continue
        
        # Read all JSON files
        for json_file in train_json_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Check whether it contains style category
                    categories = data.get('category', [])
                    if 'style' not in categories:
                        continue
                    
                    # Extract style image paths
                    input_images = data.get('input_images', [])
                    for img_path in input_images:
                        img_path_str = str(img_path)
                        # Check whether the path contains style directory
                        if '/style/' in img_path_str or '/customization/style' in img_path_str:
                            # Normalize path: extract path relative to processed/customization/style
                            if '/style/' in img_path_str:
                                # Extract filenames or relative paths
                                style_path = img_path_str.split('/style/')[-1]
                                # Build full path
                                full_path = str(DATA_DIR / "style" / style_path)
                                used_style_paths.add(full_path)
                            else:
                                used_style_paths.add(img_path_str)
            except Exception as e:
                print(f"Warning: failed to read JSON file {json_file}: {e}")
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
    Extend eval samples from unused style data
    
    Args:
        eval_samples: current eval sample list
        target_eval_count: target eval count
        processed_dir: processed directory path
        final_dir: final directory path
        seed: random seed
    
    Returns:
        extended eval sample list
    """
    current_eval_count = len(eval_samples)
    
    if current_eval_count >= target_eval_count:
        return eval_samples
    
    print(f"\nNeed to extend style eval: current {current_eval_count}, target {target_eval_count}")
    
    # Load already-used style data (from final/customization train data)
    used_style_paths = load_used_style_from_final(final_dir)
    print(f"Loaded {len(used_style_paths)} already-used style data from final/customization")
    
    # Load all style samples
    style_jsonl_path = processed_dir / "style.jsonl"
    all_style_samples = load_jsonl(style_jsonl_path)
    print(f"Loaded {len(all_style_samples)} style samples from processed")
    
    # Normalize paths: convert to absolute paths and normalize
    def normalize_path(path_str: str) -> str:
        """Normalize path"""
        if not path_str:
            return ""
        path = Path(path_str)
        if path.is_absolute():
            return str(path.resolve())
        else:
            return str((processed_dir / path_str).resolve())
    
    # Get normalized filepath set of current eval samples
    current_eval_paths = set()
    for sample in eval_samples:
        filepath = sample.get('filepath', '')
        if filepath:
            normalized = normalize_path(filepath)
            current_eval_paths.add(normalized)
    
    # Normalize already-used paths
    normalized_used_paths = {normalize_path(p) for p in used_style_paths}
    
    # Find unused style samples (neither in eval nor in final/train)
    unused_samples = []
    for sample in all_style_samples:
        filepath = sample.get('filepath', '')
        if filepath:
            normalized = normalize_path(filepath)
            if normalized not in current_eval_paths and normalized not in normalized_used_paths:
                unused_samples.append(sample)
    
    print(f"Found {len(unused_samples)} unused style samples")
    
    # Randomly select samples to supplement
    need_count = target_eval_count - current_eval_count
    if need_count > 0 and unused_samples:
        random.seed(seed)
        selected_samples = random.sample(unused_samples, min(need_count, len(unused_samples)))
        eval_samples.extend(selected_samples)
        print(f"Added {len(selected_samples)} samples to eval")
    elif need_count > 0:
        print(f"Warning: not enough unused style samples (need {need_count}, only {len(unused_samples)} available)")
    
    return eval_samples


def main():
    """Main function"""
    print("=" * 80)
    print("Customization data split script")
    print("=" * 80)
    print(f"\nEval sample count config: {EVAL_COUNTS}")
    print(f"Random seed: {RANDOM_SEED}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    all_statistics = {}
    
    # Process each category
    for class_name, eval_count in EVAL_COUNTS.items():
        print("\n" + "=" * 80)
        if eval_count is None or eval_count <= 0:
            print(f"Processing category: {class_name} (eval count: 0, all for train)")
        else:
            print(f"Processing category: {class_name} (eval count: {eval_count})")
        print("=" * 80)
        
        # For style category, inherit existing eval and train split from processed/customization
        if class_name == 'style':
            # Load existing train and eval data
            train_jsonl_path = DATA_DIR / f"{class_name}_train.jsonl"
            eval_jsonl_path = DATA_DIR / f"{class_name}_eval.jsonl"
            
            train_samples = load_jsonl(train_jsonl_path) if train_jsonl_path.exists() else []
            eval_samples = load_jsonl(eval_jsonl_path) if eval_jsonl_path.exists() else []
            
            print(f"Inherited from processed/customization: train={len(train_samples)}, eval={len(eval_samples)}")
            
            # If eval count is insufficient, extend from unused data
            if len(eval_samples) < eval_count:
                eval_samples = extend_style_eval_from_unused(
                    eval_samples,
                    eval_count,
                    DATA_DIR,
                    FINAL_DIR,
                    RANDOM_SEED
                )
            
            # Ensure train_samples does not contain samples already in eval_samples
            eval_filepaths = {s.get('filepath', '') for s in eval_samples if s.get('filepath')}
            filtered_train_samples = [
                s for s in train_samples 
                if s.get('filepath', '') not in eval_filepaths
            ]
            train_samples = filtered_train_samples
        else:
            # Load jsonl file
            jsonl_path = DATA_DIR / f"{class_name}.jsonl"
            samples = load_jsonl(jsonl_path)
            
            if not samples:
                print(f"Warning: category {class_name} has no data, skipping")
                continue
            
            # For scene category, use special handling
            if class_name == 'scene':
                train_samples, eval_samples = split_scene_samples(samples, eval_count, RANDOM_SEED)
            else:
                train_samples, eval_samples = split_samples(samples, eval_count, RANDOM_SEED)
            
            # If eval count is insufficient, extend from existing data
            if eval_count and eval_count > 0 and len(eval_samples) < eval_count:
                eval_samples = extend_eval_from_existing(
                    eval_samples,
                    eval_count,
                    DATA_DIR,
                    FINAL_OLD_DIR,
                    class_name,
                    RANDOM_SEED
                )
            
            # Train count is automatically (total - eval count)
            # Note: ensure eval_samples samples do not appear in train_samples
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
            
            # Remove samples already in eval from train_samples
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
        
        print(f"Train sample count: {len(train_samples)}")
        print(f"Eval sample count: {len(eval_samples)}")
        
        # Save train.json
        if train_samples:
            train_path = OUTPUT_DIR / f"{class_name}_train.json"
            save_json(train_samples, train_path)
            print(f"Saved train data to: {train_path}")
        
        # Save eval.json
        if eval_samples:
            eval_path = OUTPUT_DIR / f"{class_name}_eval.json"
            save_json(eval_samples, eval_path)
            print(f"Saved eval data to: {eval_path}")
        
        # Record statistics
        all_statistics[class_name] = {
            'total': len(train_samples) + len(eval_samples),
            'train': len(train_samples),
            'eval': len(eval_samples)
        }
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Split statistics")
    print("=" * 80)
    total_train = 0
    total_eval = 0
    total_all = 0
    
    for class_name, stats in sorted(all_statistics.items()):
        print(f"\nCategory: {class_name}")
        print(f"  Total samples: {stats['total']}")
        print(f"  Train samples: {stats['train']}")
        print(f"  Eval samples: {stats['eval']}")
        total_train += stats['train']
        total_eval += stats['eval']
        total_all += stats['total']
    
    print("\n" + "=" * 80)
    print(f"Total samples: {total_all}")
    print(f"Total Train samples: {total_train}")
    print(f"Total Eval samples: {total_eval}")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("All categories processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

