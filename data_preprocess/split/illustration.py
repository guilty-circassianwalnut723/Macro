#!/usr/bin/env python3
"""
Illustration data split script: sample from processed/illustration and split into train/eval

Features:
1. Read all jsonl files from processed/illustration directory
2. Group by category label and image count
3. Sample based on configured eval count
4. All remaining data goes to train
5. Generate train and eval datasets, save to data_hl02/split/illustration directory
6. Save as json format (using list), control saved content to avoid large invalid storage
7. Record topic category and image num category statistics
8. If data volume is large, split into multiple sub-files
"""

import json
import random
from collections import defaultdict
from pathlib import Path

MACRO_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = MACRO_DIR / "data"
from typing import Any, Dict, List, Optional, Set, Tuple
from tqdm import tqdm

# ====== Configuration parameters ======
PROCESSED_DIR = DATA_DIR / "processed" / "illustration"
SOURCE_DIR = DATA_DIR / "source" / "illustration"
OUTPUT_DIR = DATA_DIR / "split" / "illustration"

# Eval sample count config: {image_group: eval_count}
EVAL_COUNTS = {
    "1-3": 500,
    "4-5": 500,
    "6-7": 500,
    ">=8": 500,
}

# Maximum number of jsonl files to read; None means read all files
MAX_FILES: Optional[int] = None

# Maximum samples per json file (for splitting large files)
MAX_SAMPLES_PER_FILE = 10000

# Random seed
RANDOM_SEED = 42
# ======================


def find_true_indices(image_information_flow: List[bool]) -> List[int]:
    """Find all indices that are true"""
    return [i for i, val in enumerate(image_information_flow) if val]


def get_image_group(image_count: int) -> str:
    """Determine group based on image count"""
    if image_count <= 3:
        return "1-3"
    elif image_count <= 5:
        return "4-5"
    elif image_count <= 7:
        return "6-7"
    else:
        return ">=8"


def load_jsonl_files(data_dir: Path, max_files: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load data from jsonl files"""
    all_samples = []
    all_jsonl_files = list(data_dir.glob("*.jsonl"))
    jsonl_files = sorted(all_jsonl_files)
    
    if max_files is not None and max_files > 0:
        jsonl_files = jsonl_files[:max_files]
        print(f"Found {len(all_jsonl_files)} jsonl files, will read the first {len(jsonl_files)}")
    else:
        print(f"Found {len(jsonl_files)} jsonl files")
    
    for jsonl_file in tqdm(jsonl_files, desc="Loading files"):
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
    """Load original sample from source file to get image count info"""
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
    """Get true index and image count info from sample"""
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
    """Create a minimal sample keeping only necessary fields, to avoid large invalid storage"""
    minimal = {
        "source_file": sample.get("source_file"),
        "source_line": sample.get("source_line"),
        "category": sample.get("category"),
        "true_index": true_index,
        "image_count": true_index + 1,
        "actual_image_count": true_index,
        "image_num_category": image_group,  # Add image_num_category field
    }
    
    # Keep only necessary fields, avoid saving large amounts of invalid data
    # If other fields are needed, reload from source during the gen stage
    return minimal


def organize_samples_by_category_and_image_group(
    samples: List[Dict[str, Any]],
    source_dir: Path,
) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    """Organize samples grouped by category and image count"""
    organized: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    
    print("Organizing samples...")
    for sample in tqdm(samples, desc="Grouping"):
        category = sample.get("category")
        if category is None:
            continue
        
        info = get_true_indices_and_image_count_from_sample(sample, source_dir)
        if info is None:
            continue
        
        true_indices = info["true_indices"]
        
        if not true_indices:
            continue
        
        # Create an independent sample for each true index
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
    """Sample eval data by image_group, prioritizing uniform sampling across different topic categories"""
    rng = random.Random(seed)
    all_selected_samples = []
    used_samples: Set[Tuple[str, int]] = set()
    
    # Process by image_group
    all_image_groups = set()
    for category_data in organized_samples.values():
        all_image_groups.update(category_data.keys())
    
    for img_grp in sorted(all_image_groups):
        eval_limit = eval_limits.get(img_grp, 0)
        if eval_limit <= 0:
            continue
        
        # Collect available samples from all categories under this image_group
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
        
        # Group by sample_id, build sample_id index for each category
        category_samples_by_id: Dict[int, Dict[Tuple[str, int], List[Dict[str, Any]]]] = {}
        category_total_counts: Dict[int, int] = {}  # Record total sample count per category
        for category_id, samples in available_by_category.items():
            samples_by_id: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
            for sample in samples:
                sample_id = (sample.get("source_file"), sample.get("source_line"))
                if sample_id[0] and sample_id[1] is not None:
                    samples_by_id[sample_id].append(sample)
            category_samples_by_id[category_id] = samples_by_id
            category_total_counts[category_id] = len(samples_by_id)  # Record unique sample_id count
        
        # Prioritize uniform sampling from each category
        num_categories = len(available_by_category)
        if num_categories == 0:
            continue
        
        # Calculate target sample count for each category
        samples_per_category = eval_limit // num_categories
        remaining_needed = eval_limit
        
        selected_samples = []
        category_ids = list(available_by_category.keys())
        rng.shuffle(category_ids)  # Randomize category processing order
        
        # Track how many eval samples have been allocated per category
        category_eval_counts: Dict[int, int] = {cat_id: 0 for cat_id in category_ids}
        
        # Phase 1: try to uniformly sample from each category
        for category_id in category_ids:
            if remaining_needed <= 0:
                break
            
            samples_by_id = category_samples_by_id[category_id]
            unique_sample_ids = list(samples_by_id.keys())
            rng.shuffle(unique_sample_ids)
            
            # Calculate max allocatable eval count for this category: ensure train >= eval
            # If total is N, allocate at most floor(N/2) to eval
            total_count = category_total_counts[category_id]
            max_eval_for_category = total_count // 2  # Allocate at most half to eval
            
            # Sample from this category, at most samples_per_category, but not exceeding remaining need and max limit
            target_count = min(samples_per_category, len(unique_sample_ids), remaining_needed, max_eval_for_category)
            
            for sample_id in unique_sample_ids[:target_count]:
                if remaining_needed <= 0:
                    break
                # Re-check: ensure train data will not be fewer than eval data after allocation
                current_eval_count = category_eval_counts[category_id]
                # If allocation would exceed max limit, skip
                if current_eval_count >= max_eval_for_category:
                    break
                # Check if constraint is satisfied after allocation: train >= eval
                # After allocation: eval = current_eval_count + 1, train = total_count - (current_eval_count + 1)
                # Must ensure: train >= eval, i.e. total_count >= 2 * (current_eval_count + 1)
                if total_count < 2 * (current_eval_count + 1):
                    break
                
                samples_for_id = samples_by_id[sample_id]
                if samples_for_id:
                    selected_sample = rng.choice(samples_for_id)
                    selected_samples.append(selected_sample)
                    used_samples.add(sample_id)
                    category_eval_counts[category_id] += 1
                    remaining_needed -= 1
        
        # Phase 2: if there is remaining demand, randomly sample from all categories to fill
        if remaining_needed > 0:
            # Collect all unused sample_ids, checking whether constraints are satisfied
            all_remaining_samples_by_id: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
            sample_id_to_category: Dict[Tuple[str, int], int] = {}  # Record which category each sample_id belongs to
            
            for category_id, samples_by_id in category_samples_by_id.items():
                current_eval_count = category_eval_counts[category_id]
                total_count = category_total_counts[category_id]
                
                # Max eval count allocatable for this category (ensuring train >= eval)
                max_eval_for_category = total_count // 2
                
                # If max limit reached, skip this category
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
                
                # Check whether allocation is still possible (not exceeding max limit, and train >= eval after allocation)
                if current_eval_count >= max_eval_for_category:
                    continue
                
                # After allocation: eval = current_eval_count + 1, train = total_count - (current_eval_count + 1)
                # Must ensure: train >= eval, i.e. total_count - (current_eval_count + 1) >= current_eval_count + 1
                # i.e.: total_count >= 2 * (current_eval_count + 1)
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
    """Save samples to json file (using list format)"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def save_samples_in_chunks(
    samples: List[dict],
    output_dir: Path,
    split_type: str,
    max_samples_per_file: int = 10000,
):
    """Save samples to json file (using list format); split into multiple files if data volume is large"""
    if not samples:
        return
    
    if len(samples) <= max_samples_per_file:
        # Single file
        output_file = output_dir / f"{split_type}.json"
        save_json(samples, output_file)
        print(f"Saved {len(samples)} {split_type} samples to: {output_file}")
    else:
        # Multiple files
        num_files = (len(samples) + max_samples_per_file - 1) // max_samples_per_file
        for file_idx in range(num_files):
            start_idx = file_idx * max_samples_per_file
            end_idx = min(start_idx + max_samples_per_file, len(samples))
            chunk = samples[start_idx:end_idx]
            
            output_file = output_dir / f"{split_type}_{file_idx:04d}.json"
            save_json(chunk, output_file)
            print(f"Saved {len(chunk)} {split_type} samples to: {output_file} (file {file_idx + 1}/{num_files})")


def main():
    """Main function"""
    print("=" * 80)
    print("Illustration data split script")
    print("=" * 80)
    print(f"Processing directory: {PROCESSED_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Max files to read: {MAX_FILES if MAX_FILES else 'all'}")
    print(f"Max samples per file: {MAX_SAMPLES_PER_FILE}")
    print(f"\nEval sample count config:")
    for img_grp, count in EVAL_COUNTS.items():
        print(f"  {img_grp}: {count}")
    print("=" * 80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all samples
    print("\nStep 1: Loading samples...")
    all_samples = load_jsonl_files(PROCESSED_DIR, max_files=MAX_FILES)
    print(f"Total loaded: {len(all_samples)} successful samples")
    
    # Organize samples
    print("\nStep 2: Grouping by category and image count...")
    organized_samples = organize_samples_by_category_and_image_group(all_samples, SOURCE_DIR)
    
    print(f"\nFound {len(organized_samples)} categories")
    for category_id in sorted(organized_samples.keys()):
        category_data = organized_samples[category_id]
        total = sum(len(samples) for samples in category_data.values())
        print(f"  Category {category_id}: {total} samples")
        for img_grp, samples in category_data.items():
            print(f"    {img_grp}: {len(samples)} samples")
    
    # Sample eval data
    print("\nStep 3: Sampling eval data...")
    random.seed(RANDOM_SEED)
    all_eval_samples, used_samples = sample_eval_data_by_image_group(
        organized_samples, EVAL_COUNTS, RANDOM_SEED
    )
    
    # Organize train data (all samples not used for eval)
    print("\nStep 4: Organizing train data...")
    all_train_samples = []
    for category_id, category_data in organized_samples.items():
        for img_grp, samples in category_data.items():
            for sample in samples:
                sample_id = (sample.get("source_file"), sample.get("source_line"))
                if sample_id not in used_samples:
                    all_train_samples.append(sample)
    
    # Calculate statistics: {topic_category: {image_num_category: count}}
    category_stats = {}  # {topic_category: {image_num_category: {train: count, eval: count}}}
    
    for category_id in organized_samples.keys():
        category_stats[category_id] = {
            "1-3": {"train": 0, "eval": 0},
            "4-5": {"train": 0, "eval": 0},
            "6-7": {"train": 0, "eval": 0},
            ">=8": {"train": 0, "eval": 0},
        }
    
    # Count train data
    for sample in all_train_samples:
        category_id = sample.get("category")
        image_group = sample.get("image_num_category")
        if category_id is not None and image_group:
            if category_id in category_stats and image_group in category_stats[category_id]:
                category_stats[category_id][image_group]["train"] += 1
    
    # Count eval data
    for sample in all_eval_samples:
        category_id = sample.get("category")
        image_group = sample.get("image_num_category")
        if category_id is not None and image_group:
            if category_id in category_stats and image_group in category_stats[category_id]:
                category_stats[category_id][image_group]["eval"] += 1
    
    # Save train and eval data
    print("\nStep 5: Saving data...")
    
    # Save train data (split by file)
    if all_train_samples:
        save_samples_in_chunks(
            all_train_samples,
            OUTPUT_DIR,
            "train",
            MAX_SAMPLES_PER_FILE
        )
    
    # Save eval data (split by file)
    if all_eval_samples:
        save_samples_in_chunks(
            all_eval_samples,
            OUTPUT_DIR,
            "eval",
            MAX_SAMPLES_PER_FILE
        )
    
    # Save statistics (including topic category and image num category stats)
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
    print(f"\nSaved statistics to: {stats_file}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Split statistics")
    print("=" * 80)
    print(f"Total Train samples: {len(all_train_samples)}")
    print(f"Total Eval samples: {len(all_eval_samples)}")
    print(f"Total samples: {len(all_train_samples) + len(all_eval_samples)}")
    print("\nStatistics by topic category and image num category:")
    for category_id in sorted(category_stats.keys()):
        print(f"  Category {category_id}:")
        for img_grp in ["1-3", "4-5", "6-7", ">=8"]:
            train_count = category_stats[category_id][img_grp]["train"]
            eval_count = category_stats[category_id][img_grp]["eval"]
            if train_count > 0 or eval_count > 0:
                print(f"    {img_grp}: Train={train_count}, Eval={eval_count}, Total={train_count + eval_count}")
    print("=" * 80)
    
    print("\nSplit complete!")


if __name__ == "__main__":
    main()
