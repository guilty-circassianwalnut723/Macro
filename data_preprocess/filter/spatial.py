#!/usr/bin/env python3
"""
Spatial data filtering script

Features:
1. Unified control of filter config and count for outdoor, indoor, object subtypes
2. Read data from final/spatial directory (organized by subtype)
3. Load all three types of data, merge, shuffle, and take the first n
4. Convert to minimal format, keeping only: task, idx, prompt, input_images, output_image
5. Re-number and save to filter/spatial directory (without subtype path)
"""

import json
import random
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List

# Add utils path
import sys
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from utils.convert_to_minimal import convert_to_minimal

# ====== Configuration parameters ======
FINAL_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "final" / "spatial")
FILTER_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "filter" / "spatial")
# You can override FILTER_DIR to use a custom path if needed

# Subtype selection: outdoor, indoor, or object
# Set to None to process all subtypes, or a specific subtype string to process only that subtype
SUB_TYPE = None  # Modify this value to select the subtype to process; None processes all subtypes

# Filter config: {sub_type: {image_count_category: {train: count, eval: count}}}
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

# Random seed
RANDOM_SEED = 42
# ======================


def get_deterministic_seed(seed_str: str) -> int:
    """
    Generate a deterministic random seed (using hashlib to ensure cross-run consistency)
    
    Args:
        seed_str: seed string
    
    Returns:
        deterministic integer seed
    """
    # Use hashlib to generate a deterministic hash value
    hash_obj = hashlib.md5(seed_str.encode('utf-8'))
    hash_int = int(hash_obj.hexdigest(), 16)
    # Modulo to keep within reasonable range
    return hash_int % (2**31)


def filter_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter samples (for spatial, all are currently retained)
    
    Args:
        samples: list of samples
    
    Returns:
        filtered sample list
    """
    # For spatial, all samples are currently retained

    filtered = []
    for sample in samples:
        input_num = len(sample.get("input_images", []))
        if input_num > 10:
            print(f"More than 10 input images: {input_num}")
            continue
        filtered.append(sample)
    return filtered


def main():
    """Main function"""
    # Determine the list of subtypes to process
    if SUB_TYPE is None:
        # Process all subtypes
        sub_types_to_process = list(FILTER_CONFIG.keys())
        print("=" * 80)
        print("Spatial data filtering script - processing all subtypes (merged unified sampling)")
        print("=" * 80)
    else:
        # Process only the specified subtype
        if SUB_TYPE not in FILTER_CONFIG:
            raise ValueError(f"Unsupported subtype: {SUB_TYPE}, supported types: {list(FILTER_CONFIG.keys())}")
        sub_types_to_process = [SUB_TYPE]
        print("=" * 80)
        print(f"Spatial {SUB_TYPE.upper()} data filtering script")
        print("=" * 80)
    
    print(f"Final directory: {FINAL_DIR}")
    print(f"Filter directory: {FILTER_DIR}")
    print(f"Subtypes to process: {sub_types_to_process}")
    print(f"Filter config: {FILTER_CONFIG}")
    print("=" * 80)
    
    if not FINAL_DIR.exists():
        print(f"Error: Final directory does not exist: {FINAL_DIR}")
        return
    
    # Ensure FILTER_DIR exists
    FILTER_DIR.mkdir(parents=True, exist_ok=True)
    
    # For train, clear the entire train directory and rebuild
    train_dir = FILTER_DIR / "train"
    if train_dir.exists():
        print(f"Clearing Train directory: {train_dir}")
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all image_count_categories (from all subtype configs)
    all_categories = set()
    for sub_type_config in FILTER_CONFIG.values():
        all_categories.update(sub_type_config.keys())
    all_categories = sorted(list(all_categories))
    
    # Process train and eval data
    for split_type in ["train", "eval"]:
        print(f"\n{'=' * 80}")
        print(f"Processing {split_type} data")
        print(f"{'=' * 80}")
        
        # For each image_count_category, merge all subtype data and process together
        for image_count_category in all_categories:
            print(f"\nProcessing category: {image_count_category}")
            
            # Collect data from all subtypes
            all_samples = []
            category_total_count = 0
            sub_type_counts = {}  # Record configured count for each subtype
            
            for current_sub_type in sub_types_to_process:
                # Calculate target count for this subtype and category
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
                
                # Read samples for this subtype and category
                sub_type_dir = FINAL_DIR / split_type / current_sub_type
                json_dir = sub_type_dir / image_count_category / "json"
                
                if not json_dir.exists():
                    print(f"  Skipping {current_sub_type}: JSON directory does not exist")
                    continue
                
                samples = []
                for json_file in sorted(json_dir.glob("*.json")):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            sample = json.load(f)
                            # Save subtype info and original idx
                            sample['_sub_type'] = current_sub_type
                            sample['_original_idx'] = int(json_file.stem)
                            samples.append(sample)
                    except Exception as e:
                        print(f"  Warning: failed to read JSON file {json_file}: {e}")
                        continue
                
                count_info = f"Loaded {len(samples)} samples"
                if current_sub_type in sub_type_counts:
                    count_info += f" (configured count: {sub_type_counts[current_sub_type]})"
                print(f"  {current_sub_type}: {count_info}")
                all_samples.extend(samples)
            
            if not all_samples:
                print(f"  Skipping: no samples found")
                continue
            
            print(f"  Total loaded: {len(all_samples)} samples (from all subtypes)")
            
            # Filter samples
            filtered_samples = filter_samples(all_samples)
            print(f"  After filtering: {len(filtered_samples)} samples")
            
            # Calculate total target count (sum of all subtype configs)
            target_count = category_total_count if category_total_count > 0 else None
            
            if target_count is not None:
                print(f"  Target total: {target_count} (sum of all subtype configs)")
            
            # For eval, check the existing data count
            output_dir = FILTER_DIR / split_type / image_count_category
            existing_count = 0
            existing_identifiers = set()
            if split_type == "eval" and output_dir.exists():
                existing_files = list(output_dir.glob("*.json"))
                existing_count = len(existing_files)
                if target_count is not None and existing_count >= target_count:
                    print(f"  Eval data already meets target count ({existing_count} >= {target_count}), skipping")
                    continue
                elif existing_count > 0:
                    print(f"  Existing Eval data: {existing_count}, target: {target_count}, need {target_count - existing_count} more")
                    # Read unique_id or other unique identifier from existing files
                    for existing_file in output_dir.glob("*.json"):
                        try:
                            with open(existing_file, 'r', encoding='utf-8') as f:
                                existing_sample = json.load(f)
                                # Try to use unique_id; fall back to other unique identifier if unavailable
                                unique_id = existing_sample.get("unique_id")
                                if unique_id:
                                    existing_identifiers.add(unique_id)
                                else:
                                    # Use source_file+source_line+true_index as fallback
                                    source_file = existing_sample.get("source_file", "")
                                    source_line = existing_sample.get("source_line", -1)
                                    true_index = existing_sample.get("true_index", -1)
                                    existing_identifiers.add((source_file, source_line, true_index))
                        except Exception as e:
                            print(f"  Warning: failed to read existing file {existing_file}: {e}")
                            continue
            
            # For eval, exclude already existing samples
            if split_type == "eval" and existing_count > 0:
                original_count = len(filtered_samples)
                filtered_samples = [
                    sample for sample in filtered_samples
                    if not (
                        sample.get("unique_id") in existing_identifiers or
                        (sample.get("source_file", ""), sample.get("source_line", -1), sample.get("true_index", -1)) in existing_identifiers
                    )
                ]
                print(f"  After excluding existing samples: {len(filtered_samples)} samples (removed {original_count - len(filtered_samples)})")
            
            # If target count is configured and filtered samples exceed it, shuffle and take the first n
            if target_count is not None:
                if split_type == "eval":
                    # Number needed to fill eval quota
                    needed_count = target_count - existing_count
                    if needed_count > 0 and len(filtered_samples) > needed_count:
                        # Set random seed for reproducibility (based on split_type and image_count_category)
                        seed_str = f"{RANDOM_SEED}_{split_type}_{image_count_category}"
                        seed_hash = get_deterministic_seed(seed_str)
                        random.seed(seed_hash)
                        random.shuffle(filtered_samples)
                        # Take the first needed_count
                        filtered_samples = filtered_samples[:needed_count]
                        print(f"  Shuffled and took first {needed_count} samples to fill quota")
                    elif needed_count > 0:
                        print(f"  Sample count {len(filtered_samples)} <= needed count {needed_count}, keeping all samples to fill quota")
                else:
                    # Train keeps the original logic
                    if len(filtered_samples) > target_count:
                        # Set random seed for reproducibility (based on split_type and image_count_category)
                        seed_str = f"{RANDOM_SEED}_{split_type}_{image_count_category}"
                        seed_hash = get_deterministic_seed(seed_str)
                        random.seed(seed_hash)
                        random.shuffle(filtered_samples)
                        # Take the first target_count
                        filtered_samples = filtered_samples[:target_count]
                        print(f"  Shuffled and took first {target_count} samples")
                    else:
                        print(f"  Sample count {len(filtered_samples)} <= target count {target_count}, keeping all samples")
            else:
                print(f"  No count limit, keeping all {len(filtered_samples)} samples")
            
            # Convert to minimal format and save with sequential numbering
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if split_type == "eval" and existing_count > 0:
                # eval: find current max index, start from next index
                existing_indices = []
                for existing_file in output_dir.glob("*.json"):
                    try:
                        idx = int(existing_file.stem)
                        existing_indices.append(idx)
                    except ValueError:
                        continue
                start_idx = max(existing_indices, default=0) + 1
            else:
                # train: re-number starting from 1
                start_idx = 1
                # Clear existing files (train only)
                for existing_file in output_dir.glob("*.json"):
                    existing_file.unlink()
            
            # Save with re-numbered indices
            saved_count = 0
            for i, sample in enumerate(filtered_samples, start=start_idx):
                # Remove temporary fields
                sub_type = sample.pop('_sub_type', 'unknown')
                original_idx = sample.pop('_original_idx', -1)
                
                minimal = convert_to_minimal(sample, "spatial", i)
                output_file = output_dir / f"{i:08d}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(minimal, f, ensure_ascii=False, indent=2)
                saved_count += 1
            
            final_count = existing_count + saved_count if split_type == "eval" and existing_count > 0 else saved_count
            print(f"  Saved to: {output_dir} (total {final_count} samples, indices: {start_idx}-{start_idx + saved_count - 1 if saved_count > 0 else start_idx - 1})")
    
    print(f"\n{'=' * 80}")
    print("All data processing complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

