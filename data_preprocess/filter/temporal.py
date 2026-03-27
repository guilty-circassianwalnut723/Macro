#!/usr/bin/env python3
"""
Temporal data filtering script

Features:
1. Read data from the final/temporal directory
2. Filter samples with scores above threshold
3. Convert to minimal format, keeping only: task, idx, prompt, input_images, output_image
4. Save to filter/temporal directory
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
FINAL_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "final" / "temporal")
FILTER_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "filter" / "temporal")
# You can override FILTER_DIR to use a custom path if needed

# Filtering threshold (adjust according to actual needs)
TEMPORAL_SCORE_THRESHOLD = 6

# Filter config: {image_count_category: {train: count, eval: count}}
FILTER_CONFIG = {
    "1-3": {"train": 25000, "eval": 250},
    "4-5": {"train": 25000, "eval": 250},
    "6-7": {"train": 25000, "eval": 250},
    ">=8": {"train": 25000, "eval": 250},
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
    Filter samples
    
    Args:
        samples: list of samples
    
    Returns:
        filtered sample list
    """
    filtered = []
    
    for sample in samples:
        # Check scores (adjust field names according to actual data)
        temporal_score = sample.get("temporal_score", sample.get("score", 0))
        
        # Keep samples with scores above threshold
        if temporal_score >= TEMPORAL_SCORE_THRESHOLD:
            input_num = len(sample.get("input_images", []))
            if input_num > 10:
                # print(f"More than 10 input images: {input_num}")
                continue
            filtered.append(sample)
    
    return filtered


def main():
    """Main function"""
    print("=" * 80)
    print("Temporal data filtering script")
    print("=" * 80)
    print(f"Final directory: {FINAL_DIR}")
    print(f"Filter directory: {FILTER_DIR}")
    print(f"Temporal Score threshold: {TEMPORAL_SCORE_THRESHOLD}")
    print(f"Filter config: {FILTER_CONFIG}")
    print("=" * 80)
    
    if not FINAL_DIR.exists():
        print(f"Error: Final directory does not exist: {FINAL_DIR}")
        return
    
    # Ensure FILTER_DIR exists
    FILTER_DIR.mkdir(parents=True, exist_ok=True)
    
    # For train, clear the entire train directory and rebuild
    if (FILTER_DIR / "train").exists():
        print(f"Clearing Train directory: {FILTER_DIR / 'train'}")
        shutil.rmtree(FILTER_DIR / "train")
    (FILTER_DIR / "train").mkdir(parents=True, exist_ok=True)
    
    # Process train and eval data
    for split_type in ["train", "eval"]:
        print(f"\nProcessing {split_type} data...")
        
        # Iterate over all image_count_category directories
        for category_dir in (FINAL_DIR / split_type).glob("*"):
            if not category_dir.is_dir():
                continue
            
            image_count_category = category_dir.name
            
            # Check config; skip count control if not configured
            if image_count_category not in FILTER_CONFIG:
                print(f"\nProcessing category: {image_count_category} (no count limit)")
                target_count = None
            else:
                target_count = FILTER_CONFIG[image_count_category].get(split_type)
                if target_count is None:
                    print(f"\nProcessing category: {image_count_category} (no count limit)")
                    target_count = None
                else:
                    print(f"\nProcessing category: {image_count_category} (target count: {target_count})")
            
            if split_type == "eval" and target_count is None:
                continue
            
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
            
            # Read JSON files
            json_dir = category_dir / "json"
            if not json_dir.exists():
                print(f"  Skipping: JSON directory does not exist")
                continue
            
            samples = []
            for json_file in sorted(json_dir.glob("*.json")):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        sample = json.load(f)
                        # Extract idx
                        idx = int(json_file.stem)
                        samples.append((idx, sample))
                except Exception as e:
                    print(f"  Warning: failed to read JSON file {json_file}: {e}")
                    continue
            
            print(f"  Loaded {len(samples)} samples")
            
            # Filter samples
            filtered_samples = filter_samples([s[1] for s in samples])
            print(f"  After filtering: {len(filtered_samples)} samples")
            
            # Create list of filtered samples with their indices
            filtered_with_idx = [(idx, sample) for idx, sample in samples if sample in filtered_samples]
            
            # For eval, exclude already existing samples
            if split_type == "eval" and existing_count > 0:
                original_count = len(filtered_with_idx)
                filtered_with_idx = [
                    (idx, sample) for idx, sample in filtered_with_idx
                    if not (
                        sample.get("unique_id") in existing_identifiers or
                        (sample.get("source_file", ""), sample.get("source_line", -1), sample.get("true_index", -1)) in existing_identifiers
                    )
                ]
                print(f"  After excluding existing samples: {len(filtered_with_idx)} samples (removed {original_count - len(filtered_with_idx)})")
            
            # If target count is configured and filtered samples exceed it, shuffle and take the first n
            if target_count is not None:
                if split_type == "eval":
                    # Number needed to fill eval quota
                    needed_count = target_count - existing_count
                    if needed_count > 0 and len(filtered_with_idx) > needed_count:
                        # Set random seed for reproducibility (based on split_type and image_count_category)
                        seed_str = f"{RANDOM_SEED}_{split_type}_{image_count_category}"
                        seed_hash = get_deterministic_seed(seed_str)
                        random.seed(seed_hash)
                        random.shuffle(filtered_with_idx)
                        # Take the first needed_count
                        filtered_with_idx = filtered_with_idx[:needed_count]
                        print(f"  Shuffled and took first {needed_count} samples to fill quota")
                else:
                    # Train keeps the original logic
                    if len(filtered_with_idx) > target_count:
                        # Set random seed for reproducibility (based on split_type and image_count_category)
                        seed_str = f"{RANDOM_SEED}_{split_type}_{image_count_category}"
                        seed_hash = get_deterministic_seed(seed_str)
                        random.seed(seed_hash)
                        random.shuffle(filtered_with_idx)
                        # Take the first target_count
                        filtered_with_idx = filtered_with_idx[:target_count]
                        print(f"  Shuffled and took first {target_count} samples")
            
            # Convert to minimal format and save
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
            
            # Save samples
            for i, (original_idx, sample) in enumerate(filtered_with_idx, start=start_idx):
                minimal = convert_to_minimal(sample, "temporal", i)
                output_file = output_dir / f"{i:08d}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(minimal, f, ensure_ascii=False, indent=2)
            
            final_count = existing_count + len(filtered_with_idx) if split_type == "eval" and existing_count > 0 else len(filtered_with_idx)
            print(f"  Saved to: {output_dir} (total {final_count} samples)")
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()

