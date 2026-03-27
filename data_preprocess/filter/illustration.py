#!/usr/bin/env python3
"""
Illustration data filtering script

Features:
1. Read data from the final/illustration directory
2. Filter samples with scores above threshold
3. Remove samples where all image_contributions are false
4. Convert to minimal format, keeping only: task, idx, prompt, input_images, output_image
5. Save to filter/illustration directory
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


def get_image_count_category(image_count: int) -> str:
    """Return category directory name based on image count"""
    if image_count <= 3:
        return "1-3"
    elif image_count <= 5:
        return "4-5"
    elif image_count <= 7:
        return "6-7"
    else:
        return ">=8"

# ====== Configuration parameters ======
FINAL_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "final" / "illustration")
FILTER_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "filter" / "illustration")
# You can override FILTER_DIR to use a custom path if needed

# Filtering thresholds
TRAINING_SCORE_THRESHOLD = 6
GUIDANCE_SCORE_THRESHOLD = 6

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
        # Check image_contributions and compute effective_image_count (count of true entries)
        image_contributions = sample.get("image_contributions", [])
        input_images = sample.get("input_images", [])
        if input_images == [] or image_contributions == []:
            continue
        assert len(input_images) == len(image_contributions), f"input_images and image_contributions length mismatch: {len(input_images)} != {len(image_contributions)}, sample: {sample.get('idx')}"
        
        filtered_input_images = [input_images[i] for i in range(len(input_images)) if image_contributions[i]]
        if len(filtered_input_images) == 0 or len(filtered_input_images) > 10:
            continue

        if isinstance(image_contributions, list):
            effective_image_count = sum(1 for x in image_contributions if x is True)
        else:
            effective_image_count = sample.get("effective_image_count", 0)
        assert effective_image_count == len(filtered_input_images), f"effective_image_count and filtered_input_images length mismatch: {effective_image_count} != {len(filtered_input_images)}, sample: {sample.get('idx')}"

        sample['input_images'] = filtered_input_images
        image_count = sample.get("image_count", 0)
        
        # Remove samples where effective_image_count or image_count is 0
        if effective_image_count == 0 or image_count == 0:
            continue

        is_invalid = sample.get("is_invalid", False)
        
        # Check scores
        training_score = sample.get("suitable", sample.get("training_score", 0))
        guidance_score = sample.get("guidance_score", 0)
        
        # Keep samples with scores above threshold
        if training_score >= TRAINING_SCORE_THRESHOLD and guidance_score >= GUIDANCE_SCORE_THRESHOLD \
            and effective_image_count <= 10:
            if not is_invalid:
                filtered.append(sample)
            else:
                # print(f"  Removing invalid sample: {sample.get('unique_id')}")
                continue
    
    return filtered


def get_actual_image_count(sample: Dict[str, Any]) -> int:
    """
    Get the actual number of input images for a sample (excluding output image)
    
    Args:
        sample: sample data
    
    Returns:
        number of input images (input_images count only)
    """
    input_images = sample.get("input_images", [])
    
    # Count only input images, excluding the output image
    count = len(input_images) if isinstance(input_images, list) else 0
    
    return count


def main():
    """Main function"""
    print("=" * 80)
    print("Illustration data filtering script")
    print("=" * 80)
    print(f"Final directory: {FINAL_DIR}")
    print(f"Filter directory: {FILTER_DIR}")
    print(f"Training Score threshold: {TRAINING_SCORE_THRESHOLD}")
    print(f"Guidance Score threshold: {GUIDANCE_SCORE_THRESHOLD}")
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
    
    # Process train and eval data
    for split_type in ["train", "eval"]:
        print(f"\nProcessing {split_type} data...")
        
        # Read all data from the final directory at once
        print("  Reading all data...")
        all_samples = []
        split_dir = FINAL_DIR / split_type
        if not split_dir.exists():
            print(f"  Skipping: {split_type} directory does not exist")
            continue
        
        # Iterate over all image_count_category directories
        for category_dir in split_dir.glob("*"):
            if not category_dir.is_dir():
                continue
            
            json_dir = category_dir / "json"
            if not json_dir.exists():
                continue
            
            # Read all JSON files in this directory
            for json_file in sorted(json_dir.glob("*.json")):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        sample = json.load(f)
                        # Extract idx
                        idx = int(json_file.stem)
                        all_samples.append((idx, sample))
                except Exception as e:
                    print(f"  Warning: failed to read JSON file {json_file}: {e}")
                    continue
        
        print(f"  Total loaded: {len(all_samples)} samples")
        
        # Filter samples
        filtered_samples = filter_samples([s[1] for s in all_samples])
        print(f"  After filtering: {len(filtered_samples)} samples")
        
        # Create list of filtered samples with their indices
        filtered_with_idx = [(idx, sample) for idx, sample in all_samples if sample in filtered_samples]
        
        # Deduplicate by unique_id to avoid processing duplicates from different category directories
        print("  Deduplicating (by unique_id or source_file+source_line+true_index)...")
        seen_unique_ids = set()
        seen_source_keys = set()  # For samples without unique_id
        deduplicated_samples = []
        duplicate_count = 0
        
        for idx, sample in filtered_with_idx:
            unique_id = sample.get("unique_id")
            if unique_id:
                # Deduplicate by unique_id
                if unique_id in seen_unique_ids:
                    duplicate_count += 1
                    continue
                seen_unique_ids.add(unique_id)
            else:
                # For samples without unique_id, use source_file+source_line+true_index combination
                source_file = sample.get("source_file", "")
                source_line = sample.get("source_line", -1)
                true_index = sample.get("true_index", -1)
                source_key = (source_file, source_line, true_index)
                if source_key in seen_source_keys:
                    duplicate_count += 1
                    continue
                seen_source_keys.add(source_key)
            
            deduplicated_samples.append((idx, sample))
        
        if duplicate_count > 0:
            print(f"  Deduplication: removed {duplicate_count} duplicate samples")
        print(f"  After deduplication: {len(deduplicated_samples)} samples")
        
        # Re-categorize based on actual image count
        print("  Re-categorizing based on actual image count...")
        samples_by_category = {}  # {image_count_category: [(idx, sample), ...]}
        
        for idx, sample in deduplicated_samples:
            # Calculate actual image count
            actual_count = get_actual_image_count(sample)
            
            # Calculate new category based on actual image count
            new_category = get_image_count_category(actual_count)
            
            if new_category not in samples_by_category:
                samples_by_category[new_category] = []
            samples_by_category[new_category].append((idx, sample))
        
        print(f"  After re-categorization, {len(samples_by_category)} categories:")
        for category, samples_list in sorted(samples_by_category.items()):
            print(f"    {category}: {len(samples_list)} samples")
        
        # Save by category
        for image_count_category, samples_list in sorted(samples_by_category.items()):
            print(f"\n  Processing category: {image_count_category} ({len(samples_list)} samples)")
            
            # Check config; skip count control if not configured
            if image_count_category not in FILTER_CONFIG:
                print(f"    (no count limit)")
                target_count = None
            else:
                target_count = FILTER_CONFIG[image_count_category].get(split_type)
                if target_count is None:
                    print(f"    (no count limit)")
                    target_count = None
                else:
                    print(f"    (target count: {target_count})")

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
                    print(f"    Eval data already meets target count ({existing_count} >= {target_count}), skipping")
                    continue
                elif existing_count > 0:
                    print(f"    Existing Eval data: {existing_count}, target: {target_count}, need {target_count - existing_count} more")
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
                            print(f"    Warning: failed to read existing file {existing_file}: {e}")
                            continue
            
            # For eval, exclude already existing samples
            if split_type == "eval" and existing_count > 0:
                original_count = len(samples_list)
                samples_list = [
                    (idx, sample) for idx, sample in samples_list
                    if not (
                        sample.get("unique_id") in existing_identifiers or
                        (sample.get("source_file", ""), sample.get("source_line", -1), sample.get("true_index", -1)) in existing_identifiers
                    )
                ]
                print(f"    After excluding existing samples: {len(samples_list)} samples (removed {original_count - len(samples_list)})")
            
            # If target count is configured and filtered samples exceed it, shuffle and take the first n
            if target_count is not None:
                if split_type == "eval":
                    # Number needed to fill eval quota
                    needed_count = target_count - existing_count
                    if needed_count > 0 and len(samples_list) > needed_count:
                        # Set random seed for reproducibility (based on split_type and image_count_category)
                        seed_str = f"{RANDOM_SEED}_{split_type}_{image_count_category}"
                        seed_hash = get_deterministic_seed(seed_str)
                        random.seed(seed_hash)
                        random.shuffle(samples_list)
                        # Take the first needed_count
                        samples_list = samples_list[:needed_count]
                        print(f"    Shuffled and took first {needed_count} samples to fill quota")
                else:
                    # Train keeps the original logic
                    if len(samples_list) > target_count:
                        # Set random seed for reproducibility (based on split_type and image_count_category)
                        seed_str = f"{RANDOM_SEED}_{split_type}_{image_count_category}"
                        seed_hash = get_deterministic_seed(seed_str)
                        random.seed(seed_hash)
                        random.shuffle(samples_list)
                        # Take the first target_count
                        samples_list = samples_list[:target_count]
                        print(f"    Shuffled and took first {target_count} samples")
            
            # Convert to minimal format and save
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if split_type == "eval" and existing_count > 0:
                if target_count is not None and existing_count >= target_count:
                    print(f"    Eval data already meets target count ({existing_count} >= {target_count}), skipping")
                    continue

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
            for i, (original_idx, sample) in enumerate(samples_list, start=start_idx):
                minimal = convert_to_minimal(sample, "illustration", i)
                output_file = output_dir / f"{i:08d}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(minimal, f, ensure_ascii=False, indent=2)
            
            final_count = existing_count + len(samples_list) if split_type == "eval" and existing_count > 0 else len(samples_list)
            print(f"    Saved to: {output_dir} (total {final_count} samples)")
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()

