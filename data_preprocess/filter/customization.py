import os
#!/usr/bin/env python3
"""
Customization data filtering script

Features:
1. Read data from the final/customization directory
2. Check for consistency_scores and following_score; call API to fill missing scores
3. Filter samples based on threshold
4. Convert to minimal format, keeping only: task, idx, prompt, input_images, output_image
5. Save to filter/customization directory
"""

import json
import random
import shutil
import sys
import threading
import importlib.util
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add utils path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from utils.convert_to_minimal import convert_to_minimal

# Add reference scoring script path
SCORE_MODULE_PATH = None  # Set this to your score module path

# ====== Configuration parameters ======
FINAL_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "final" / "customization")
FILTER_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "filter" / "customization")
# You can override FILTER_DIR to use a custom path if needed

# Filtering thresholds
SKIP_SCORE = False
CONSISTENCY_SCORE_THRESHOLD = 6.0  # Each score in consistency_scores list must be >= this threshold
FOLLOWING_SCORE_THRESHOLD = 6.0    # following_score must be >= this threshold

# Number of parallel workers
MAX_PARALLEL_WORKERS = 256

# Filter config: {image_count_category: {train: count, eval: count}}
FILTER_CONFIG = {
    "1-3": {"train": 20000, "eval": 250},
    "4-5": {"train": 20000, "eval": 250},
    "6-7": {"train": 30000, "eval": 250},
    ">=8": {"train": 30000, "eval": 250},
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


def get_combination_key_from_sample(sample: Dict[str, Any]) -> Optional[str]:
    """
    Generate combination_key from sample input_images (for customization task)
    
    Args:
        sample: sample data
    
    Returns:
        combination_key, or None if it cannot be generated
    """
    input_images = sample.get("input_images", [])
    if not isinstance(input_images, list) or len(input_images) == 0:
        return None
    
    # Sort file list and generate MD5 hash
    sorted_files = sorted(input_images)
    key_str = "|".join(sorted_files)
    return hashlib.md5(key_str.encode()).hexdigest()


def get_unique_id_from_sample(sample: Dict[str, Any]) -> Optional[str]:
    """
    Get or generate unique_id from sample (for customization task)
    
    Args:
        sample: sample data
    
    Returns:
        unique_id, or None if it cannot be generated
    """
    # Prefer existing unique_id if available
    unique_id = sample.get("unique_id")
    if unique_id:
        return unique_id
    
    # If no unique_id, generate combination_key from input_images, then generate unique_id
    combination_key = get_combination_key_from_sample(sample)
    if combination_key:
        # For customization, unique_id format is customization_{combination_key}
        # May also be MD5-hashed; here we use the raw format for simplicity
        # If MD5 hashing is required, check config; for simplicity we use the raw format
        return f"customization_{combination_key}"
    
    return None


def load_score_module():
    """
    Load the optional reference scoring module.

    Set ``SCORE_MODULE_PATH`` at the top of this file to a valid path
    before calling this function.  When ``SCORE_MODULE_PATH`` is ``None``
    (the default) the function raises ``RuntimeError`` immediately so that
    the caller can fall back to the Gemini-based scorer.

    Returns:
        Loaded score module object.
    """
    if SCORE_MODULE_PATH is None:
        raise RuntimeError(
            "SCORE_MODULE_PATH is not configured. "
            "Set it to the path of your scoring module, or set SKIP_SCORE=True "
            "to skip reference-based scoring."
        )
    score_module_path = Path(SCORE_MODULE_PATH)
    if not score_module_path.exists():
        raise FileNotFoundError(f"Score module not found: {score_module_path}")

    # Ensure the module's parent directory is importable
    module_dir = str(score_module_path.parent)
    # Clean up any previously cached utils modules to avoid import conflicts
    modules_to_remove = [key for key in sys.modules if key.startswith('utils.')]
    for module_name in modules_to_remove:
        del sys.modules[module_name]
    if 'utils' in sys.modules:
        del sys.modules['utils']
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location("score.customization", score_module_path)
    score_module = importlib.util.module_from_spec(spec)
    # Set __file__ so that relative paths inside the module resolve correctly
    score_module.__file__ = str(score_module_path.resolve())
    spec.loader.exec_module(score_module)

    return score_module


def has_score_fields(sample: Dict[str, Any]) -> bool:
    """
    Check whether a sample has scoring fields
    
    Args:
        sample: sample data
    
    Returns:
        Whether the sample has consistency_scores and following_score
    """
    consistency_scores = sample.get("consistency_scores")
    following_score = sample.get("following_score")
    
    # Check whether consistency_scores is a valid list
    if not isinstance(consistency_scores, list) or len(consistency_scores) == 0:
        return False
    
    # Check whether following_score is a valid number
    if following_score is None or not isinstance(following_score, (int, float)):
        return False
    
    return True


def evaluate_sample(json_path: Path, score_module, gemini_generator) -> Optional[Dict[str, Any]]:
    """
    Score a single sample (using Gemini-3-flash)
    
    Args:
        json_path: JSON file path
        score_module: scoring module
        gemini_generator: Gemini generator instance
    
    Returns:
        Score result dict containing consistency_scores, following_score, overall_reasoning
        Returns None on failure
    """
    try:
        # Read sample data
        with open(json_path, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        
        sample_id = json_path.stem
        
        # Use Gemini only for scoring
        if gemini_generator is None:
            print(f"  Error: Gemini generator not initialized")
            return None
        
        evaluate_with_gemini = getattr(score_module, 'evaluate_with_gemini')
        result = evaluate_with_gemini(sample_data, gemini_generator, sample_id)
        
        return result
    except Exception as e:
        print(f"  Error: scoring failed {json_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_score_to_json(json_path: Path, score_result: Dict[str, Any]) -> bool:
    """
    Save scoring results to the JSON file (without overwriting existing fields)
    
    Args:
        json_path: JSON file path
        score_result: score result dict
    
    Returns:
        Whether saving succeeded
    """
    try:
        # Read existing data
        with open(json_path, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        
        # Update scoring fields
        sample_data["consistency_scores"] = score_result.get("consistency_scores", [])
        sample_data["following_score"] = score_result.get("following_score")
        if "overall_reasoning" in score_result:
            sample_data["overall_reasoning"] = score_result.get("overall_reasoning")
        
        # Write back to file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"  Error: failed to save score {json_path}: {e}")
        return False


def check_and_complete_scores(samples: List[Tuple[int, Path, Dict[str, Any]]], 
                               score_module, gemini_generator) -> List[Tuple[int, Path, Dict[str, Any]]]:
    """
    Check and fill missing scores (parallel processing)
    
    Args:
        samples: list of (idx, json_path, sample_data) tuples
        score_module: scoring module
        gemini_generator: Gemini generator instance
    
    Returns:
        Updated sample list (scores filled and saved)
    """
    # Find samples that need scoring
    samples_to_evaluate = []
    for idx, json_path, sample_data in samples:
        if not has_score_fields(sample_data):
            samples_to_evaluate.append((idx, json_path, sample_data))
    
    if not samples_to_evaluate:
        print(f"  All samples already have scores, no need to fill")
        return samples
    
    print(f"  Need to fill scores for {len(samples_to_evaluate)} samples...")
    
    # Parallel processing
    completed = 0
    total = len(samples_to_evaluate)
    lock = threading.Lock()
    save_lock = threading.Lock()
    
    def process_sample(item):
        nonlocal completed
        idx, json_path, sample_data = item
        
        # Perform scoring
        score_result = evaluate_sample(json_path, score_module, gemini_generator)
        
        if score_result is not None:
            # Save score to JSON file
            with save_lock:
                if save_score_to_json(json_path, score_result):
                    # Re-read updated data to ensure it is the latest
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            sample_data = json.load(f)
                    except Exception as e:
                        print(f"  Warning: failed to re-read {json_path}: {e}")
        
        with lock:
            completed += 1
            status = "success" if score_result is not None else "failed"
            print(f"  [{completed}/{total}] {json_path.name}: {status}")
        
        return (idx, json_path, sample_data)
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        futures = {executor.submit(process_sample, item): item for item in samples_to_evaluate}
        
        updated_samples_map = {}
        for future in as_completed(futures):
            idx, json_path, sample_data = future.result()
            updated_samples_map[(idx, json_path)] = (idx, json_path, sample_data)
    
    # Update sample list: only update modified samples; keep others in memory
    updated_samples = []
    for idx, json_path, sample_data in samples:
        if (idx, json_path) in updated_samples_map:
            # If the sample was updated, use the refreshed data (re-read from file)
            updated_samples.append(updated_samples_map[(idx, json_path)])
        else:
            # If the sample was not updated, use in-memory data to avoid unnecessary file I/O
            updated_samples.append((idx, json_path, sample_data))
    
    return updated_samples


def filter_samples(samples: List[Tuple[int, Path, Dict[str, Any]]]) -> List[Tuple[int, Path, Dict[str, Any]]]:
    """
    Filter samples based on threshold
    
    Args:
        samples: list of (idx, json_path, sample_data) tuples
    
    Returns:
        Filtered sample list, keeping (idx, json_path, sample_data) format
    """
    filtered = []
    
    for idx, json_path, sample in samples:
        # Check scoring fields
        if not has_score_fields(sample):
            # Skip if no scoring fields (should have been filled earlier)
            continue
        
        consistency_scores = sample.get("consistency_scores", [])
        following_score = sample.get("following_score")
        
        # Check consistency_scores: each score in the list must be >= threshold
        all_consistency_ok = True
        if isinstance(consistency_scores, list):
            for score in consistency_scores:
                if not isinstance(score, (int, float)) or score < CONSISTENCY_SCORE_THRESHOLD:
                    all_consistency_ok = False
                    break
        else:
            all_consistency_ok = False
        
        # Check following_score
        following_ok = isinstance(following_score, (int, float)) and following_score >= FOLLOWING_SCORE_THRESHOLD
        
        input_num = len(sample.get("input_images", []))
        within_threshold = True
        if input_num > 10:
            print(f"More than 10 input images: {input_num}")
            within_threshold = False
        
        # Both conditions must be satisfied to keep the sample
        if all_consistency_ok and following_ok and within_threshold:
            filtered.append((idx, json_path, sample))
    
    return filtered


def main():
    """Main function"""
    print("=" * 80)
    print("Customization data filtering script")
    print("=" * 80)
    print(f"Final directory: {FINAL_DIR}")
    print(f"Filter directory: {FILTER_DIR}")
    print(f"Consistency Score threshold: {CONSISTENCY_SCORE_THRESHOLD}")
    print(f"Following Score threshold: {FOLLOWING_SCORE_THRESHOLD}")
    print(f"Filter config: {FILTER_CONFIG}")
    print("=" * 80)
    
    if not FINAL_DIR.exists():
        print(f"Error: Final directory does not exist: {FINAL_DIR}")
        return
    
    # Load scoring module and initialize Gemini generator
    print("\nLoading scoring module...")
    try:
        score_module = load_score_module()
        
        # Initialize Gemini generator (using gemini-3-flash-preview for scoring)
        # Note: gemini_generator will be used concurrently by multiple threads; ensure GeminiAPIGenerator is thread-safe
        # If concurrency issues arise, consider creating a separate instance per thread
        GEMINI_CONFIG = {
            "api_key": os.environ.get("GEMINI_API_KEY", ""),
            "model_name": "gemini-3-flash-preview",
            "max_try": 10,
            "timeout": 60
        }
        from api_generator.text_generator.gemini_api import GeminiAPIGenerator
        gemini_generator = GeminiAPIGenerator(
            app_key=GEMINI_CONFIG["api_key"],
            model_name=GEMINI_CONFIG["model_name"],
            max_try=GEMINI_CONFIG["max_try"],
            print_log=False,
            timeout=GEMINI_CONFIG["timeout"]
        )
        print("Scoring module loaded")
    except Exception as e:
        print(f"Error: failed to load scoring module: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Ensure FILTER_DIR exists
    FILTER_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process train and eval data
    # Check if eval data is configured; if so process eval, otherwise train only
    for split_type in ["train", "eval"]:
        print(f"\nProcessing {split_type} data...")
        
        # For train, clear the entire train directory and rebuild
        if split_type == "train":
            train_dir = FILTER_DIR / "train"
            if train_dir.exists():
                print(f"Clearing Train directory: {train_dir}")
                shutil.rmtree(train_dir)
            train_dir.mkdir(parents=True, exist_ok=True)
        
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
            if split_type == "eval" and output_dir.exists():
                existing_files = list(output_dir.glob("*.json"))
                existing_count = len(existing_files)
                if target_count is not None and existing_count >= target_count:
                    print(f"  Eval data already meets target count ({existing_count} >= {target_count}), skipping")
                    continue
                elif existing_count > 0:
                    print(f"  Existing Eval data: {existing_count}, target: {target_count}, need {target_count - existing_count} more")
            
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
                        samples.append((idx, json_file, sample))
                except Exception as e:
                    print(f"  Warning: failed to read JSON file {json_file}: {e}")
                    continue
            
            print(f"  Loaded {len(samples)} samples")
            
            # Check and fill scores
            if not SKIP_SCORE:
                print(f"  Checking and filling scores...")
                samples = check_and_complete_scores(samples, score_module, gemini_generator)
            else:
                print(f"  Skipping score check and fill")
            
            # Filter samples (based on threshold), return list of (idx, json_path, sample) tuples
            # Note: do not filter eval data, use raw samples directly
            if split_type == "eval":
                filtered_with_idx = samples
                print(f"  Eval data skips filtering, using raw samples: {len(filtered_with_idx)} samples")
            else:
                filtered_with_idx = filter_samples(samples)
                print(f"  After filtering: {len(filtered_with_idx)} samples")
            
            # For eval, exclude already existing samples (based on unique_id)
            if split_type == "eval" and existing_count > 0:
                # Read unique_id from existing files
                existing_identifiers = set()
                for existing_file in output_dir.glob("*.json"):
                    try:
                        with open(existing_file, 'r', encoding='utf-8') as f:
                            existing_sample = json.load(f)
                            # For customization, use unique_id as the unique identifier
                            # If no unique_id, generate combination_key from input_images and produce unique_id
                            unique_id = get_unique_id_from_sample(existing_sample)
                            if unique_id:
                                existing_identifiers.add(unique_id)
                    except Exception as e:
                        print(f"  Warning: failed to read existing file {existing_file}: {e}")
                        continue
                
                # Filter out already existing samples
                original_count = len(filtered_with_idx)
                filtered_with_idx = [
                    (idx, json_path, sample) for idx, json_path, sample in filtered_with_idx
                    if get_unique_id_from_sample(sample) not in existing_identifiers
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
                # Note: train directory was fully deleted and rebuilt above; no need to clear files here
                start_idx = 1
            
            # Save samples
            for i, (original_idx, json_path, sample) in enumerate(filtered_with_idx, start=start_idx):
                minimal = convert_to_minimal(sample, "customization", i)
                output_file = output_dir / f"{i:08d}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(minimal, f, ensure_ascii=False, indent=2)
            
            final_count = existing_count + len(filtered_with_idx) if split_type == "eval" and existing_count > 0 else len(filtered_with_idx)
            print(f"  Saved to: {output_dir} (total {final_count} samples)")
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()

