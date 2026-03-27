#!/usr/bin/env python3
"""
Evaluation script

Load data from inference output directory and score using LLM (GPT4o and Gemini-3-flash)
Supports multi-task parallel evaluation and checkpoint resuming
Supports evaluation of multiple baseline models

Outputs directory structure:
  outputs/{baseline}/{exp_name}/{task}/{image_num_category}/
  Example: outputs/bagel/exp_001/customization/1-3/
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# ============================================================================
# Configuration constants
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
MACRO_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = MACRO_DIR / "outputs"

# Add Macro root to path for importing utils
if str(MACRO_DIR) not in sys.path:
    sys.path.insert(0, str(MACRO_DIR))

# Supported task list and image num categories (tasks using LLM scoring)
SUPPORTED_TASKS = ["customization", "illustration", "spatial", "temporal"]
IMAGE_NUM_CATEGORIES = ["1-3", "4-5", "6-7", ">=8"]

# Supported baseline models
SUPPORTED_BASELINES = ["bagel", "omnigen", "qwen"]

# LLM configuration
GPT_CONFIG = {
    "url": os.environ.get("OPENAI_URL", "https://api.openai.com/v1/chat/completions"),
    "key": os.environ.get("OPENAI_KEY", "")
}

GEMINI_CONFIG = {
    "api_key": os.environ.get("GEMINI_API_KEY", ""),
    "model_name": os.environ.get("GEMINI_MODEL_NAME", "gemini-3.0-flash-preview"),
    "max_try": 100
}

# Retry configuration
MAX_RETRIES = 10
RETRY_DELAY = 2
TIMEOUT = 60  # 60-second timeout
PARALLEL_WORKERS = 128  # Number of parallel worker samples


# ============================================================================
# Data loading functions
# ============================================================================
def load_samples_from_output(
    baseline: str,
    exp_name: str,
    task: str,
    image_num_category: str,
    output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load sample data from output directory
    
    Directory structure: {output_root}/{baseline}/{exp_name}/{task}/{image_num_category}/
    When output_root is None, use the default OUTPUT_DIR.
    
    Args:
        baseline: model type (bagel, omnigen, qwen, api)
        exp_name: experiment name (or api name such as gpt/seed/nano)
        task: task type
        image_num_category: image count category
        output_root: output root directory; use default OUTPUT_DIR when None
        
    Returns:
        sample dict, key is idx (string format), value is sample data (contains json_file_path field)
    """
    root = Path(output_root) if output_root is not None else OUTPUT_DIR
    if baseline == "any":
        # Any directory: output_root is the parent of the "result root", exp_name is the result root directory name
        input_dir = root / exp_name / task / image_num_category
    else:
        input_dir = root / baseline / exp_name / task / image_num_category
    
    if not input_dir.exists():
        return {}
    
    samples = {}
    # Only load sample JSONs ({idx:08d}.json), skip non-sample files like metadata.json to avoid overwriting
    for json_file in sorted(input_dir.glob("*.json")):
        if not json_file.stem.isdigit():
            continue
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                sample = json.load(f)
            
            # Use idx as key; some models (e.g. omnigen) only write base_idx not idx, need compatibility
            idx = sample.get("idx")
            if idx is None:
                idx = sample.get("base_idx")
            if idx is None and json_file.stem.isdigit():
                idx = int(json_file.stem)
            if idx is None:
                idx = 0
            # Prefer filename as sample_id to avoid overwriting when multiple JSONs share the same idx or metadata.json
            if json_file.stem.isdigit() and len(json_file.stem) <= 8:
                sample_id = f"{int(json_file.stem):08d}"
            else:
                sample_id = f"{idx:08d}"
            
            # Standardize field names (prompt -> instruction)
            if 'prompt' in sample and 'instruction' not in sample:
                sample['instruction'] = sample['prompt']
            
            row = {
                'sample_id': sample_id,
                'idx': idx,
                'instruction': sample.get('instruction', ''),
                'input_images': sample.get('input_images', []),
                'output_image': sample.get('output_image', ''),
                'target_image': sample.get('target_image', ''),
                'category': sample.get('category', image_num_category),
                'json_file_path': str(json_file),  # Save JSON file path, used for saving score files
                'json_dir': str(json_file.parent)  # Save directory of JSON file
            }
            # If JSON contains relative path, resolve output_image relative to json_dir
            if row['output_image'] and not os.path.isabs(row['output_image']):
                row['output_image'] = str(row['output_image'])
            if row['input_images']:
                resolved = []
                for p in row['input_images']:
                    if p and not os.path.isabs(p):
                        p = str(p)
                    resolved.append(p)
                row['input_images'] = resolved
            samples[sample_id] = row
        except Exception as e:
            print(f"Warning: failed to load JSON file {json_file}: {e}")
            continue
    
    return samples


# ============================================================================
# Main function
# ============================================================================
def load_score_file(score_file: Path) -> Dict[str, Any]:
    """Load a single sample score file, for checkpoint resuming"""
    if not score_file.exists():
        return {}
    
    try:
        with open(score_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: failed to load score file {score_file}: {e}")
        return {}


def save_score_file(score_file: Path, scores: Dict[str, Any]):
    """Save a single sample score file"""
    try:
        with open(score_file, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: failed to save score file {score_file}: {e}")


def run_evaluation(
    baseline: str,
    exp_name: str,
    task: str,
    image_num_category: str = "all",
    output_dir: str = None,
    output_root: str = None,
    use_gpt: bool = True,
    use_gemini: bool = True,
    max_samples: Optional[int] = None,
):
    """
    Run evaluation (supports GPT and Gemini scoring, supports checkpoint resuming)
    
    Args:
        baseline: model type (bagel, omnigen, qwen, api)
        exp_name: experiment name (model checkpoint name, or gpt/seed/nano for api)
        task: task type (customization, illustration, spatial, temporal, all)
        image_num_category: image count category (1-3, 4-5, 6-7, >=8, all)
        output_dir: output directory; default is to save score files in the same directory as samples
        output_root: generation result root directory; use default OUTPUT_DIR when None (for api or non-default paths)
        use_gpt: whether to use GPT scoring
        use_gemini: whether to use Gemini scoring
        max_samples: max samples to evaluate per (task, category); None means all
    """
    # Validate baseline
    if baseline not in SUPPORTED_BASELINES:
        raise ValueError(f"Unsupported baseline: {baseline}. Supported baselines: {SUPPORTED_BASELINES}")
    
    # Determine tasks to process
    if task == "all":
        tasks_to_process = SUPPORTED_TASKS
    else:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}. Supported tasks: {SUPPORTED_TASKS}")
        tasks_to_process = [task]
    
    # Determine categories to process
    if image_num_category == "all":
        categories_to_process = IMAGE_NUM_CATEGORIES
    else:
        if image_num_category not in IMAGE_NUM_CATEGORIES:
            raise ValueError(f"Unsupported image_num_category: {image_num_category}. Supported: {IMAGE_NUM_CATEGORIES}")
        categories_to_process = [image_num_category]
    
    # Process each task and category combination
    for current_task in tasks_to_process:
        for current_category in categories_to_process:
            print("=" * 80)
            print(f"Processing task: {current_task}, image count category: {current_category}")
            print(f"Baseline: {baseline}, experiment: {exp_name}")
            print("=" * 80)
            
            # 1. Load sample data
            samples = load_samples_from_output(
                baseline, exp_name, current_task, current_category,
                output_root=Path(output_root) if output_root else None,
            )
            if not samples:
                print(f"Warning: no sample data found, skipping {current_task}/{current_category}")
                continue
            
            # Optional: only evaluate the first max_samples (sorted by sample_id)
            if max_samples is not None and max_samples > 0:
                sorted_ids = sorted(samples.keys())[:max_samples]
                samples = {sid: samples[sid] for sid in sorted_ids}
                print(f"Limiting evaluation count: {max_samples}, current {len(samples)} samples")
            else:
                print(f"Found {len(samples)} samples")
            
            # Determine JSON file directory (from first sample; all samples should be in the same directory)
            first_sample = next(iter(samples.values()))
            json_dir = Path(first_sample.get('json_dir', ''))
            if not json_dir.exists():
                print(f"Warning: JSON file directory not found: {json_dir}, skipping {current_task}/{current_category}")
                continue
            
            print(f"JSON file directory: {json_dir}")
            
            # Dynamically import the corresponding scoring module (using file path import)
            score_module_path = SCRIPT_DIR / "score" / f"{current_task}.py"
            if not score_module_path.exists():
                print(f"Error: scoring module not found: {score_module_path}")
                continue
            
            # Use importlib to load module
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"score.{current_task}", score_module_path)
            score_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(score_module)
            
            evaluate_with_gpt_func = getattr(score_module, 'evaluate_with_gpt')
            evaluate_with_gemini_func = getattr(score_module, 'evaluate_with_gemini')
            is_score_valid_func = getattr(score_module, 'is_score_valid')
            
            # 3. Initialize Gemini generator (if needed)
            gemini_generator = None
            if use_gemini:
                try:
                    from api_generator.text_generator.gemini_api import GeminiAPIGenerator
                    gemini_generator = GeminiAPIGenerator(
                        app_key=GEMINI_CONFIG["api_key"],
                        model_name=GEMINI_CONFIG["model_name"],
                        max_try=MAX_RETRIES,
                        print_log=False,
                        timeout=TIMEOUT
                    )
                except ImportError as e:
                    print(f"Warning: cannot import GeminiAPIGenerator: {e}, skipping Gemini scoring")
                    use_gemini = False
            
            # 4. Load existing score files (checkpoint resuming)
            existing_scores = {}
            for sample_id in samples.keys():
                score_file = json_dir / f"{sample_id}.score"
                if score_file.exists():
                    existing_scores[sample_id] = load_score_file(score_file)
            
            print(f"Loaded {len(existing_scores)} existing score files")
            
            # Get the list of samples to process (filter out completed ones)
            sample_items = list(samples.items())
            samples_to_process = []
            for sample_id, sample_data in sample_items:
                existing_score = existing_scores.get(sample_id, {})
                gpt_done = is_score_valid_func(existing_score.get('gpt_scores')) if use_gpt else True
                gemini_done = is_score_valid_func(existing_score.get('gemini_scores')) if use_gemini else True
                if gpt_done and gemini_done:
                    continue
                samples_to_process.append((sample_id, sample_data))
            
            print(f"Need to process {len(samples_to_process)} samples")
            
            # 5. GPT scoring (if enabled)
            if use_gpt:
                print("\n" + "-" * 80)
                print("Step 1: GPT4o scoring (parallel processing)")
                print("-" * 80)
                
                gpt_completed = 0
                gpt_total = len(samples_to_process)
                gpt_lock = threading.Lock()
                
                def process_gpt_sample(item):
                    nonlocal gpt_completed
                    sample_id, sample_data = item
                    # Check whether GPT score already exists
                    existing_score = existing_scores.get(sample_id, {})
                    if is_score_valid_func(existing_score.get('gpt_scores')):
                        with gpt_lock:
                            gpt_completed += 1
                        return sample_id, existing_score.get('gpt_scores')
                    
                    gpt_score = None
                    if GPT_CONFIG["key"]:
                        gpt_score = evaluate_with_gpt_func(sample_data, sample_id)
                    
                    with gpt_lock:
                        gpt_completed += 1
                        print(f"[GPT] [{gpt_completed}/{gpt_total}] {sample_id}: done")
                    
                    return sample_id, gpt_score
                
                # Parallel GPT scoring
                save_lock = threading.Lock()
                with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                    gpt_futures = {executor.submit(process_gpt_sample, item): item for item in samples_to_process}
                    
                    for future in as_completed(gpt_futures):
                        sample_id, gpt_score = future.result()
                        
                        # Load or create score file (saved in the same directory as JSON file)
                        score_file = json_dir / f"{sample_id}.score"
                        with save_lock:
                            current_scores = load_score_file(score_file)
                            if not current_scores:
                                current_scores = {}
                            
                            # Update GPT score (only save when score is valid, distinguishing 0 from None)
                            if is_score_valid_func(gpt_score):
                                # Score is valid (including 0), save the score
                                current_scores['gpt_scores'] = gpt_score
                                # If using Gemini, preserve existing gemini_scores
                                if use_gemini and 'gemini_scores' not in current_scores:
                                    existing_score = existing_scores.get(sample_id, {})
                                    if is_score_valid_func(existing_score.get('gemini_scores')):
                                        current_scores['gemini_scores'] = existing_score.get('gemini_scores')
                                # Immediately save to .score file
                                save_score_file(score_file, current_scores)
                            elif gpt_score is None and not is_score_valid_func(current_scores.get('gpt_scores')):
                                # Scoring failed and no valid score currently exists; do not save None, keep as-is
                                pass
                
                print(f"\nGPT score results saved to corresponding .score files")
            
            # 6. Gemini scoring (if enabled)
            if use_gemini and gemini_generator:
                print("\n" + "-" * 80)
                print("Step 2: Gemini scoring (parallel processing)")
                print("-" * 80)
                
                gemini_completed = 0
                gemini_total = len(samples_to_process)
                gemini_lock = threading.Lock()
                
                def process_gemini_sample(item):
                    nonlocal gemini_completed
                    sample_id, sample_data = item
                    # Check whether Gemini score already exists
                    existing_score = existing_scores.get(sample_id, {})
                    if is_score_valid_func(existing_score.get('gemini_scores')):
                        with gemini_lock:
                            gemini_completed += 1
                        return sample_id, existing_score.get('gemini_scores')
                    
                    gemini_score = evaluate_with_gemini_func(sample_data, gemini_generator, sample_id)
                    
                    with gemini_lock:
                        gemini_completed += 1
                        print(f"[Gemini] [{gemini_completed}/{gemini_total}] {sample_id}: done")
                    
                    return sample_id, gemini_score
                
                # Parallel Gemini scoring
                save_lock = threading.Lock()
                with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                    gemini_futures = {executor.submit(process_gemini_sample, item): item for item in samples_to_process}
                    
                    for future in as_completed(gemini_futures):
                        sample_id, gemini_score = future.result()
                        
                        # Load or create score file (saved in the same directory as JSON file)
                        score_file = json_dir / f"{sample_id}.score"
                        with save_lock:
                            current_scores = load_score_file(score_file)
                            if not current_scores:
                                current_scores = {}
                            
                            # Update Gemini score (only save when valid, distinguishing 0 from None)
                            if is_score_valid_func(gemini_score):
                                # Score is valid (including 0), save the score
                                current_scores['gemini_scores'] = gemini_score
                                # If using GPT, preserve existing gpt_scores
                                if use_gpt and 'gpt_scores' not in current_scores:
                                    existing_score = existing_scores.get(sample_id, {})
                                    if is_score_valid_func(existing_score.get('gpt_scores')):
                                        current_scores['gpt_scores'] = existing_score.get('gpt_scores')
                                # Immediately save to .score file
                                save_score_file(score_file, current_scores)
                            elif gemini_score is None and not is_score_valid_func(current_scores.get('gemini_scores')):
                                # Scoring failed and no valid score currently exists; do not save None, keep as-is
                                pass
                
                print(f"\nGemini score results saved to corresponding .score files")
            
            # 7. Summarize results
            total_samples = len(samples)
            gpt_success = 0
            gemini_success = 0
            for sample_id in samples.keys():
                score_file = json_dir / f"{sample_id}.score"
                if score_file.exists():
                    scores = load_score_file(score_file)
                    if use_gpt and is_score_valid_func(scores.get('gpt_scores')):
                        gpt_success += 1
                    if use_gemini and is_score_valid_func(scores.get('gemini_scores')):
                        gemini_success += 1
            print(f"\nSummary: total_samples={total_samples}, GPT success={gpt_success}, Gemini success={gemini_success}")
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Multi-baseline model evaluation script (supports GPT and Gemini scoring, supports checkpoint resuming)")
    
    parser.add_argument("--baseline", type=str, required=True,
                       choices=SUPPORTED_BASELINES,
                       help="model type (bagel, omnigen, qwen)")
    parser.add_argument("--exp_name", type=str, required=True,
                       help="experiment name (model checkpoint name), corresponding to subdirectory under outputs/{baseline}/")
    parser.add_argument("--task", type=str, default="all",
                       choices=SUPPORTED_TASKS + ["all"],
                       help="task type")
    parser.add_argument("--image_num_category", type=str, default="all",
                       choices=IMAGE_NUM_CATEGORIES + ["all"],
                       help="image count category")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="output directory; default saves score files in the same directory as samples")
    parser.add_argument("--output_root", type=str, default=None,
                       help="generation result root directory, default is Macro/outputs; for api evaluation, must match the output_root of the run script")
    parser.add_argument("--use_gpt", action="store_true", default=True,
                       help="whether to use GPT scoring (default True)")
    parser.add_argument("--no_gpt", dest="use_gpt", action="store_false",
                       help="do not use GPT scoring")
    parser.add_argument("--use_gemini", action="store_true", default=True,
                       help="whether to use Gemini scoring (default True)")
    parser.add_argument("--no_gemini", dest="use_gemini", action="store_false",
                       help="do not use Gemini scoring")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="max samples to evaluate per (task, image_num_category); if not set, evaluate all (for quick test runs)")
    
    args = parser.parse_args()
    
    # Run evaluation
    run_evaluation(
        baseline=args.baseline,
        exp_name=args.exp_name,
        task=args.task,
        image_num_category=args.image_num_category,
        output_dir=args.output_dir,
        output_root=args.output_root,
        use_gpt=args.use_gpt,
        use_gemini=args.use_gemini,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
