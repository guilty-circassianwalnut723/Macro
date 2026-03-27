#!/usr/bin/env python3
"""
Temporal data generation script

Features:
1. Read train/eval data from split/temporal directory
2. Extract frame images
3. Call LLM to generate summary and temporal_score
4. Save to final/temporal/{train/eval}/{image_count_category}/data and json directories
5. Support unique IDs to avoid duplicate generation
"""

import json
import os
import random
import re
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image

try:
    from decord import VideoReader, cpu
except ImportError:
    raise ImportError("The decord library is required: pip install decord")

# Add utils path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from utils.common import (
    get_image_count_category,
    load_generated_ids,
    save_sample_data,
    generate_unique_id
)

# Ensure project root is in sys.path
MACRO_DIR = CURRENT_DIR.parent.parent
if str(MACRO_DIR) not in sys.path:
    sys.path.insert(0, str(MACRO_DIR))

from api_generator.text_generator.gemini_api import GeminiAPIGenerator

# ====== Configuration parameters ======
DATA_DIR = MACRO_DIR / "data"
SPLIT_DIR = DATA_DIR / "split" / "temporal"
FINAL_DIR = DATA_DIR / "final" / "temporal"

# Gemini API configuration (set via environment variables)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_API_URL = os.environ.get("GEMINI_API_URL", "")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# Generation config: {image_count_category: {train: count, eval: count}}
GEN_CONFIG = {
    "1-3": {"train": 43000, "eval": 500},
    "4-5": {"train": 30000, "eval": 500},
    "6-7": {"train": 30000, "eval": 500},
    ">=8": {"train": 35000, "eval": 500},
}

# Thread configuration
MAX_WORKERS = 64
MAX_TRIES = 3

# Random seed
RANDOM_SEED = 42

# Logging configuration
LOG_TO_SHELL = False  # Whether to output logs to shell
# ======================

# ====== Thread-local storage ======
thread_local = threading.local()

# ====== Unique ID check lock ======
unique_id_lock = threading.Lock()
# =============================


def get_or_create_generator(
    gemini_api_key: str,
    gemini_model_name: str,
    max_try: int
) -> GeminiAPIGenerator:
    """
    Get or create thread-local generator

    Args:
        gemini_api_key: Gemini API key
        gemini_model_name: Gemini model name
        max_try: max number of retries

    Returns:
        text generator
    """
    if not hasattr(thread_local, 'generator'):
        thread_local.generator = GeminiAPIGenerator(
            app_key=gemini_api_key,
            app_url=GEMINI_API_URL,
            model_name=gemini_model_name,
            max_try=max_try,
            print_log=LOG_TO_SHELL
        )
    return thread_local.generator


def load_json(json_path: Path) -> List[dict]:
    """Load JSON file (list format)"""
    if not json_path.exists():
        return []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Warning: failed to load file {json_path}: {e}")
        return []


def load_jsonl(jsonl_path: Path) -> List[dict]:
    """Load jsonl file (compatible with old format)"""
    samples = []
    if not jsonl_path.exists():
        return samples
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'error' not in data:
                    samples.append(data)
            except json.JSONDecodeError:
                continue
    return samples


def load_split_data(split_dir: Path, split_type: str) -> List[Dict]:
    """Load data from split directory"""
    all_samples = []

    json_file = split_dir / f"{split_type}.json"
    if json_file.exists():
        return load_json(json_file)

    json_files = sorted(split_dir.glob(f"{split_type}_*.json"))
    if json_files:
        for jf in json_files:
            all_samples.extend(load_json(jf))
        return all_samples

    jsonl_file = split_dir / f"{split_type}.jsonl"
    if jsonl_file.exists():
        return load_jsonl(jsonl_file)

    jsonl_files = sorted(split_dir.glob(f"{split_type}_*.jsonl"))
    for jf in jsonl_files:
        all_samples.extend(load_jsonl(jf))

    return all_samples


def generate_summary(
    frames: List[Image.Image],
    text_generator: GeminiAPIGenerator,
) -> Optional[Tuple[str, float]]:
    """Generate summary and temporal score using Gemini"""
    prompt = """You are given a sequence of video frames. Please complete the following two tasks:

1. Write a concise summary describing what happens in this video sequence. The summary should be clear and descriptive, focusing on the main actions and events shown in the frames. The summary must be less than 50 words.

   IMPORTANT: The summary should start directly with the description of the content. Do NOT include phrases like "The video sequence depicts", "This video sequence shows", or similar introductory phrases. Start directly with the actual content description (e.g., "a person walking" instead of "The video sequence depicts a person walking").

2. Evaluate how well this video sequence demonstrates temporal information (i.e., shows clear changes, actions, or progression over time). Give a score from 1 to 10, where:
   - 1-3: Little to no temporal information (static scenes, minimal changes)
   - 4-6: Some temporal information (moderate changes, some progression)
   - 7-8: Good temporal information (clear changes, noticeable progression)
   - 9-10: Excellent temporal information (very clear temporal dynamics, strong progression)

Please respond with a JSON object containing "summary" and "temporal_score" fields."""

    response_format = {
        "summary": "str",
        "temporal_score": "float"
    }

    try:
        response = text_generator.gen_response(
            prompt=prompt,
            response_format=response_format,
            images=frames,
            think=False
        )

        if response is None or not isinstance(response, dict):
            return None

        summary = response.get("summary", "").strip()
        if not summary:
            return None

        temporal_score = response.get("temporal_score")
        if temporal_score is None:
            temporal_score = 5.0
        else:
            try:
                temporal_score = float(temporal_score)
                temporal_score = max(1.0, min(10.0, temporal_score))
            except (ValueError, TypeError):
                temporal_score = 5.0

        return (summary, temporal_score)
    except Exception as e:
        print(f"Failed to generate summary: {e}")
        return None


def build_prompt(summary: str, image_num: int) -> str:
    """Build prompt"""
    image_placeholders = ", ".join([f"<image {i+1}>" for i in range(image_num)])
    cleaned_summary = summary.strip()

    patterns_to_remove = [
        r"^the video sequence depicts\s+",
        r"^the video sequence\s+depicts\s+",
        r"^this video sequence depicts\s+",
        r"^this video sequence\s+depicts\s+",
    ]
    original_summary = cleaned_summary
    for pattern in patterns_to_remove:
        cleaned_summary = re.sub(pattern, "", cleaned_summary, flags=re.IGNORECASE)

    if cleaned_summary != original_summary and cleaned_summary:
        cleaned_summary = cleaned_summary[0].lower() + cleaned_summary[1:] if len(cleaned_summary) > 1 else cleaned_summary.lower()

    return f"The video sequence {image_placeholders} depicts {cleaned_summary} Generate the next key frame for this video."


def process_sample(
    sample: Dict,
    final_dir: Path,
    split_type: str,
    image_count_category: str,
    idx: int,
    generated_ids: Set[str],
    text_generator: GeminiAPIGenerator
) -> bool:
    """Process a single sample"""
    yt_file = sample.get('yt_file', '')
    yt_line = sample.get('yt_line', -1)
    timestamps = sample.get('timestamps', [])
    part_idx = sample.get('part_idx', 0)

    if not timestamps or len(timestamps) < 2:
        return False

    input_num = len(timestamps)
    if input_num > 11:
        if split_type == "train":
            return False
        else:
            timestamps = timestamps[:11]

    # Generate unique ID
    from utils.common import SAVE_ORIGINAL_STRING
    unique_id_result = generate_unique_id(
        "temporal",
        return_original=SAVE_ORIGINAL_STRING,
        yt_file=yt_file,
        yt_line=yt_line,
        part_idx=part_idx
    )
    unique_id = unique_id_result[0] if isinstance(unique_id_result, tuple) else unique_id_result

    with unique_id_lock:
        if unique_id in generated_ids:
            return False
        generated_ids.add(unique_id)

    # Get video path
    video_path_str = sample.get('video_path')
    if video_path_str:
        video_path = Path(video_path_str)
        if not video_path.exists():
            with unique_id_lock:
                generated_ids.discard(unique_id)
            return False
    else:
        with unique_id_lock:
            generated_ids.discard(unique_id)
        return False

    # Extract frames
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        video_length = len(vr)

        valid_indices = [fi for fi in timestamps if fi < video_length]
        if not valid_indices:
            with unique_id_lock:
                generated_ids.discard(unique_id)
            return False

        batch_frames = vr.get_batch(valid_indices).asnumpy()
        frames = [Image.fromarray(frame) for frame in batch_frames]
    except Exception as e:
        with unique_id_lock:
            generated_ids.discard(unique_id)
        return False

    if len(frames) < 2:
        with unique_id_lock:
            generated_ids.discard(unique_id)
        return False

    # Generate summary and temporal_score
    result = generate_summary(frames, text_generator)
    if not result:
        with unique_id_lock:
            generated_ids.discard(unique_id)
        return False

    input_frames = frames[:-1]
    output_frame = frames[-1]
    summary, temporal_score = result

    prompt = build_prompt(summary, len(input_frames))

    image_files = {}
    for i, frame in enumerate(input_frames):
        image_files[f"image_{i+1}.jpg"] = frame
    image_files["image_output.jpg"] = output_frame

    data_dir = final_dir / split_type / image_count_category / "data" / f"{idx:08d}"
    input_image_paths = [str(data_dir / f"image_{i+1}.jpg") for i in range(len(input_frames))]
    output_image_path = str(data_dir / "image_output.jpg")

    json_data = {
        'yt_file': yt_file,
        'yt_line': yt_line,
        'part_idx': part_idx,
        'timestamps': timestamps,
        'image_num': len(input_frames),
        'summary': summary,
        'temporal_score': temporal_score,
        'prompt': prompt,
        'input_images': input_image_paths,
        'output_image': output_image_path
    }

    success = save_sample_data(
        final_dir,
        split_type,
        image_count_category,
        idx,
        unique_id_result,
        json_data,
        image_files
    )

    if not success:
        with unique_id_lock:
            generated_ids.discard(unique_id)

    return success


def worker_task(
    idx: int,
    sample: Dict,
    final_dir: Path,
    split_type: str,
    image_count_category: str,
    stop_event: threading.Event,
    gemini_api_key: str,
    gemini_model_name: str,
    max_try: int,
    generated_ids: Set[str]
) -> bool:
    """Worker thread task"""
    if stop_event.is_set():
        return False

    text_generator = get_or_create_generator(gemini_api_key, gemini_model_name, max_try)
    return process_sample(
        sample, final_dir, split_type, image_count_category,
        idx, generated_ids, text_generator
    )


def process_split_data(
    split_dir: Path,
    final_dir: Path,
    split_type: str,
    image_count_category: str,
    target_count: int,
    generated_ids: Set[str]
) -> None:
    """Process split data and generate final data"""
    samples = load_split_data(split_dir, split_type)

    filtered_samples = [
        s for s in samples
        if get_image_count_category(s.get('image_num', 0)) == image_count_category
    ]
    print(f"Found {len(filtered_samples)} samples matching {image_count_category}")

    random.seed(RANDOM_SEED)
    random.shuffle(filtered_samples)

    stop_event = threading.Event()
    current_idx = len(generated_ids)
    completed_count = len(generated_ids)
    total_submitted = completed_count

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            sample_idx = 0

            with tqdm(total=target_count, desc=f"{split_type}/{image_count_category}", unit="sample") as pbar:
                pbar.update(completed_count)

                while len(futures) < MAX_WORKERS * 2 and sample_idx < len(filtered_samples) and completed_count < target_count:
                    if not stop_event.is_set():
                        future = executor.submit(
                            worker_task, current_idx, filtered_samples[sample_idx],
                            final_dir, split_type, image_count_category, stop_event,
                            GEMINI_API_KEY, GEMINI_MODEL_NAME, MAX_TRIES, generated_ids
                        )
                        futures.append(future)
                        current_idx += 1
                        sample_idx += 1
                        total_submitted += 1

                while completed_count < target_count and not stop_event.is_set():
                    done_futures = []
                    for future in futures:
                        if future.done():
                            try:
                                if future.result():
                                    completed_count += 1
                                    pbar.update(1)
                                    if completed_count >= target_count:
                                        stop_event.set()
                                        break
                            except Exception as e:
                                print(f"Task execution exception: {e}")
                            done_futures.append(future)

                    for f in done_futures:
                        futures.remove(f)

                    while len(futures) < MAX_WORKERS * 2 and sample_idx < len(filtered_samples) and completed_count < target_count:
                        if stop_event.is_set():
                            break
                        future = executor.submit(
                            worker_task, current_idx, filtered_samples[sample_idx],
                            final_dir, split_type, image_count_category, stop_event,
                            GEMINI_API_KEY, GEMINI_MODEL_NAME, MAX_TRIES, generated_ids
                        )
                        futures.append(future)
                        current_idx += 1
                        sample_idx += 1
                        total_submitted += 1

                    if sample_idx >= len(filtered_samples) and len(futures) == 0:
                        break
                    time.sleep(0.1)

                for future in as_completed(futures):
                    try:
                        if future.result() and completed_count < target_count:
                            completed_count += 1
                            pbar.update(1)
                    except Exception as e:
                        print(f"Task execution exception: {e}")

    except KeyboardInterrupt:
        print("\nInterrupt signal received, stopping...")
        stop_event.set()
        raise

    print(f"\n{split_type}/{image_count_category} done: {completed_count}/{target_count}")


def main():
    """Main function"""
    print("=" * 80)
    print("Temporal data generation script")
    print("=" * 80)
    print(f"Split directory: {SPLIT_DIR}")
    print(f"Final directory: {FINAL_DIR}")
    print(f"Generation config: {GEN_CONFIG}")
    print("=" * 80)

    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY is not set, please set it via environment variable")
    if not GEMINI_API_URL:
        print("Warning: GEMINI_API_URL is not set, please set it via environment variable")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    for split_type in ["train", "eval"]:
        print(f"\nProcessing {split_type} data...")
        for image_count_category, config in GEN_CONFIG.items():
            target_count = config.get(split_type, 0)
            if target_count <= 0:
                print(f"Skipping {split_type}/{image_count_category}")
                continue
            generated_ids = load_generated_ids(FINAL_DIR, split_type, image_count_category)
            print(f"Loaded {len(generated_ids)} already-generated sample IDs")
            process_split_data(
                split_dir=SPLIT_DIR,
                final_dir=FINAL_DIR,
                split_type=split_type,
                image_count_category=image_count_category,
                target_count=target_count,
                generated_ids=generated_ids
            )

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
