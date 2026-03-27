#!/usr/bin/env python3
"""
Customization data generation script

Features:
1. Read train/eval data from split/customization directory
2. Generate images using Gemini
3. Save to final/customization/{train/eval}/{image_count_category}/data and json directories
4. Support unique IDs to avoid duplicate generation

Usage:
    python customization.py

Configuration:
    - Set SPLIT_DIR to the split/customization directory path
    - Set FINAL_DIR to the final/customization directory path
    - Set GEMINI_API_KEY via environment variable or modify the config directly
    - Set GEMINI_API_URL and IMAGE_API_URL environment variables
"""

import json
import hashlib
import os
import random
import sys
import time
import threading
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image

# Add utils path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from utils.common import (
    get_image_count_category,
    load_generated_ids,
    save_sample_data,
    generate_unique_id,
    get_combination_key
)

# Ensure project root is in sys.path
ROOT_DIR = CURRENT_DIR
while ROOT_DIR != ROOT_DIR.parent:
    if (ROOT_DIR / 'api_generator').exists():
        break
    ROOT_DIR = ROOT_DIR.parent
else:
    ROOT_DIR = CURRENT_DIR.parent.parent

ROOT_DIR = ROOT_DIR.resolve()
ROOT_DIR_STR = str(ROOT_DIR)
if ROOT_DIR_STR not in sys.path:
    sys.path.insert(0, ROOT_DIR_STR)

from api_generator.text_generator.gemini_api import GeminiAPIGenerator
from api_generator.image_generator.nano_api import NanoAPIGenerator

# ====== Configuration parameters ======
# Change to your actual paths
MACRO_DIR = CURRENT_DIR.parent.parent  # Macro root directory
DATA_DIR = MACRO_DIR / "data"
SPLIT_DIR = DATA_DIR / "split" / "customization"
FINAL_DIR = DATA_DIR / "final" / "customization"

# Gemini API configuration
# Set API Key and URL via environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# Image generation API configuration
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY", "")
IMAGE_API_URL = os.getenv("IMAGE_API_URL", "")
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "gemini-image-preview")

# Generation config: {image_count_category: {train: count, eval: count}}
GEN_CONFIG = {
    "1-3": {"train": 20000, "eval": 250},
    "4-5": {"train": 20000, "eval": 250},
    "6-7": {"train": 30000, "eval": 250},
    ">=8": {"train": 60000, "eval": 250},
}

# Thread configuration
MAX_WORKERS = 128
MAX_TRIES = 3
MAX_ATTEMPTS = 10  # Max attempts per sample
PASS_THRESHOLD = 8  # Pass threshold

# Random seed
RANDOM_SEED = 42

# Data sampling ratio config (human, cloth, object, scene, style)
DATA_RATIO = [0.4, 0.1, 0.3, 0.15, 0.05]

# Logging configuration
LOG_TO_SHELL = False  # Whether to output logs to shell
# ======================

# ====== Thread-local storage ======
thread_local = threading.local()
# =============================

# ====== Global progress tracking variables ======
progress_lock = threading.Lock()
progress_stats = {
    'completed': 0,
    'failed': 0,
    'total': 0,
    'start_time': None
}
# =============================

# ====== Global combination dict lock ======
combination_lock = threading.Lock()

# ====== Unique ID check lock ======
unique_id_lock = threading.Lock()
# =============================

# ====== Prompt template ======
GEN_PROMPT = """
You will receive several reference images. Each image is tagged as exactly one of the following categories: {human, cloth, object, scene, style}.

Your task is to return ONE valid JSON object with:
{
  "suitable": <integer from 0 to 10>,
  "image_descriptions": ["image 1: ...", "image 2: ...", ...],
  "instruction": "..."
}

### Part 0 — suitable
- First, judge whether it is natural to combine these images into a single scene.
- Rate "suitable" as an **integer from 0 to 10**:
  - **0-3:** The combination would be very unnatural, conflicting, or incoherent.
  - **4-6:** The combination is somewhat natural but has some issues.
  - **7-9:** The combination is mostly natural and coherent.
  - **10:** The combination is perfectly natural and coherent.
- Only consider whether the combination of the concepts are appropriate, do not consider the specific visual details of the images.

### Part 1 — image_descriptions
For each image i (1-indexed):
- Write a concise, general description like "a young woman", "an elderly man", "a blue jacket", "a wooden chair", "a mountain landscape", "an oil-painting style".
- Keep each description under 8 words.

### Part 2 — instruction
- Compose a natural, fluent instruction that guides an image generator to synthesize one cohesive image using all the references.
- Reference each image with the placeholder "<image n>".
- Keep the entire instruction within **100 words**.

Now generate the JSON for the following references:
"""
# ======================


def get_sample_rng(sample_id: int, global_seed: int = RANDOM_SEED) -> random.Random:
    rng = random.Random(global_seed + sample_id)
    return rng


def load_combination_dict(final_dir: Path, split_type: str, image_count_category: str) -> Dict:
    save_dir = final_dir / split_type / image_count_category
    combo_next_file = save_dir / "combination_dict_next.json"
    if combo_next_file.exists():
        try:
            with open(combo_next_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and len(data) > 0:
                    print(f"Loaded {len(data)} combination records from combination_dict_next.json")
                    return data
        except Exception as e:
            print(f"Warning: failed to load combination_dict_next.json: {e}")

    combo_file = save_dir / "combination_dict.json"
    if combo_file.exists():
        try:
            with open(combo_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    print(f"Loaded {len(data)} combination records from combination_dict.json")
                    return data
        except Exception as e:
            print(f"Warning: failed to load combination dict: {e}")
    return {}


def save_combination_dict(combination_dict: Dict, final_dir: Path, split_type: str, image_count_category: str) -> None:
    save_dir = final_dir / split_type / image_count_category
    save_dir.mkdir(parents=True, exist_ok=True)
    combo_file = save_dir / "combination_dict.json"
    combo_next_file = save_dir / "combination_dict_next.json"

    try:
        if combo_next_file.exists():
            is_valid = False
            prev_data = {}
            try:
                with open(combo_next_file, 'r', encoding='utf-8') as f:
                    prev_data = json.load(f)
                    if isinstance(prev_data, dict) and len(prev_data) > 0:
                        current_dict = {}
                        if combo_file.exists():
                            try:
                                with open(combo_file, 'r', encoding='utf-8') as f2:
                                    current_dict = json.load(f2)
                                    if not isinstance(current_dict, dict):
                                        current_dict = {}
                            except Exception:
                                current_dict = {}
                        if len(prev_data) >= len(current_dict):
                            is_valid = True
            except Exception as e:
                print(f"Warning: failed to read previous combination_dict_next.json: {e}")
                is_valid = False

            if is_valid:
                shutil.copy2(combo_next_file, combo_file)
                print(f"Updated combination_dict.json with previously saved data ({len(prev_data)} records)")

        with combination_lock:
            combination_dict_copy = dict(combination_dict)
            for key, value in combination_dict_copy.items():
                if isinstance(value, dict):
                    combination_dict_copy[key] = dict(value)

        with open(combo_next_file, 'w', encoding='utf-8') as f:
            json.dump(combination_dict_copy, f, ensure_ascii=False, indent=2)

        print(f"Saved combination_dict_next.json ({len(combination_dict_copy)} records)")
    except Exception as e:
        print(f"Warning: failed to save combination dict: {e}")


def get_or_create_logger(thread_id: int, save_root: Path) -> logging.Logger:
    if not hasattr(thread_local, 'logger'):
        logger = logging.getLogger(f"thread_{thread_id}")
        logger.setLevel(logging.INFO)

        log_dir = save_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"thread_{thread_id}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if LOG_TO_SHELL:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        thread_local.logger = logger

    return thread_local.logger


def get_or_create_generators(
    gemini_api_key: str,
    gemini_api_url: str,
    gemini_model_name: str,
    image_api_key: str,
    image_api_url: str,
    image_model_name: str,
    max_try: int
) -> Tuple[GeminiAPIGenerator, NanoAPIGenerator]:
    if not hasattr(thread_local, 'text_generator'):
        gen = GeminiAPIGenerator(
            app_key=gemini_api_key,
            model_name=gemini_model_name,
            max_try=max_try,
            print_log=LOG_TO_SHELL
        )
        gen.usage_app_url = gemini_api_url
        thread_local.text_generator = gen

    if not hasattr(thread_local, 'image_generator'):
        img_gen = NanoAPIGenerator(
            api_key=image_api_key,
            model_name=image_model_name,
            max_try=max_try,
            print_log=LOG_TO_SHELL
        )
        img_gen.base_url = image_api_url.rstrip("/") if image_api_url else ""
        img_gen.usage_app_url = f"{img_gen.base_url}/{img_gen.model_name}:imageGenerate" if img_gen.base_url else ""
        thread_local.image_generator = img_gen

    return thread_local.text_generator, thread_local.image_generator


def retry_with_generator_update(
    operation_func,
    operation_name: str,
    logger: logging.Logger,
    sample_id: int,
    max_retries: int = 5,
    wait_time: int = 10
):
    for attempt in range(max_retries):
        try:
            result = operation_func()
            return result
        except Exception as e:
            logger.warning(f"Sample {sample_id} {operation_name} attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                logger.error(f"Sample {sample_id} {operation_name} all retries failed")
                raise
    return None


def load_json(json_path: Path) -> List[Dict]:
    if not json_path.exists():
        return []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return []
    except Exception as e:
        print(f"Warning: failed to read JSON file {json_path}: {e}")
        return []


def load_jsonl(jsonl_path: Path) -> List[Dict]:
    if not jsonl_path.exists():
        return []
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                samples.append(data)
            except json.JSONDecodeError:
                continue
    return samples


def load_split_data(split_dir: Path, split_type: str, category: str) -> List[Dict]:
    json_path = split_dir / f"{category}_{split_type}.json"
    if json_path.exists():
        return load_json(json_path)
    jsonl_path = split_dir / f"{category}_{split_type}.jsonl"
    if jsonl_path.exists():
        return load_jsonl(jsonl_path)
    return []


def organize_data_by_category(samples: List[Dict]) -> Dict[str, List[str]]:
    organized = {
        'human': [],
        'cloth': [],
        'object': [],
        'scene': [],
        'style': []
    }
    for sample in samples:
        files = []
        if 'files' in sample:
            files = [f if isinstance(f, str) else f.get('filepath', '') for f in sample['files']]
        elif 'filepath' in sample:
            files = [sample['filepath']]
        category = sample.get('class', '').lower()
        if category in organized:
            organized[category].extend(files)
    return organized


def worker_task(
    idx: int,
    data: Dict[str, List[str]],
    data_ratio: List[float],
    image_count_range: List[int],
    save_root: Path,
    split_type: str,
    image_count_category: str,
    instruction_gen_prompt: str,
    stop_event: threading.Event,
    gemini_api_key: str,
    gemini_api_url: str,
    gemini_model_name: str,
    image_api_key: str,
    image_api_url: str,
    image_model_name: str,
    max_try: int,
    combination_dict: Dict,
    generated_ids: Set[str],
    scene_groups: Optional[Dict[str, List[str]]] = None,
    global_seed: int = RANDOM_SEED,
    max_attempts: int = MAX_ATTEMPTS,
    pass_threshold: int = PASS_THRESHOLD,
    min_image: int = 1,
    max_image: int = 10
) -> Optional[Dict]:
    if stop_event.is_set():
        return None

    logger = get_or_create_logger(threading.get_ident(), save_root)
    text_generator, image_generator = get_or_create_generators(
        gemini_api_key, gemini_api_url, gemini_model_name,
        image_api_key, image_api_url, image_model_name, max_try
    )

    categories_list = ['human', 'cloth', 'object', 'scene', 'style']
    total_attempts = 0
    valid_attempts = 0
    best_sample = None
    best_score = -1

    while valid_attempts < max_attempts:
        total_attempts += 1
        if stop_event.is_set():
            return None
        selected_files = []
        selected_categories = []
        select_num = [0, 0, 0, 0, 0]

        attempt_seed_str = f"{global_seed}_{idx}_{total_attempts}"
        attempt_seed_hash = int(hashlib.md5(attempt_seed_str.encode()).hexdigest()[:8], 16)
        attempt_seed = (global_seed + idx * 1000 + total_attempts + attempt_seed_hash) % (2**31)
        attempt_rng = random.Random(attempt_seed)

        item_num = attempt_rng.choice(image_count_range)

        while len(selected_files) < item_num:
            category = attempt_rng.choices(list(data.keys()), weights=data_ratio)[0]
            cat_idx = categories_list.index(category)

            if category == "style" and select_num[4] >= 1:
                continue
            if category == "scene" and select_num[3] >= 1:
                continue
            if category == "cloth" and select_num[1] >= select_num[0]:
                continue

            selected_file = None
            if category == "scene" and scene_groups:
                scene_indices = list(scene_groups.keys())
                if scene_indices:
                    selected_scene_idx = attempt_rng.choice(scene_indices)
                    scene_frames = scene_groups[selected_scene_idx]
                    if scene_frames:
                        selected_file = attempt_rng.choice(scene_frames)
                if not selected_file and data.get(category):
                    selected_file = attempt_rng.choice(data[category])
            else:
                if data.get(category):
                    selected_file = attempt_rng.choice(data[category])

            if selected_file:
                selected_files.append(selected_file)
                selected_categories.append(category)
                select_num[cat_idx] += 1

        if len(selected_files) == 0 or len(selected_files) < min_image:
            continue

        combination_key = get_combination_key(selected_files)
        with combination_lock:
            if combination_key in combination_dict:
                combo_info = combination_dict[combination_key]
                if isinstance(combo_info, dict) and combo_info.get('used', False):
                    continue
                if isinstance(combo_info, dict) and 'score' in combo_info and 'prompt' in combo_info:
                    suitable_score = combo_info['score']
                    response = {
                        'suitable': suitable_score,
                        'image_descriptions': combo_info.get('image_descriptions', []),
                        'instruction': combo_info['prompt']
                    }
                else:
                    continue
            else:
                response = None

        if response is None:
            data_category_prefix = "{" + ", ".join([f"image {i+1}: {category}" for i, category in enumerate(selected_categories)]) + "}"
            full_prompt = instruction_gen_prompt + data_category_prefix

            try:
                images = [Image.open(Path(f)) for f in selected_files]
            except Exception as e:
                logger.warning(f"[Sample {idx}] Failed to load image: {e}")
                continue

            def text_generation_operation():
                resp = text_generator.gen_response(
                    prompt=full_prompt,
                    response_format={
                        "suitable": "int",
                        "image_descriptions": "list",
                        "instruction": "str"
                    },
                    images=images,
                    think=False
                )
                if resp is None:
                    return None
                suitable_score = resp.get("suitable", 0)
                if not isinstance(suitable_score, int):
                    try:
                        suitable_score = int(suitable_score)
                    except:
                        suitable_score = 0
                if suitable_score < 0 or suitable_score > 10:
                    suitable_score = 0
                resp["suitable"] = suitable_score
                if resp.get("instruction") is None:
                    return None
                return resp

            response = retry_with_generator_update(
                text_generation_operation,
                "Text generation",
                logger, idx
            )

            if response is None:
                continue

            suitable_score = response.get("suitable", 0)
            with combination_lock:
                combination_dict[combination_key] = {
                    'score': suitable_score,
                    'image_descriptions': response.get('image_descriptions', []),
                    'prompt': response.get('instruction', ''),
                    'used': False
                }

        suitable_score = response.get("suitable", 0)

        from utils.common import SAVE_ORIGINAL_STRING
        unique_id_result = generate_unique_id("customization", return_original=SAVE_ORIGINAL_STRING, combination_key=combination_key)
        unique_id = unique_id_result[0] if isinstance(unique_id_result, tuple) else unique_id_result

        with unique_id_lock:
            if unique_id in generated_ids:
                continue

        valid_attempts += 1

        if suitable_score > best_score:
            best_score = suitable_score
            best_sample = {
                "response": response,
                "files": selected_files,
                "categories": selected_categories,
                "combination_key": combination_key,
                "unique_id": unique_id,
                "unique_id_result": unique_id_result
            }

        if suitable_score >= pass_threshold:
            break

    if best_sample is None:
        return None

    response = best_sample["response"]
    selected_files = best_sample["files"]
    selected_categories = best_sample["categories"]
    combination_key = best_sample["combination_key"]
    unique_id = best_sample["unique_id"]
    unique_id_result = best_sample["unique_id_result"]
    suitable_score = response.get("suitable", 0)

    with unique_id_lock:
        if unique_id in generated_ids:
            return None
        generated_ids.add(unique_id)

    with combination_lock:
        if combination_key in combination_dict and isinstance(combination_dict[combination_key], dict):
            combination_dict[combination_key]['used'] = True

    instruction = response.get('instruction', '')
    image_descriptions = response.get('image_descriptions', [])

    try:
        images = [Image.open(Path(f)) for f in selected_files]
    except Exception as e:
        logger.warning(f"[Sample {idx}] Failed to load image: {e}")
        with combination_lock:
            if combination_key in combination_dict and isinstance(combination_dict[combination_key], dict):
                combination_dict[combination_key]['used'] = False
        with unique_id_lock:
            generated_ids.discard(unique_id)
        return None

    def image_generation_operation():
        generated_image = image_generator.gen_response(
            prompt=instruction,
            response_format=None,
            think=False,
            images=images
        )
        return generated_image

    generated_image = retry_with_generator_update(
        image_generation_operation,
        "Image generation",
        logger, idx
    )

    if generated_image is None:
        with combination_lock:
            if combination_key in combination_dict and isinstance(combination_dict[combination_key], dict):
                combination_dict[combination_key]['used'] = False
        with unique_id_lock:
            generated_ids.discard(unique_id)
        return None

    output_dir = save_root / split_type / image_count_category / "data" / f"{idx:08d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image_path = output_dir / "image_output.jpg"

    try:
        if isinstance(generated_image, Image.Image):
            generated_image.save(output_image_path, "JPEG")
        elif isinstance(generated_image, (str, Path)):
            shutil.copy2(generated_image, output_image_path)
        else:
            logger.warning(f"[Sample {idx}] Generated image is not a valid type: {type(generated_image)}")
            with combination_lock:
                if combination_key in combination_dict and isinstance(combination_dict[combination_key], dict):
                    combination_dict[combination_key]['used'] = False
            with unique_id_lock:
                generated_ids.discard(unique_id)
            return None

        json_data = {
            'class': selected_categories[0] if len(set(selected_categories)) == 1 else 'mixed',
            'input_images': selected_files,
            'output_image': str(output_image_path),
            'instruction': instruction,
            'image_descriptions': image_descriptions,
            'suitable': suitable_score,
            'image_count': len(selected_files),
            'categories': selected_categories
        }

        image_files = {
            "image_output.jpg": output_image_path
        }

        success = save_sample_data(
            save_root,
            split_type,
            image_count_category,
            idx,
            unique_id_result,
            json_data,
            image_files
        )

        if success:
            with progress_lock:
                progress_stats['completed'] += 1
            return json_data
        else:
            with combination_lock:
                if combination_key in combination_dict and isinstance(combination_dict[combination_key], dict):
                    combination_dict[combination_key]['used'] = False
            with unique_id_lock:
                generated_ids.discard(unique_id)
            return None
    except Exception as e:
        logger.warning(f"[Sample {idx}] Failed to save generated image: {e}")
        with combination_lock:
            if combination_key in combination_dict and isinstance(combination_dict[combination_key], dict):
                combination_dict[combination_key]['used'] = False
        with unique_id_lock:
            generated_ids.discard(unique_id)
        return None


def process_split_data(
    split_dir: Path,
    final_dir: Path,
    split_type: str,
    image_count_category: str,
    target_count: int,
    generated_ids: Set[str],
    scene_groups: Optional[Dict[str, List[str]]] = None
) -> None:
    combination_dict = load_combination_dict(final_dir, split_type, image_count_category)

    if image_count_category == "1-3":
        image_count_range = [1, 2, 3]
        min_image, max_image = 1, 3
    elif image_count_category == "4-5":
        image_count_range = [4, 5]
        min_image, max_image = 4, 5
    elif image_count_category == "6-7":
        image_count_range = [6, 7]
        min_image, max_image = 6, 7
    else:  # >=8
        image_count_range = list(range(8, 11))
        min_image, max_image = 8, 10

    categories = ['human', 'cloth', 'object', 'scene', 'style']
    data = {
        'human': [],
        'cloth': [],
        'object': [],
        'scene': [],
        'style': []
    }

    for category in categories:
        samples = load_split_data(split_dir, split_type, category)
        for sample in samples:
            files = []
            if 'files' in sample:
                files = [f if isinstance(f, str) else f.get('filepath', '') for f in sample['files']]
            elif 'filepath' in sample:
                files = [sample['filepath']]
            data[category].extend(files)

    data_ratio = DATA_RATIO

    with progress_lock:
        progress_stats['total'] = target_count
        progress_stats['completed'] = len(generated_ids)
        progress_stats['failed'] = 0
        if progress_stats['start_time'] is None:
            progress_stats['start_time'] = time.time()

    stop_event = threading.Event()

    current_idx = len(generated_ids)
    completed_count = len(generated_ids)
    total_submitted = completed_count
    last_save_time = time.time()
    save_interval = 300

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []

            with tqdm(
                total=target_count,
                desc=f"{split_type}/{image_count_category}",
                unit="sample"
            ) as pbar:
                pbar.update(completed_count)

                while completed_count < target_count and not stop_event.is_set():
                    while len(futures) < MAX_WORKERS * 2 and current_idx < target_count * 2:
                        future = executor.submit(
                            worker_task,
                            current_idx,
                            data,
                            data_ratio,
                            image_count_range,
                            final_dir,
                            split_type,
                            image_count_category,
                            GEN_PROMPT,
                            stop_event,
                            GEMINI_API_KEY,
                            GEMINI_API_URL,
                            GEMINI_MODEL_NAME,
                            IMAGE_API_KEY,
                            IMAGE_API_URL,
                            IMAGE_MODEL_NAME,
                            MAX_TRIES,
                            combination_dict,
                            generated_ids,
                            scene_groups,
                            RANDOM_SEED,
                            MAX_ATTEMPTS,
                            PASS_THRESHOLD,
                            min_image,
                            max_image
                        )
                        futures.append(future)
                        current_idx += 1
                        total_submitted += 1

                    done_futures = []
                    for future in futures:
                        if future.done():
                            try:
                                result = future.result()
                                if result:
                                    completed_count += 1
                                    pbar.update(1)
                            except Exception as e:
                                print(f"Task execution exception: {e}")
                            done_futures.append(future)

                    for future in done_futures:
                        futures.remove(future)

                    current_time = time.time()
                    if current_time - last_save_time >= save_interval:
                        save_combination_dict(combination_dict, final_dir, split_type, image_count_category)
                        last_save_time = current_time

                    if completed_count >= target_count:
                        stop_event.set()
                        break

                    time.sleep(0.1)

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            completed_count += 1
                            pbar.update(1)
                    except Exception as e:
                        print(f"Task execution exception: {e}")

                    current_time = time.time()
                    if current_time - last_save_time >= save_interval:
                        save_combination_dict(combination_dict, final_dir, split_type, image_count_category)
                        last_save_time = current_time
    finally:
        save_combination_dict(combination_dict, final_dir, split_type, image_count_category)
        print(f"[{split_type}/{image_count_category}] Final save of combination dict ({len(combination_dict)} records)")

    print(f"\n{split_type}/{image_count_category} done: {completed_count}/{target_count}")


def main():
    print("=" * 80)
    print("Customization data generation script")
    print("=" * 80)
    print(f"Split directory: {SPLIT_DIR}")
    print(f"Final directory: {FINAL_DIR}")
    print(f"Generation config: {GEN_CONFIG}")
    print("=" * 80)

    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY is not set, please set it via environment variable")
    if not GEMINI_API_URL:
        print("Warning: GEMINI_API_URL is not set, please set it via environment variable")
    if not IMAGE_API_KEY:
        print("Warning: IMAGE_API_KEY is not set, please set it via environment variable")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    for split_type in ["train", "eval"]:
        print(f"\nProcessing {split_type} data...")

        for image_count_category, config in GEN_CONFIG.items():
            target_count = config.get(split_type, 0)

            if target_count <= 0:
                print(f"Skipping {split_type}/{image_count_category} data generation (target count is 0)")
                continue

            generated_ids = load_generated_ids(FINAL_DIR, split_type, image_count_category)
            print(f"Loaded {len(generated_ids)} already-generated sample IDs")

            process_split_data(
                split_dir=SPLIT_DIR,
                final_dir=FINAL_DIR,
                split_type=split_type,
                image_count_category=image_count_category,
                target_count=target_count,
                generated_ids=generated_ids,
            )

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
