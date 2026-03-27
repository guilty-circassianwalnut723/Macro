"""
Common inference utility functions
Used by inference scripts for bagel/qwen/omnigen and other models.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image


THIRDPARTY_TASKS_AVAILABLE = False
load_prompts_for_task = None


# Supported task list
SUPPORTED_TASKS = ["customization", "illustration", "spatial", "temporal"]

# Third-party task list (removed; only 4 main tasks retained)
THIRDPARTY_TASKS = []

# Supported image num categories
IMAGE_NUM_CATEGORIES = ["1-3", "4-5", "6-7", ">=8"]


def get_data_root(macro_dir: Optional[Path] = None) -> Path:
    """
    Get the data root directory.

    Args:
        macro_dir: Macro root directory path; if None, auto-infer.

    Returns:
        Data root directory path.
    """
    if macro_dir is None:
        # Auto-infer: navigate up from the current file to the Macro root directory
        current_file = Path(__file__)
        macro_dir = current_file.parent.parent

    return macro_dir / "data" / "filter"  # data/filter/{task}/eval/


def parse_image_num_category(category: str) -> Tuple[int, Optional[int]]:
    """
    Parse an image num category string.

    Args:
        category: Category string, e.g. "1-3", "4-5", ">=8".

    Returns:
        (min_num, max_num) tuple; max_num is None if there is no upper bound.
    """
    if category == ">=8":
        return (8, None)
    elif "-" in category:
        parts = category.split("-")
        return (int(parts[0]), int(parts[1]))
    else:
        raise ValueError(f"Unsupported category format: {category}")


def matches_image_num_category(num_images: int, category: str) -> bool:
    """
    Check whether an image count matches a category.

    Args:
        num_images: Number of images.
        category: Category string.

    Returns:
        True if it matches, False otherwise.
    """
    min_num, max_num = parse_image_num_category(category)
    if max_num is None:
        return num_images >= min_num
    else:
        return min_num <= num_images <= max_num


def load_eval_data(
    task: str,
    image_num_category: Optional[str] = None,
    data_root: Optional[Path] = None,
    macro_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Load eval data from the filter directory.

    Args:
        task: Task type (customization, illustration, spatial, temporal).
        image_num_category: Image count category (1-3, 4-5, 6-7, >=8, all); None means all.
        data_root: Data root directory; if None, use default path.
        macro_dir: Macro root directory path, used to infer the data root.

    Returns:
        List of samples, each containing task, idx, prompt, input_images, output_image.
    """
    if data_root is None:
        data_root = get_data_root(macro_dir)

    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}. Supported tasks: {SUPPORTED_TASKS}")

    eval_dir = data_root / task / "eval"
    if not eval_dir.exists():
        raise ValueError(f"Eval directory does not exist: {eval_dir}")

    samples = []

    # If a category is specified, only load data for that category
    if image_num_category and image_num_category != "all":
        if image_num_category not in IMAGE_NUM_CATEGORIES:
            raise ValueError(f"Unsupported image_num_category: {image_num_category}")

        category_dir = eval_dir / image_num_category
        if category_dir.exists():
            samples.extend(_load_samples_from_category(category_dir, task, image_num_category))
    else:
        # Load data for all categories
        for category in IMAGE_NUM_CATEGORIES:
            category_dir = eval_dir / category
            if category_dir.exists():
                samples.extend(_load_samples_from_category(category_dir, task, category))

    # Sort by idx
    samples.sort(key=lambda x: (x.get("category", ""), x.get("idx", 0)))

    return samples


def _load_samples_from_category(
    category_dir: Path,
    task: str,
    category: str
) -> List[Dict[str, Any]]:
    """
    Load samples from a specific category directory.

    Args:
        category_dir: Category directory path.
        task: Task type.
        category: Category name.

    Returns:
        List of samples.
    """
    samples = []

    # Find all JSON files
    json_files = sorted(category_dir.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                sample = json.load(f)
                # Ensure category info is included
                sample["category"] = category
                samples.append(sample)
        except Exception as e:
            print(f"Warning: failed to read JSON file {json_file}: {e}")
            continue

    return samples


def filter_samples_by_image_num(
    samples: List[Dict[str, Any]],
    image_num_category: str
) -> List[Dict[str, Any]]:
    """
    Filter samples by image num category.

    Args:
        samples: List of samples.
        image_num_category: Image count category.

    Returns:
        Filtered list of samples.
    """
    if image_num_category == "all":
        return samples

    filtered = []
    for sample in samples:
        num_images = len(sample.get("input_images", []))
        if matches_image_num_category(num_images, image_num_category):
            filtered.append(sample)

    return filtered


def get_available_tasks(data_root: Optional[Path] = None, macro_dir: Optional[Path] = None) -> List[str]:
    """
    Get the list of available tasks.

    Args:
        data_root: Data root directory.
        macro_dir: Macro root directory path.

    Returns:
        List of available tasks.
    """
    if data_root is None:
        data_root = get_data_root(macro_dir)

    available_tasks = []
    for task in SUPPORTED_TASKS:
        eval_dir = data_root / task / "eval"
        if eval_dir.exists():
            available_tasks.append(task)

    return available_tasks


def get_available_categories(task: str, data_root: Optional[Path] = None, macro_dir: Optional[Path] = None) -> List[str]:
    """
    Get the list of available categories for a given task.

    Args:
        task: Task type.
        data_root: Data root directory.
        macro_dir: Macro root directory path.

    Returns:
        List of available categories.
    """
    if data_root is None:
        data_root = get_data_root(macro_dir)

    eval_dir = data_root / task / "eval"
    if not eval_dir.exists():
        return []

    available_categories = []
    for category in IMAGE_NUM_CATEGORIES:
        category_dir = eval_dir / category
        if category_dir.exists() and any(category_dir.glob("*.json")):
            available_categories.append(category)

    return available_categories


def check_sample_exists(output_dir: Path, idx: int) -> bool:
    """
    Check whether a sample has already been generated (resume checkpoint check).

    Strict checks:
    1. Both the image file and JSON file must exist.
    2. The image file must be openable and readable.
    3. The image must have valid dimensions (width and height both > 0).
    4. The JSON file must be parseable.
    5. The JSON file must contain the required fields.

    Args:
        output_dir: Output directory.
        idx: Sample index.

    Returns:
        True if the sample exists and files are readable; False otherwise.
    """
    image_file = output_dir / f"{idx:08d}.jpg"
    json_file = output_dir / f"{idx:08d}.json"

    # Check if files exist
    if not image_file.exists() or not json_file.exists():
        return False

    # Check if the image file is readable (not just verify, but actually load and check)
    try:
        with Image.open(image_file) as img:
            # Verify image integrity
            img.verify()

        # Reopen image to check dimensions (must reopen after verify)
        with Image.open(image_file) as img:
            # Convert to RGB to ensure readability
            img = img.convert("RGB")
            width, height = img.size
            # Check that image dimensions are valid
            if width <= 0 or height <= 0:
                return False
            # Try to load one pixel to ensure the image is fully readable
            img.load()
    except Exception as e:
        # If the image cannot be read, return False
        return False

    # Check if the JSON file is readable
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Check required fields
            if 'output_image' not in data:
                return False
            # Verify that output_image path points to an existing file
            output_image_path = data.get('output_image', '')
            if output_image_path and not Path(output_image_path).exists():
                return False
    except Exception:
        return False

    return True


def load_data_for_task(
    task: str,
    image_num_category: Optional[str] = None,
    data_root: Optional[Path] = None,
    macro_dir: Optional[Path] = None,
    use_refine_prompt: bool = False
) -> List[Dict[str, Any]]:
    """
    Unified data loading interface supporting all task types (regular and third-party tasks).

    Args:
        task: Task type (customization, illustration, spatial, temporal, omnicontext, geneval, dpg).
        image_num_category: Image count category (1-3, 4-5, 6-7, >=8, all); only valid for regular tasks.
        data_root: Data root directory; if None, use default path.
        macro_dir: Macro root directory path, used to infer the data root.
        use_refine_prompt: For geneval, whether to use a refined prompt.

    Returns:
        List of samples, each containing:
        - Regular tasks: task, idx, prompt, input_images, output_image, category
        - Third-party tasks: idx, prompt/instruction, input_images, task, and other task-specific fields
    """
    # Third-party tasks
    if task in THIRDPARTY_TASKS:
        if not THIRDPARTY_TASKS_AVAILABLE:
            raise ImportError("Third-party task module not found. Please check inference_utils/thirdparty_tasks.py")

        samples = load_prompts_for_task(task, data_root, use_refine_prompt=use_refine_prompt)

        # Ensure all samples have unified field names
        for sample in samples:
            # Ensure prompt field exists (from instruction or other fields)
            if "prompt" not in sample:
                if "instruction" in sample:
                    sample["prompt"] = sample["instruction"]
                elif "text" in sample:
                    sample["prompt"] = sample["text"]

            # Ensure instruction field exists (from prompt)
            if "instruction" not in sample:
                sample["instruction"] = sample.get("prompt", "")

            # Ensure input_images field exists (from images or other fields)
            if "input_images" not in sample:
                if "images" in sample:
                    sample["input_images"] = sample["images"]
                elif "image_paths" in sample:
                    sample["input_images"] = sample["image_paths"]
                else:
                    sample["input_images"] = []

            # Ensure task field exists (for identifying task type)
            if "task" not in sample:
                sample["task"] = task

            # Ensure idx field exists
            if "idx" not in sample:
                # Try to get from other fields
                if "id" in sample:
                    sample["idx"] = sample["id"]
                elif "index" in sample:
                    sample["idx"] = sample["index"]
                else:
                    # Use list index
                    sample["idx"] = samples.index(sample)

        return samples

    # Regular tasks
    return load_eval_data(
        task=task,
        image_num_category=image_num_category,
        data_root=data_root,
        macro_dir=macro_dir
    )


def save_sample(
    output_dir: Path,
    idx: int,
    sample: Dict[str, Any],
    output_image_path: str,
    target_image_path: Optional[str] = None,
    seed: Optional[int] = None
):
    """
    Unified sample data saving interface supporting all task types.

    Args:
        output_dir: Output directory.
        idx: Sample index.
        sample: Original sample data.
        output_image_path: Path of the generated image.
        target_image_path: Target image path (optional).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON file
    output_json = sample.copy()
    output_json["output_image"] = str(output_image_path)
    if target_image_path:
        output_json["target_image"] = target_image_path
    elif "output_image" in sample:
        # If the original sample has output_image, use it as target_image
        output_json["target_image"] = sample.get("output_image", "")

    # Record seed in JSON if specified
    if seed is not None:
        output_json["seed"] = seed

    # Ensure idx exists (consistent with bagel/vis etc., for evaluation and visualization)
    if "idx" not in output_json:
        output_json["idx"] = idx

    # Ensure prompt and instruction fields exist (for compatibility)
    if "prompt" not in output_json and "instruction" in output_json:
        output_json["prompt"] = output_json["instruction"]
    if "instruction" not in output_json and "prompt" in output_json:
        output_json["instruction"] = output_json["prompt"]

    # Handle input_images field: convert PIL Image objects to path strings or remove
    if "input_images" in output_json:
        serializable_input_images = []
        for img_item in output_json["input_images"]:
            if isinstance(img_item, (str, Path)):
                # Already a path string
                serializable_input_images.append(str(img_item))
            elif hasattr(img_item, 'filename') and img_item.filename:
                # PIL Image object, try to get filename
                serializable_input_images.append(str(img_item.filename))
            else:
                # Cannot serialize; skip or use placeholder
                pass
        output_json["input_images"] = serializable_input_images

    # Clean up other fields that may contain non-serializable objects
    def make_json_serializable(obj):
        """Recursively convert an object to a JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            # Try to get the string representation of the object
            return str(obj)
        else:
            # For other types, try to convert to string
            return str(obj)

    # Apply serialization to output_json
    output_json = make_json_serializable(output_json)

    # Save JSON file
    # For geneval and dpg, if seed is specified, include seed info in the filename
    if seed is not None:
        json_file = output_dir / f"{idx:08d}_seed{seed:02d}.json"
    else:
        json_file = output_dir / f"{idx:08d}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    # Set file permissions
    os.chmod(json_file, 0o777)
