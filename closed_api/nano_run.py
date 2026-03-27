#!/usr/bin/env python3
"""
Generate images using the Nano API.
Supports our tasks (customization/illustration/spatial/temporal) and omni (omnicontext).
Configure the tasks and image_num_category to run via the CONFIG block at the top.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# ============== Configure here ==============
CONFIG = {
    # Our tasks: key is the task name, value is the list of image_num_category to run;
    # use "all" to run all categories for that task, empty list [] to skip the task
    "ours": {
        "customization": ["1-3", "4-5", "6-7", ">=8"],
        "illustration": ["1-3", "4-5", "6-7", ">=8"],
        "spatial": ["1-3", "4-5", "6-7", ">=8"],
        "temporal": ["1-3", "4-5", "6-7", ">=8"],
    },
    # Whether to generate omnicontext (omni)
    "omni": True,
    # Output root directory
    "output_root": "./outputs/api/nano",
    # Number of parallel workers (concurrent API requests)
    "num_workers": 16,
    # API parameters (modify as needed)
    "api_key": os.environ.get("NANO_API_KEY", ""),
    "model_name": "gemini-3-pro-image-preview",
    "timeout": 60,
    "print_log": True,
}
# =====================================

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
MACRO_DIR = SCRIPT_DIR.parent  # Macro root directory
INFERENCE_UTILS_DIR = MACRO_DIR / "inference_utils"

for p in [MACRO_DIR, INFERENCE_UTILS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from api_generator.image_generator.nano_api import NanoAPIGenerator
from common_utils import (
    load_data_for_task,
    check_sample_exists,
    save_sample,
    SUPPORTED_TASKS,
)


def load_images_for_sample(sample: Dict[str, Any]) -> List[Image.Image]:
    """Convert sample['input_images'] (paths or PIL) to a list of PIL Images."""
    images = []
    for item in sample.get("input_images", []):
        if isinstance(item, (str, Path)):
            path = Path(item)
            if path.exists():
                images.append(Image.open(path).convert("RGB"))
        elif isinstance(item, Image.Image):
            images.append(item.convert("RGB"))
    return images


def process_one_sample(
    generator: NanoAPIGenerator,
    sample: Dict[str, Any],
    output_dir: Path,
    task: str,
) -> bool:
    """Process a single sample. Returns True on success or skip, False on failure."""
    idx = sample.get("idx", 0)
    if check_sample_exists(output_dir, idx):
        return True
    prompt = sample.get("prompt", "")
    images = load_images_for_sample(sample)
    try:
        out_img = generator.gen_response(prompt, images=images if images else None)
        if out_img is None:
            return False
        out_path = output_dir / f"{idx:08d}.jpg"
        out_img.save(out_path)
        os.chmod(out_path, 0o777)
        save_sample(
            output_dir=output_dir,
            idx=idx,
            sample=sample,
            output_image_path=str(out_path),
            target_image_path=sample.get("output_image"),
        )
        return True
    except Exception as e:
        print(f"[nano] idx={idx} failed: {e}")
        return False


def run_task(
    generators: List[NanoAPIGenerator],
    task: str,
    image_num_category: Optional[str],
    output_dir: Path,
    macro_dir: Path,
    num_workers: int,
) -> int:
    """Run a single task + category (parallel). Returns the number of failures."""
    samples = load_data_for_task(
        task=task,
        image_num_category=image_num_category,
        macro_dir=macro_dir,
    )
    if not samples:
        return 0

    to_do = [(generators[i % len(generators)], s) for i, s in enumerate(samples)]
    failed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_one_sample, g, s, output_dir, task): s
            for g, s in to_do
        }
        for future in as_completed(futures):
            if not future.result():
                failed += 1
    return failed


def main():
    cfg = CONFIG
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    num_workers = max(1, int(cfg.get("num_workers", 4)))
    generators = [
        NanoAPIGenerator(
            api_key=cfg.get("api_key", os.environ.get("NANO_API_KEY", "")),
            model_name=cfg.get("model_name", "gemini-3-pro-image-preview"),
            timeout=cfg.get("timeout", 60),
            print_log=cfg.get("print_log", False),
            max_try=cfg.get("max_try", 5),
        )
        for _ in range(num_workers)
    ]

    total_failed = 0

    # Our tasks
    ours = cfg.get("ours", {})
    for task in SUPPORTED_TASKS:
        if task not in ours:
            continue
        cats = ours[task]
        if not cats:
            continue
        if "all" in cats:
            task_out = output_root / task / "all"
            task_out.mkdir(parents=True, exist_ok=True)
            total_failed += run_task(generators, task, None, task_out, MACRO_DIR, num_workers)
        else:
            for cat in cats:
                task_out = output_root / task / cat
                task_out.mkdir(parents=True, exist_ok=True)
                total_failed += run_task(generators, task, cat, task_out, MACRO_DIR, num_workers)

    # omni（omnicontext）
    if cfg.get("omni", False):
        task_out = output_root / "omnicontext"
        task_out.mkdir(parents=True, exist_ok=True)
        total_failed += run_task(generators, "omnicontext", None, task_out, MACRO_DIR, num_workers)

    if total_failed > 0:
        sys.exit(1)
    print("nano_run all done")
    sys.exit(0)


if __name__ == "__main__":
    main()
