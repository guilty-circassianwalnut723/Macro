#!/usr/bin/env python3
"""
使用 GPT API (images/edits) 生成图像。
支持 ours 任务（customization/illustration/spatial/temporal）和 omni（omnicontext）。
通过顶部 CONFIG 配置要跑的任务与 image_num_category。
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# ============== 在此配置 ==============
CONFIG = {
    # ours 任务：key 为任务名，value 为该任务要跑的 image_num_category 列表；
    # 用 "all" 表示该任务下所有 category，空列表 [] 表示不跑该任务
    "ours": {
        "customization": ["1-3", "4-5", "6-7", ">=8"],
        "illustration": ["1-3", "4-5", "6-7", ">=8"],
        "spatial": ["1-3", "4-5", "6-7", ">=8"],
        "temporal": ["1-3", "4-5", "6-7", ">=8"],
    },
    # 是否生成 omnicontext（omni）
    "omni": True,
    # 输出根目录
    "output_root": "./outputs/api/gpt",
    # 并行 worker 数（同时发起的 API 请求数）
    "num_workers": 4,
    # API 参数（按需修改）
    "api_key": os.environ.get("GPT_API_KEY", ""),
    "model_name": "gpt-image-1.5",
    "size": "auto",
    "quality": "auto",
    "timeout": 180,
    "print_log": True,
}
# =====================================

# Path setup - api_generator and inference_utils are in the parent Macro directory
SCRIPT_DIR = Path(__file__).resolve().parent
MACRO_DIR = SCRIPT_DIR.parent  # Macro root directory
INFERENCE_UTILS_DIR = MACRO_DIR / "inference_utils"

for p in [MACRO_DIR, INFERENCE_UTILS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from api_generator.image_generator.gpt_image_api import GPTImageAPIGenerator as GPTAPIGenerator
from common_utils import (
    load_data_for_task,
    check_sample_exists,
    save_sample,
    SUPPORTED_TASKS,
)


def load_images_for_sample(sample: Dict[str, Any]) -> List[Image.Image]:
    """将 sample['input_images']（路径或 PIL）转为 PIL 列表。"""
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
    generator: GPTAPIGenerator,
    sample: Dict[str, Any],
    output_dir: Path,
    task: str,
) -> bool:
    """处理单条样本，返回 True 表示成功或已跳过，False 表示失败。"""
    idx = sample.get("idx", 0)
    if check_sample_exists(output_dir, idx):
        return True
    prompt = sample.get("prompt", "")
    images = load_images_for_sample(sample)
    if not images:
        print(f"[gpt] 跳过 {task} idx={idx}: 无有效输入图像")
        return False
    try:
        out_img = generator.gen_response(prompt, images=images)
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
        print(f"[gpt] idx={idx} 失败: {e}")
        return False


def run_task(
    generators: List[GPTAPIGenerator],
    task: str,
    image_num_category: Optional[str],
    output_dir: Path,
    macro_dir: Path,
    num_workers: int,
) -> int:
    """跑单个 task + category（并行），返回失败数。"""
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
        GPTAPIGenerator(
            api_key=cfg.get("api_key", os.environ.get("GPT_API_KEY", "")),
            model_name=cfg.get("model_name", "gpt-image-1"),
            size=cfg.get("size", "auto"),
            quality=cfg.get("quality", "auto"),
            timeout=cfg.get("timeout", 180),
            print_log=cfg.get("print_log", False),
            max_try=cfg.get("max_try", 5),
        )
        for _ in range(num_workers)
    ]

    total_failed = 0

    # ours 任务
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
    print("gpt_run 全部完成")
    sys.exit(0)


if __name__ == "__main__":
    main()
