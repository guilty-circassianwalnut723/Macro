"""
Qwen inference data loading utilities

Loads eval data for different tasks from Macro/data/filter/{task}/eval directory.
Uses the unified common_utils module.
"""

import sys
from pathlib import Path

# Add inference_utils path
SCRIPT_DIR = Path(__file__).parent
QWEN_DIR = SCRIPT_DIR.parent
MACRO_DIR = QWEN_DIR.parent
INFERENCE_UTILS_DIR = MACRO_DIR / "inference_utils"

if str(INFERENCE_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_UTILS_DIR))

# Import all functions from common_utils
from common_utils import (
    SUPPORTED_TASKS,
    IMAGE_NUM_CATEGORIES,
    THIRDPARTY_TASKS,
    parse_image_num_category,
    matches_image_num_category,
    load_eval_data,
    load_data_for_task,  # unified data loading interface
    filter_samples_by_image_num,
    get_available_tasks,
    get_available_categories,
    check_sample_exists,
    save_sample,
    get_data_root,
)

# Define DATA_ROOT for backward compatibility
DATA_ROOT = get_data_root(MACRO_DIR)

# Re-export all functions to ensure backward compatibility
__all__ = [
    'SUPPORTED_TASKS',
    'IMAGE_NUM_CATEGORIES',
    'THIRDPARTY_TASKS',
    'DATA_ROOT',
    'parse_image_num_category',
    'matches_image_num_category',
    'load_eval_data',
    'load_data_for_task',  # unified data loading interface
    'filter_samples_by_image_num',
    'get_available_tasks',
    'get_available_categories',
    'check_sample_exists',
    'save_sample',
]
