"""
数据加载工具

从 Macro/data/filter/{task}/eval 目录下加载不同task的eval数据
使用统一的 common_utils 模块
"""

import sys
from pathlib import Path

# 添加 inference_utils 路径
SCRIPT_DIR = Path(__file__).parent
BAGEL_DIR = SCRIPT_DIR.parent
MACRO_DIR = BAGEL_DIR.parent
INFERENCE_UTILS_DIR = MACRO_DIR / "inference_utils"

if str(INFERENCE_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_UTILS_DIR))

# 从 common_utils 导入所有函数
from common_utils import (
    SUPPORTED_TASKS,
    IMAGE_NUM_CATEGORIES,
    THIRDPARTY_TASKS,
    parse_image_num_category,
    matches_image_num_category,
    load_eval_data,
    load_data_for_task,  # 统一的数据加载接口
    filter_samples_by_image_num,
    get_available_tasks,
    get_available_categories,
    check_sample_exists,
    save_sample,
    get_data_root,
)

# 为了向后兼容，定义 DATA_ROOT
DATA_ROOT = get_data_root(MACRO_DIR)

# 重新导出所有函数，确保向后兼容
__all__ = [
    'SUPPORTED_TASKS',
    'IMAGE_NUM_CATEGORIES',
    'THIRDPARTY_TASKS',
    'DATA_ROOT',
    'parse_image_num_category',
    'matches_image_num_category',
    'load_eval_data',
    'load_data_for_task',  # 统一的数据加载接口
    'filter_samples_by_image_num',
    'get_available_tasks',
    'get_available_categories',
    'check_sample_exists',
    'save_sample',
]
