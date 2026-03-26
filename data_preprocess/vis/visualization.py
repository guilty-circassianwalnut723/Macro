#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据可视化脚本 - 支持 data/final 和 data/filter 两种目录结构

用法:
    # 可视化 data/filter（默认）
    python visualization.py

    # 可视化 data/final
    python visualization.py --data_source final

    # 指定自定义数据根目录
    python visualization.py --data_root /path/to/data/filter
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from flask import Flask, render_template, jsonify, send_from_directory, request
from flask_cors import CORS
from natsort import natsorted

# 使用本地的模板目录（静态文件也放在这里）
SCRIPT_DIR = Path(__file__).parent
TEMPLATES_DIR = SCRIPT_DIR / 'templates'
STATIC_DIR = SCRIPT_DIR / 'static'

app = Flask(__name__, 
            template_folder=str(TEMPLATES_DIR),
            static_folder=str(STATIC_DIR),
            static_url_path='/static')
CORS(app)

# ============================================================================
# 配置常量
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 解析命令行参数（在模块加载时处理，供 Flask 路由使用）
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--data_source', choices=['final', 'filter'], default='filter',
                     help='数据来源: final=data/final, filter=data/filter (默认: filter)')
_parser.add_argument('--data_root', type=str, default=None,
                     help='自定义数据根目录路径（优先级高于 --data_source）')
_parser.add_argument('--port', type=int, default=8413, help='服务端口（默认: 8413）')
_args, _ = _parser.parse_known_args()

# 确定数据目录
if _args.data_root:
    DATA_DIR = Path(_args.data_root).resolve()
elif _args.data_source == 'final':
    DATA_DIR = BASE_DIR / "data" / "final"
else:
    DATA_DIR = BASE_DIR / "data" / "filter"

# 同时保留 FINAL_DIR 以供图片路径解析使用
FINAL_DIR = BASE_DIR / "data" / "final"

SAMPLES_PER_PAGE = 10  # 每页显示的样本数

# 任务列表
TASKS = ["spatial", "temporal", "customization", "illustration"]

# Spatial 子类型
SPATIAL_SUB_TYPES = ["indoor", "outdoor", "object"]

# 图像数量类别
IMAGE_COUNT_CATEGORIES = ["1-3", "4-5", "6-7", ">=8"]

# 分割类型
SPLIT_TYPES = ["train", "eval"]

# ============================================================================
# 数据布局检测
# ============================================================================

def detect_layout(category_dir: Path) -> str:
    """
    检测目录布局：
      - 'nested': 有 json/ 子目录（data/final 格式）
      - 'flat':   JSON 文件直接在 category_dir 下（data/filter 格式）
      - 'empty':  目录不存在或为空
    """
    if not category_dir.exists():
        return 'empty'
    json_subdir = category_dir / "json"
    if json_subdir.exists() and any(json_subdir.glob("*.json")):
        return 'nested'
    if any(category_dir.glob("*.json")):
        return 'flat'
    return 'empty'

# ============================================================================
# Pass/Fail 筛选配置参数（参考filter目录下的脚本）
# ============================================================================
# Temporal任务阈值
TEMPORAL_SCORE_THRESHOLD = 6

# Customization任务阈值
CONSISTENCY_SCORE_THRESHOLD = 6.0  # consistency_scores列表中每个分数都需要 >= 此阈值
FOLLOWING_SCORE_THRESHOLD = 6.0    # following_score需要 >= 此阈值

# Illustration任务阈值
TRAINING_SCORE_THRESHOLD = 6
GUIDANCE_SCORE_THRESHOLD = 6

# Spatial任务：没有筛选标准，全部保留（所有样本都是pass）


# ============================================================================
# Pass/Fail 判断函数
# ============================================================================

def is_sample_pass(sample: Dict[str, Any], task: str) -> bool:
    """
    判断样本是否通过筛选（pass）
    
    Args:
        sample: 样本数据
        task: 任务类型
    
    Returns:
        True表示pass，False表示fail
    """
    # 首先检查输入图像数量，不超过10张
    input_images = sample.get("input_images", [])
    if isinstance(input_images, list) and len(input_images) > 10:
        return False
    
    if task == "spatial":
        # Spatial任务：没有筛选标准，全部保留（但需要满足图像数量限制）
        return True
    
    elif task == "temporal":
        # Temporal任务：检查temporal_score或score
        temporal_score = sample.get("temporal_score", sample.get("score", 0))
        return temporal_score >= TEMPORAL_SCORE_THRESHOLD
    
    elif task == "customization":
        # Customization任务：检查consistency_scores和following_score
        consistency_scores = sample.get("consistency_scores", [])
        following_score = sample.get("following_score")
        
        # 检查consistency_scores：列表中每个分数都需要 >= threshold
        all_consistency_ok = True
        if isinstance(consistency_scores, list) and len(consistency_scores) > 0:
            for score in consistency_scores:
                if not isinstance(score, (int, float)) or score < CONSISTENCY_SCORE_THRESHOLD:
                    all_consistency_ok = False
                    break
        else:
            all_consistency_ok = False
        
        # 检查following_score
        following_ok = isinstance(following_score, (int, float)) and following_score >= FOLLOWING_SCORE_THRESHOLD
        
        # 两个条件都满足才pass
        return all_consistency_ok and following_ok
    
    elif task == "illustration":
        # Illustration任务：检查training_score和guidance_score，以及image_contributions
        # 检查image_contributions并计算effective_image_count
        image_contributions = sample.get("image_contributions", [])
        if isinstance(image_contributions, list):
            effective_image_count = sum(1 for x in image_contributions if x is True)
        else:
            effective_image_count = sample.get("effective_image_count", 0)
        
        image_count = sample.get("image_count", 0)
        
        # 筛去effective_image_count为0或image_count为0的样本
        if effective_image_count == 0 or image_count == 0:
            return False
        
        # 检查分数
        training_score = sample.get("suitable", sample.get("training_score", 0))
        guidance_score = sample.get("guidance_score", 0)
        
        # 两个分数都需要 >= threshold
        return training_score >= TRAINING_SCORE_THRESHOLD and guidance_score >= GUIDANCE_SCORE_THRESHOLD
    
    else:
        # 未知任务类型，默认返回True（但需要满足图像数量限制）
        return True


# ============================================================================
# 数据加载函数
# ============================================================================

def spatial_has_sub_types() -> bool:
    """
    检测当前数据源的 spatial 目录是否存在 sub_type 层（indoor/outdoor/object）。
    - data/final/spatial: train/{sub_type}/{category}/  → True
    - data/filter/spatial: train/{category}/            → False
    """
    spatial_dir = DATA_DIR / "spatial"
    if not spatial_dir.exists():
        return False
    for split_type in SPLIT_TYPES:
        split_dir = spatial_dir / split_type
        if split_dir.exists():
            for item in split_dir.iterdir():
                if item.is_dir() and item.name in SPATIAL_SUB_TYPES:
                    return True
            # 如果第一个有效 split 目录下没有 sub_type，就不需要 sub_type
            return False
    return False


def get_spatial_sub_types() -> List[str]:
    """获取可用的 spatial 子类型。若当前布局无 sub_type 层，返回空列表。"""
    if not spatial_has_sub_types():
        return []
    spatial_dir = DATA_DIR / "spatial"
    sub_types = []
    for split_type in SPLIT_TYPES:
        split_dir = spatial_dir / split_type
        if split_dir.exists():
            for item in split_dir.iterdir():
                if item.is_dir() and item.name in SPATIAL_SUB_TYPES:
                    if item.name not in sub_types:
                        sub_types.append(item.name)
    return natsorted(sub_types)


def get_image_count_categories(task: str, split_type: str, sub_type: Optional[str] = None) -> List[str]:
    """获取指定任务、分割类型和子类型下的所有图像数量类别（兼容 nested/flat 布局）"""
    task_dir = DATA_DIR / task
    if not task_dir.exists():
        return []
    
    # spatial 且当前布局无 sub_type 层时，忽略传入的 sub_type
    if task == "spatial" and not spatial_has_sub_types():
        sub_type = None
    
    if sub_type:
        split_dir = task_dir / split_type / sub_type
    else:
        split_dir = task_dir / split_type
    
    if not split_dir.exists():
        return []
    
    categories = []
    for item in split_dir.iterdir():
        if item.is_dir() and item.name in IMAGE_COUNT_CATEGORIES:
            layout = detect_layout(item)
            if layout != 'empty':
                categories.append(item.name)
    
    return natsorted(categories)


def get_json_files_list(
    task: str, 
    split_type: str, 
    image_count_category: str,
    sub_type: Optional[str] = None
) -> List[Path]:
    """获取指定条件下的所有 JSON 文件路径列表（兼容 nested/flat 布局）"""
    task_dir = DATA_DIR / task
    if not task_dir.exists():
        return []
    
    # spatial 且当前布局无 sub_type 层时，忽略传入的 sub_type
    if task == "spatial" and not spatial_has_sub_types():
        sub_type = None
    
    if sub_type:
        category_dir = task_dir / split_type / sub_type / image_count_category
    else:
        category_dir = task_dir / split_type / image_count_category
    
    layout = detect_layout(category_dir)
    if layout == 'nested':
        json_dir = category_dir / "json"
        json_files = natsorted(json_dir.glob("*.json"))
    elif layout == 'flat':
        json_files = natsorted(category_dir.glob("*.json"))
    else:
        return []
    
    return json_files


def load_samples(
    task: str, 
    split_type: str, 
    image_count_category: str,
    sub_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """加载指定条件下的所有样本数据（已废弃，保留用于兼容）"""
    json_files = get_json_files_list(task, split_type, image_count_category, sub_type)
    
    samples = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 添加文件路径信息，用于后续定位图片
                data['_json_file'] = str(json_file)
                data['_data_dir'] = str(FINAL_DIR)
                samples.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return samples


def load_samples_by_page(
    task: str, 
    split_type: str, 
    image_count_category: str,
    page: int,
    per_page: int = SAMPLES_PER_PAGE,
    sub_type: Optional[str] = None,
    filter_type: str = "all"  # "all", "pass", "fail"
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    按页加载样本数据，只加载当前页需要的文件
    
    Args:
        task: 任务类型
        split_type: 分割类型（train/eval）
        image_count_category: 图像数量类别
        page: 页码
        per_page: 每页样本数
        sub_type: 子类型（spatial任务需要）
        filter_type: 筛选类型，"all"表示全部，"pass"表示通过筛选的，"fail"表示未通过筛选的
    
    Returns:
        (样本列表, 总页数, 当前页码)
    """
    json_files = get_json_files_list(task, split_type, image_count_category, sub_type)
    
    if not json_files:
        return [], 1, 1
    
    # 如果需要筛选，先加载所有文件进行筛选
    if filter_type != "all":
        all_samples = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 添加文件路径信息，用于后续定位图片
                    data['_json_file'] = str(json_file)
                    data['_data_dir'] = str(FINAL_DIR)
                    all_samples.append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        # 根据filter_type筛选样本
        if filter_type == "pass":
            filtered_samples = [s for s in all_samples if is_sample_pass(s, task)]
        elif filter_type == "fail":
            filtered_samples = [s for s in all_samples if not is_sample_pass(s, task)]
        else:
            filtered_samples = all_samples
        
        # 计算分页
        total_files = len(filtered_samples)
        if total_files == 0:
            return [], 1, 1
        
        total_pages = (total_files + per_page - 1) // per_page
        page = max(1, min(page, total_pages))
        
        # 获取当前页的样本
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        samples = filtered_samples[start_idx:end_idx]
        
        return samples, total_pages, page
    else:
        # 不需要筛选，直接按页加载
        total_files = len(json_files)
        total_pages = (total_files + per_page - 1) // per_page
        page = max(1, min(page, total_pages))
        
        # 只加载当前页需要的文件
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_files = json_files[start_idx:end_idx]
        
        samples = []
        for json_file in page_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 添加文件路径信息，用于后续定位图片
                    data['_json_file'] = str(json_file)
                    data['_data_dir'] = str(FINAL_DIR)
                    samples.append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        return samples, total_pages, page


def paginate_samples(
    samples: List[Dict[str, Any]], 
    page: int, 
    per_page: int = SAMPLES_PER_PAGE
) -> Tuple[List[Dict[str, Any]], int, int]:
    """对样本列表进行分页（已废弃，保留用于兼容）"""
    if not samples:
        return [], 1, 1
    
    total_pages = (len(samples) + per_page - 1) // per_page
    page = max(1, min(page, total_pages))
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    current_page_samples = samples[start_idx:end_idx]
    
    return current_page_samples, total_pages, page


def fix_image_path(image_path: str, data_dir: str, json_file: Optional[str] = None) -> str:
    """
    修正图像路径。
    - 绝对路径且存在：直接返回
    - 相对路径：优先相对于 BASE_DIR 解析（data/final/... 格式），否则回退到其他策略
    """
    if not image_path:
        return ""
    
    path_obj = Path(image_path)
    
    # 绝对路径且存在，直接返回
    if path_obj.is_absolute() and path_obj.exists():
        return str(path_obj)
    
    # 相对路径：相对于 BASE_DIR 解析（适用于 data/final/... 和 data/filter/... 格式）
    if not path_obj.is_absolute():
        candidate = BASE_DIR / path_obj
        if candidate.exists():
            return str(candidate)
    
    # 尝试用文件名在 data_dir 下匹配（兜底）
    filename = path_obj.name
    if data_dir:
        data_dir_obj = Path(data_dir)
        if data_dir_obj.exists():
            for subdir in data_dir_obj.iterdir():
                if subdir.is_dir():
                    test_path = subdir / filename
                    if test_path.exists():
                        return str(test_path)
            test_path = data_dir_obj / filename
            if test_path.exists():
                return str(test_path)
    
    # 相对于 json 文件所在目录的 data/ 子目录查找
    if json_file:
        json_path = Path(json_file)
        if json_path.exists():
            json_dir = json_path.parent
            data_dir_from_json = json_dir.parent / "data"
            if data_dir_from_json.exists():
                for subdir in data_dir_from_json.iterdir():
                    if subdir.is_dir():
                        test_path = subdir / filename
                        if test_path.exists():
                            return str(test_path)
    
    # 找不到时返回原始路径（前端处理加载失败）
    return image_path



# ============================================================================
# Flask路由函数
# ============================================================================

@app.route('/')
def index():
    """主页面路由"""
    return render_template('index.html')


@app.route('/api/tasks')
def api_tasks():
    """获取所有可用task的API"""
    available_tasks = []
    for task in TASKS:
        task_dir = DATA_DIR / task
        if task_dir.exists():
            has_data = False
            for split_type in SPLIT_TYPES:
                if task == "spatial" and spatial_has_sub_types():
                    # final 布局：需要在 sub_type 子目录下查找
                    for sub_type in SPATIAL_SUB_TYPES:
                        sub_dir = task_dir / split_type / sub_type
                        if sub_dir.exists() and any(sub_dir.iterdir()):
                            has_data = True
                            break
                else:
                    # filter 布局或非 spatial 任务：直接在 split 目录下查找
                    split_dir = task_dir / split_type
                    if split_dir.exists() and any(split_dir.iterdir()):
                        has_data = True
                if has_data:
                    break
            if has_data:
                available_tasks.append(task)
    
    return jsonify(available_tasks)


@app.route('/api/spatial_sub_types')
def api_spatial_sub_types():
    """获取 spatial 子类型列表"""
    sub_types = get_spatial_sub_types()
    return jsonify(sub_types)


@app.route('/api/image_count_categories')
def api_image_count_categories():
    """获取指定条件下的图像数量类别列表"""
    task = request.args.get('task', '')
    split_type = request.args.get('split_type', '')
    sub_type = request.args.get('sub_type', '') or None
    
    if not task or not split_type:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    categories = get_image_count_categories(task, split_type, sub_type)
    return jsonify(categories)


@app.route('/api/samples')
def api_samples():
    """获取分页样本数据的API"""
    task = request.args.get('task', '')
    split_type = request.args.get('split_type', '')
    image_count_category = request.args.get('image_count_category', '')
    sub_type = request.args.get('sub_type', '') or None
    page_param = request.args.get('page', '1')
    filter_type = request.args.get('filter', 'all')  # 新增filter参数
    
    if not task or not split_type or not image_count_category:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    # 验证filter_type
    if filter_type not in ['all', 'pass', 'fail']:
        filter_type = 'all'
    
    try:
        page = int(page_param)
        if page < 1:
            page = 1
    except ValueError:
        page = 1
    
    # 按页加载样本，只加载当前页需要的文件
    current_page_samples, total_pages, current_page = load_samples_by_page(
        task, split_type, image_count_category, page, SAMPLES_PER_PAGE, sub_type, filter_type
    )
    
    # 获取总样本数（用于显示总数）
    # 如果进行了筛选，需要重新计算总数
    if filter_type != "all":
        json_files = get_json_files_list(task, split_type, image_count_category, sub_type)
        all_samples = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_samples.append(data)
            except Exception as e:
                continue
        
        if filter_type == "pass":
            total_samples = sum(1 for s in all_samples if is_sample_pass(s, task))
        elif filter_type == "fail":
            total_samples = sum(1 for s in all_samples if not is_sample_pass(s, task))
        else:
            total_samples = len(all_samples)
    else:
        json_files = get_json_files_list(task, split_type, image_count_category, sub_type)
        total_samples = len(json_files)
    
    # 处理样本数据，修正图片路径
    processed_samples = []
    for sample in current_page_samples:
        processed_sample = sample.copy()
        
        # 获取数据目录和 JSON 文件路径
        data_dir = processed_sample.pop('_data_dir', '')
        json_file = processed_sample.pop('_json_file', '')
        
        # 修正 input_images 路径
        input_images = processed_sample.get('input_images', [])
        if isinstance(input_images, list):
            processed_sample['input_images'] = [
                fix_image_path(img, data_dir, json_file) for img in input_images
            ]
        else:
            processed_sample['input_images'] = []
        
        # 修正 output_image 路径
        output_image = processed_sample.get('output_image', '')
        if output_image:
            processed_sample['output_image'] = fix_image_path(output_image, data_dir, json_file)
        
        processed_samples.append(processed_sample)
    
    return jsonify({
        'samples': processed_samples,
        'current_page': current_page,
        'total_pages': total_pages,
        'total_samples': total_samples
    })


@app.route('/api/image')
def api_image():
    """提供图片访问的API"""
    import urllib.parse
    image_path = request.args.get('path', '')
    if not image_path:
        return jsonify({'error': 'No path provided'}), 400
    
    # 处理URL编码的路径
    image_path = urllib.parse.unquote(image_path)
    
    # 转换为Path对象
    full_path = Path(image_path)
    
    # 如果是绝对路径，直接使用
    if full_path.is_absolute():
        if full_path.exists() and full_path.is_file():
            directory = full_path.parent
            filename = full_path.name
            return send_from_directory(str(directory), filename)
    
    return jsonify({'error': 'Image not found'}), 404


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='数据可视化服务')
    parser.add_argument('--data_source', choices=['final', 'filter'], default='filter',
                        help='数据来源: final=data/final, filter=data/filter (默认: filter)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='自定义数据根目录路径（优先级高于 --data_source）')
    parser.add_argument('--port', type=int, default=8413, help='服务端口（默认: 8413）')
    args = parser.parse_args()

    print(f"Data directory: {DATA_DIR}")
    print(f"Templates directory: {TEMPLATES_DIR}")
    print(f"Static directory: {STATIC_DIR}")
    print(f"Samples per page: {SAMPLES_PER_PAGE}")
    print(f"Server will start on port {args.port}")
    
    app.run(host='0.0.0.0', port=args.port, debug=True)

