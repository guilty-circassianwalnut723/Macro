#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data visualization script - supports both data/final and data/filter directory structures

Usage:
    # Visualize data/filter (default)
    python visualization.py

    # Visualize data/final
    python visualization.py --data_source final

    # Specify a custom data root directory
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

# Use local templates directory (static files are also placed here)
SCRIPT_DIR = Path(__file__).parent
TEMPLATES_DIR = SCRIPT_DIR / 'templates'
STATIC_DIR = SCRIPT_DIR / 'static'

app = Flask(__name__, 
            template_folder=str(TEMPLATES_DIR),
            static_folder=str(STATIC_DIR),
            static_url_path='/static')
CORS(app)

# ============================================================================
# Configuration constants
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Parse command-line arguments (processed at module load time, used by Flask routes)
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--data_source', choices=['final', 'filter'], default='filter',
                     help='Data source: final=data/final, filter=data/filter (default: filter)')
_parser.add_argument('--data_root', type=str, default=None,
                     help='Custom data root directory path (takes precedence over --data_source)')
_parser.add_argument('--port', type=int, default=8413, help='Server port (default: 8413)')
_args, _ = _parser.parse_known_args()

# Determine data directory
if _args.data_root:
    DATA_DIR = Path(_args.data_root).resolve()
elif _args.data_source == 'final':
    DATA_DIR = BASE_DIR / "data" / "final"
else:
    DATA_DIR = BASE_DIR / "data" / "filter"

# Also keep FINAL_DIR for image path resolution
FINAL_DIR = BASE_DIR / "data" / "final"

SAMPLES_PER_PAGE = 10  # Number of samples per page

# Task list
TASKS = ["spatial", "temporal", "customization", "illustration"]

# Spatial subtypes
SPATIAL_SUB_TYPES = ["indoor", "outdoor", "object"]

# Image count categories
IMAGE_COUNT_CATEGORIES = ["1-3", "4-5", "6-7", ">=8"]

# Split types
SPLIT_TYPES = ["train", "eval"]

# ============================================================================
# Data layout detection
# ============================================================================

def detect_layout(category_dir: Path) -> str:
    """
    Detect directory layout:
      - 'nested': has json/ subdirectory (data/final format)
      - 'flat':   JSON files directly under category_dir (data/filter format)
      - 'empty':  directory does not exist or is empty
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
# Pass/Fail filter configuration parameters (see scripts in filter directory)
# ============================================================================
# Temporal task threshold
TEMPORAL_SCORE_THRESHOLD = 6

# Customization task threshold
CONSISTENCY_SCORE_THRESHOLD = 6.0  # Each score in consistency_scores list must be >= this threshold
FOLLOWING_SCORE_THRESHOLD = 6.0    # following_score must be >= this threshold

# Illustration task threshold
TRAINING_SCORE_THRESHOLD = 6
GUIDANCE_SCORE_THRESHOLD = 6

# Spatial task: no filter criteria, all retained (all samples are pass)


# ============================================================================
# Pass/Fail determination functions
# ============================================================================

def is_sample_pass(sample: Dict[str, Any], task: str) -> bool:
    """
    Determine whether a sample passes the filter (pass)
    
    Args:
        sample: sample data
        task: task type
    
    Returns:
        True means pass, False means fail
    """
    # First check input image count, must not exceed 10
    input_images = sample.get("input_images", [])
    if isinstance(input_images, list) and len(input_images) > 10:
        return False
    
    if task == "spatial":
        # Spatial task: no filter criteria, all retained (but must satisfy image count limit)
        return True
    
    elif task == "temporal":
        # Temporal task: check temporal_score or score
        temporal_score = sample.get("temporal_score", sample.get("score", 0))
        return temporal_score >= TEMPORAL_SCORE_THRESHOLD
    
    elif task == "customization":
        # Customization task: check consistency_scores and following_score
        consistency_scores = sample.get("consistency_scores", [])
        following_score = sample.get("following_score")
        
        # Check consistency_scores: each score in the list must be >= threshold
        all_consistency_ok = True
        if isinstance(consistency_scores, list) and len(consistency_scores) > 0:
            for score in consistency_scores:
                if not isinstance(score, (int, float)) or score < CONSISTENCY_SCORE_THRESHOLD:
                    all_consistency_ok = False
                    break
        else:
            all_consistency_ok = False
        
        # Check following_score
        following_ok = isinstance(following_score, (int, float)) and following_score >= FOLLOWING_SCORE_THRESHOLD
        
        # Both conditions must be satisfied to pass
        return all_consistency_ok and following_ok
    
    elif task == "illustration":
        # Illustration task: check training_score and guidance_score, and image_contributions
        # Check image_contributions and calculate effective_image_count
        image_contributions = sample.get("image_contributions", [])
        if isinstance(image_contributions, list):
            effective_image_count = sum(1 for x in image_contributions if x is True)
        else:
            effective_image_count = sample.get("effective_image_count", 0)
        
        image_count = sample.get("image_count", 0)
        
        # Remove samples where effective_image_count or image_count is 0
        if effective_image_count == 0 or image_count == 0:
            return False
        
        # Check scores
        training_score = sample.get("suitable", sample.get("training_score", 0))
        guidance_score = sample.get("guidance_score", 0)
        
        # Both scores must be >= threshold
        return training_score >= TRAINING_SCORE_THRESHOLD and guidance_score >= GUIDANCE_SCORE_THRESHOLD
    
    else:
        # Unknown task type, default to True (but must satisfy image count limit)
        return True


# ============================================================================
# Data loading functions
# ============================================================================

def spatial_has_sub_types() -> bool:
    """
    Detect whether the spatial directory of the current data source has a sub_type layer (indoor/outdoor/object).
    - data/final/spatial: train/{sub_type}/{category}/  -> True
    - data/filter/spatial: train/{category}/            -> False
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
            # If no sub_type found under the first valid split directory, sub_type is not needed
            return False
    return False


def get_spatial_sub_types() -> List[str]:
    """Get available spatial subtypes. Returns empty list if current layout has no sub_type layer."""
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
    """Get all image count categories for the specified task, split type, and subtype (compatible with nested/flat layout)"""
    task_dir = DATA_DIR / task
    if not task_dir.exists():
        return []
    
    # For spatial without sub_type layer in current layout, ignore the passed sub_type
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
    """Get all JSON file paths for the specified conditions (compatible with nested/flat layout)"""
    task_dir = DATA_DIR / task
    if not task_dir.exists():
        return []
    
    # For spatial without sub_type layer in current layout, ignore the passed sub_type
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
    """Load all sample data for the specified conditions (deprecated, kept for compatibility)"""
    json_files = get_json_files_list(task, split_type, image_count_category, sub_type)
    
    samples = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Add file path info, used for locating images later
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
    Load sample data by page, only loading files needed for the current page
    
    Args:
        task: task type
        split_type: split type (train/eval)
        image_count_category: image count category
        page: page number
        per_page: samples per page
        sub_type: subtype (required for spatial task)
        filter_type: filter type, "all" means all, "pass" means samples that pass, "fail" means samples that do not pass
    
    Returns:
        (sample list, total pages, current page)
    """
    json_files = get_json_files_list(task, split_type, image_count_category, sub_type)
    
    if not json_files:
        return [], 1, 1
    
    # If filtering is needed, load all files first
    if filter_type != "all":
        all_samples = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Add file path info, used for locating images later
                    data['_json_file'] = str(json_file)
                    data['_data_dir'] = str(FINAL_DIR)
                    all_samples.append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        # Filter samples based on filter_type
        if filter_type == "pass":
            filtered_samples = [s for s in all_samples if is_sample_pass(s, task)]
        elif filter_type == "fail":
            filtered_samples = [s for s in all_samples if not is_sample_pass(s, task)]
        else:
            filtered_samples = all_samples
        
        # Calculate pagination
        total_files = len(filtered_samples)
        if total_files == 0:
            return [], 1, 1
        
        total_pages = (total_files + per_page - 1) // per_page
        page = max(1, min(page, total_pages))
        
        # Get samples for the current page
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        samples = filtered_samples[start_idx:end_idx]
        
        return samples, total_pages, page
    else:
        # No filtering needed, load directly by page
        total_files = len(json_files)
        total_pages = (total_files + per_page - 1) // per_page
        page = max(1, min(page, total_pages))
        
        # Only load files needed for the current page
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_files = json_files[start_idx:end_idx]
        
        samples = []
        for json_file in page_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Add file path info, used for locating images later
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
    """Paginate a sample list (deprecated, kept for compatibility)"""
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
    Fix image path.
    - Absolute path that exists: return directly
    - Relative path: prefer to resolve relative to BASE_DIR (data/final/... format); otherwise fall back to other strategies
    """
    if not image_path:
        return ""
    
    path_obj = Path(image_path)
    
    # Absolute path that exists, return directly
    if path_obj.is_absolute() and path_obj.exists():
        return str(path_obj)
    
    # Relative path: resolve relative to BASE_DIR (applicable for data/final/... and data/filter/... formats)
    if not path_obj.is_absolute():
        candidate = BASE_DIR / path_obj
        if candidate.exists():
            return str(candidate)
    
    # Try to match filename under data_dir (fallback)
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
    
    # Look in data/ subdirectory relative to the json file directory
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
    
    # If not found, return the original path (frontend handles load failure)
    return image_path



# ============================================================================
# Flask route functions
# ============================================================================

@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html')


@app.route('/api/tasks')
def api_tasks():
    """API to get all available tasks"""
    available_tasks = []
    for task in TASKS:
        task_dir = DATA_DIR / task
        if task_dir.exists():
            has_data = False
            for split_type in SPLIT_TYPES:
                if task == "spatial" and spatial_has_sub_types():
                    # final layout: need to search under sub_type subdirectory
                    for sub_type in SPATIAL_SUB_TYPES:
                        sub_dir = task_dir / split_type / sub_type
                        if sub_dir.exists() and any(sub_dir.iterdir()):
                            has_data = True
                            break
                else:
                    # filter layout or non-spatial task: search directly under split directory
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
    """Get spatial subtype list"""
    sub_types = get_spatial_sub_types()
    return jsonify(sub_types)


@app.route('/api/image_count_categories')
def api_image_count_categories():
    """Get image count category list for specified conditions"""
    task = request.args.get('task', '')
    split_type = request.args.get('split_type', '')
    sub_type = request.args.get('sub_type', '') or None
    
    if not task or not split_type:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    categories = get_image_count_categories(task, split_type, sub_type)
    return jsonify(categories)


@app.route('/api/samples')
def api_samples():
    """API to get paginated sample data"""
    task = request.args.get('task', '')
    split_type = request.args.get('split_type', '')
    image_count_category = request.args.get('image_count_category', '')
    sub_type = request.args.get('sub_type', '') or None
    page_param = request.args.get('page', '1')
    filter_type = request.args.get('filter', 'all')  # New filter parameter
    
    if not task or not split_type or not image_count_category:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    # Validate filter_type
    if filter_type not in ['all', 'pass', 'fail']:
        filter_type = 'all'
    
    try:
        page = int(page_param)
        if page < 1:
            page = 1
    except ValueError:
        page = 1
    
    # Load samples by page, only loading files for the current page
    current_page_samples, total_pages, current_page = load_samples_by_page(
        task, split_type, image_count_category, page, SAMPLES_PER_PAGE, sub_type, filter_type
    )
    
    # Get total sample count (for display)
    # If filtering was applied, need to recalculate total count
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
    
    # Process sample data, fix image paths
    processed_samples = []
    for sample in current_page_samples:
        processed_sample = sample.copy()
        
        # Get data directory and JSON file path
        data_dir = processed_sample.pop('_data_dir', '')
        json_file = processed_sample.pop('_json_file', '')
        
        # Fix input_images paths
        input_images = processed_sample.get('input_images', [])
        if isinstance(input_images, list):
            processed_sample['input_images'] = [
                fix_image_path(img, data_dir, json_file) for img in input_images
            ]
        else:
            processed_sample['input_images'] = []
        
        # Fix output_image path
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
    """API to serve image access"""
    import urllib.parse
    image_path = request.args.get('path', '')
    if not image_path:
        return jsonify({'error': 'No path provided'}), 400
    
    # Handle URL-encoded path
    image_path = urllib.parse.unquote(image_path)
    
    # Convert to Path object
    full_path = Path(image_path)
    
    # If absolute path, use directly
    if full_path.is_absolute():
        if full_path.exists() and full_path.is_file():
            directory = full_path.parent
            filename = full_path.name
            return send_from_directory(str(directory), filename)
    
    return jsonify({'error': 'Image not found'}), 404


# ============================================================================
# Main program entry point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data visualization service')
    parser.add_argument('--data_source', choices=['final', 'filter'], default='filter',
                        help='Data source: final=data/final, filter=data/filter (default: filter)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Custom data root directory path (takes precedence over --data_source)')
    parser.add_argument('--port', type=int, default=8413, help='Server port (default: 8413)')
    args = parser.parse_args()

    print(f"Data directory: {DATA_DIR}")
    print(f"Templates directory: {TEMPLATES_DIR}")
    print(f"Static directory: {STATIC_DIR}")
    print(f"Samples per page: {SAMPLES_PER_PAGE}")
    print(f"Server will start on port {args.port}")
    
    app.run(host='0.0.0.0', port=args.port, debug=True)

