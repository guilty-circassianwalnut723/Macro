"""
通用工具函数
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Set, Optional, Union, Tuple, List


def get_image_count_category(image_count: int) -> str:
    """根据图像数量返回类别目录名"""
    if image_count <= 3:
        return "1-3"
    elif image_count <= 5:
        return "4-5"
    elif image_count <= 7:
        return "6-7"
    else:
        return ">=8"


def load_generated_ids(
    task_dir: Path,
    split_type: str,
    image_count_category: str,
    sub_type: Optional[str] = None
) -> Set[str]:
    """
    加载已生成的样本唯一识别编号集合
    
    Args:
        task_dir: task目录，例如 final/customization
        split_type: "train" 或 "eval"
        image_count_category: 图像数量类别，例如 "1-3"
        sub_type: 子类型（可选）
    
    Returns:
        已生成的唯一识别编号集合
    """
    generated_ids = set()

    if sub_type:
        json_dir = task_dir / split_type / sub_type / image_count_category / "json"
    else:
        json_dir = task_dir / split_type / image_count_category / "json"
    if not json_dir.exists():
        return generated_ids

    for json_file in json_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                unique_id = data.get('unique_id')
                if unique_id:
                    generated_ids.add(unique_id)
        except Exception as e:
            print(f"警告: 读取JSON文件失败 {json_file}: {e}")
            continue

    return generated_ids


def load_all_generated_ids(
    task_dir: Path,
    split_type: str,
    sub_type: Optional[str] = None
) -> Set[str]:
    """
    加载整个 split_type 目录下所有 category 的已生成样本唯一识别编号集合
    """
    generated_ids = set()

    if sub_type:
        base_dir = task_dir / split_type / sub_type
    else:
        base_dir = task_dir / split_type

    if not base_dir.exists():
        return generated_ids

    for category_dir in base_dir.glob("*"):
        if not category_dir.is_dir():
            continue

        json_dir = category_dir / "json"
        if not json_dir.exists():
            continue

        for json_file in json_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    unique_id = data.get('unique_id')
                    if unique_id:
                        generated_ids.add(unique_id)
            except Exception as e:
                print(f"警告: 读取JSON文件失败 {json_file}: {e}")
                continue

    return generated_ids


# 控制是否使用 MD5 哈希作为 unique_id
USE_MD5_HASH = False
SAVE_ORIGINAL_STRING = False


def save_sample_data(
    task_dir: Path,
    split_type: str,
    image_count_category: str,
    idx: int,
    unique_id: Union[str, Tuple[str, str]],
    json_data: Dict,
    image_files: Optional[Dict] = None,
    sub_type: Optional[str] = None
) -> bool:
    """
    保存样本数据到指定目录
    """
    try:
        if sub_type:
            data_dir = task_dir / split_type / sub_type / image_count_category / "data" / f"{idx:08d}"
            json_dir = task_dir / split_type / sub_type / image_count_category / "json"
        else:
            data_dir = task_dir / split_type / image_count_category / "data" / f"{idx:08d}"
            json_dir = task_dir / split_type / image_count_category / "json"

        data_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        if image_files:
            import shutil
            for filename, source in image_files.items():
                dest_path = data_dir / filename

                if isinstance(source, Path):
                    if source.exists():
                        try:
                            if source.resolve() == dest_path.resolve():
                                pass
                            else:
                                shutil.copy2(source, dest_path)
                        except (OSError, ValueError):
                            if str(source) != str(dest_path):
                                shutil.copy2(source, dest_path)
                        if 'input_images' in json_data and isinstance(json_data['input_images'], list):
                            for i, img_path in enumerate(json_data['input_images']):
                                if str(source) == str(img_path) or source.name in str(img_path):
                                    json_data['input_images'][i] = str(dest_path)
                        if 'output_image' in json_data:
                            if str(source) == str(json_data['output_image']) or source.name in str(json_data['output_image']):
                                json_data['output_image'] = str(dest_path)
                else:
                    try:
                        if hasattr(source, 'save'):
                            source.save(dest_path, quality=95)
                        else:
                            import numpy as np
                            from PIL import Image
                            if isinstance(source, np.ndarray):
                                img = Image.fromarray(source)
                                img.save(dest_path, quality=95)
                            else:
                                raise ValueError(f"无法保存图像文件 {filename}，不支持的类型: {type(source)}")
                    except Exception as e:
                        print(f"警告: 保存图像文件 {filename} 失败: {e}")

        if isinstance(unique_id, tuple):
            md5_hash, original_string = unique_id
            json_data['unique_id'] = md5_hash
            if SAVE_ORIGINAL_STRING:
                json_data['original_unique_id'] = original_string
        else:
            json_data['unique_id'] = unique_id
            if not USE_MD5_HASH and SAVE_ORIGINAL_STRING:
                json_data['original_unique_id'] = unique_id

        json_data['idx'] = idx

        json_path = json_dir / f"{idx:08d}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        print(f"错误: 保存样本数据失败 idx={idx}, unique_id={unique_id}: {e}")
        return False


def get_combination_key(files: List[str]) -> str:
    """
    生成图像组合的唯一键（用于去重）
    """
    sorted_files = sorted(files)
    key_str = "|".join(sorted_files)
    return hashlib.md5(key_str.encode()).hexdigest()


def generate_unique_id(task: str, return_original: bool = False, **kwargs) -> Union[str, Tuple[str, str]]:
    """
    生成唯一识别编号
    """
    if task == "customization":
        combination_key = kwargs.get('combination_key', '')
        if combination_key:
            original_id = f"customization_{combination_key}"
        else:
            original_id = f"customization_{kwargs.get('idx', 'unknown')}"

    elif task == "illustration":
        source_file = kwargs.get('source_file', 'unknown')
        source_line = kwargs.get('source_line', -1)
        true_index = kwargs.get('true_index', 0)
        source_basename = Path(source_file).stem if source_file != 'unknown' else 'unknown'
        original_id = f"illustration_{source_basename}_{source_line:08d}_{true_index:02d}"

    elif task == "spatial":
        sub_type = kwargs.get('sub_type', 'unknown')
        source = kwargs.get('source', 'unknown')
        view_config = kwargs.get('view_config', 'unknown')
        if hasattr(source, '__fspath__') or hasattr(source, 'parts'):
            source = str(source)
        original_id = f"spatial_{sub_type}_{source}_{view_config}"

    elif task == "temporal":
        video_path = kwargs.get('video_path', 'unknown')
        frame_indices = kwargs.get('frame_indices', [])
        video_name = Path(video_path).stem if video_path != 'unknown' else 'unknown'
        frames_str = "_".join([str(i) for i in sorted(frame_indices)]) if frame_indices else "unknown"
        original_id = f"temporal_{video_name}_{frames_str}"
    else:
        original_id = f"{task}_{hashlib.md5(str(kwargs).encode()).hexdigest()[:8]}"

    if USE_MD5_HASH:
        md5_hash = hashlib.md5(original_id.encode()).hexdigest()
        if return_original:
            return (md5_hash, original_id)
        return md5_hash
    else:
        return original_id
