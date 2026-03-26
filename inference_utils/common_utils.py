"""
通用的推理工具函数
用于 bagel/qwen/omnigen 等模型的推理脚本
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image


THIRDPARTY_TASKS_AVAILABLE = False
load_prompts_for_task = None


# 支持的task列表
SUPPORTED_TASKS = ["customization", "illustration", "spatial", "temporal"]

# 第三方任务列表（已移除，仅保留4类主要任务）
THIRDPARTY_TASKS = []

# 支持的image num category
IMAGE_NUM_CATEGORIES = ["1-3", "4-5", "6-7", ">=8"]


def get_data_root(macro_dir: Optional[Path] = None) -> Path:
    """
    获取数据根目录
    
    Args:
        macro_dir: Macro 根目录路径，如果为 None，则自动推断
        
    Returns:
        数据根目录路径
    """
    if macro_dir is None:
        # 自动推断：从当前文件位置向上找到 Macro 根目录
        current_file = Path(__file__)
        macro_dir = current_file.parent.parent
    
    return macro_dir / "data" / "filter"  # data/filter/{task}/eval/


def parse_image_num_category(category: str) -> Tuple[int, Optional[int]]:
    """
    解析image num category字符串
    
    Args:
        category: category字符串，如 "1-3", "4-5", ">=8"
    
    Returns:
        (min_num, max_num) 元组，max_num为None表示无上限
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
    检查图像数量是否匹配category
    
    Args:
        num_images: 图像数量
        category: category字符串
    
    Returns:
        是否匹配
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
    从filter目录加载eval数据
    
    Args:
        task: 任务类型 (customization, illustration, spatial, temporal)
        image_num_category: 图像数量类别 (1-3, 4-5, 6-7, >=8, all), None表示all
        data_root: 数据根目录，如果为None，则使用默认路径
        macro_dir: Macro 根目录路径，用于推断数据根目录
        
    Returns:
        样本列表，每个样本包含 task, idx, prompt, input_images, output_image
    """
    if data_root is None:
        data_root = get_data_root(macro_dir)
    
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}. Supported tasks: {SUPPORTED_TASKS}")
    
    eval_dir = data_root / task / "eval"
    if not eval_dir.exists():
        raise ValueError(f"Eval directory does not exist: {eval_dir}")
    
    samples = []
    
    # 如果指定了category，只加载该category的数据
    if image_num_category and image_num_category != "all":
        if image_num_category not in IMAGE_NUM_CATEGORIES:
            raise ValueError(f"Unsupported image_num_category: {image_num_category}")
        
        category_dir = eval_dir / image_num_category
        if category_dir.exists():
            samples.extend(_load_samples_from_category(category_dir, task, image_num_category))
    else:
        # 加载所有category的数据
        for category in IMAGE_NUM_CATEGORIES:
            category_dir = eval_dir / category
            if category_dir.exists():
                samples.extend(_load_samples_from_category(category_dir, task, category))
    
    # 按idx排序
    samples.sort(key=lambda x: (x.get("category", ""), x.get("idx", 0)))
    
    return samples


def _load_samples_from_category(
    category_dir: Path,
    task: str,
    category: str
) -> List[Dict[str, Any]]:
    """
    从特定category目录加载样本
    
    Args:
        category_dir: category目录路径
        task: 任务类型
        category: category名称
    
    Returns:
        样本列表
    """
    samples = []
    
    # 查找所有JSON文件
    json_files = sorted(category_dir.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                sample = json.load(f)
                # 确保包含category信息
                sample["category"] = category
                samples.append(sample)
        except Exception as e:
            print(f"警告: 读取JSON文件失败 {json_file}: {e}")
            continue
    
    return samples


def filter_samples_by_image_num(
    samples: List[Dict[str, Any]],
    image_num_category: str
) -> List[Dict[str, Any]]:
    """
    根据image num category过滤样本
    
    Args:
        samples: 样本列表
        image_num_category: 图像数量类别
    
    Returns:
        过滤后的样本列表
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
    获取可用的task列表
    
    Args:
        data_root: 数据根目录
        macro_dir: Macro 根目录路径
        
    Returns:
        可用task列表
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
    获取指定task的可用category列表
    
    Args:
        task: 任务类型
        data_root: 数据根目录
        macro_dir: Macro 根目录路径
        
    Returns:
        可用category列表
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
    检查样本是否已经生成（断点续传检查）
    
    严格检查：
    1. 图像文件和JSON文件都必须存在
    2. 图像文件必须可以正确打开和读取
    3. 图像必须有有效的尺寸（宽高都大于0）
    4. JSON文件必须可以正确解析
    5. JSON文件必须包含必要的字段
    
    Args:
        output_dir: 输出目录
        idx: 样本索引
    
    Returns:
        如果样本已存在且文件可读，返回True；否则返回False
    """
    image_file = output_dir / f"{idx:08d}.jpg"
    json_file = output_dir / f"{idx:08d}.json"
    
    # 检查文件是否存在
    if not image_file.exists() or not json_file.exists():
        return False
    
    # 检查图像文件是否可读（不仅要verify，还要实际加载检查）
    try:
        with Image.open(image_file) as img:
            # 验证图像完整性
            img.verify()
        
        # 重新打开图像以检查尺寸（verify后需要重新打开）
        with Image.open(image_file) as img:
            # 转换为RGB以确保可以读取
            img = img.convert("RGB")
            width, height = img.size
            # 检查图像尺寸是否有效
            if width <= 0 or height <= 0:
                return False
            # 尝试读取一个像素以确保图像完全可读
            img.load()
    except Exception as e:
        # 如果图像无法读取，返回False
        return False
    
    # 检查JSON文件是否可读
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 检查必要的字段
            if 'output_image' not in data:
                return False
            # 验证output_image路径是否指向实际存在的文件
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
    统一的数据加载接口，支持所有任务类型（常规任务和第三方任务）
    
    Args:
        task: 任务类型 (customization, illustration, spatial, temporal, omnicontext, geneval, dpg)
        image_num_category: 图像数量类别 (1-3, 4-5, 6-7, >=8, all)，仅对常规任务有效
        data_root: 数据根目录，如果为None，则使用默认路径
        macro_dir: Macro 根目录路径，用于推断数据根目录
        use_refine_prompt: 对于 geneval，是否使用 refine prompt
        
    Returns:
        样本列表，每个样本包含：
        - 常规任务: task, idx, prompt, input_images, output_image, category
        - 第三方任务: idx, prompt/instruction, input_images, task, 以及其他任务特定字段
    """
    # 第三方任务
    if task in THIRDPARTY_TASKS:
        if not THIRDPARTY_TASKS_AVAILABLE:
            raise ImportError("第三方任务模块未找到，请检查 inference_utils/thirdparty_tasks.py")
        
        samples = load_prompts_for_task(task, data_root, use_refine_prompt=use_refine_prompt)
        
        # 确保所有样本都有统一的字段名
        for sample in samples:
            # 确保有 prompt 字段（从 instruction 或其他字段获取）
            if "prompt" not in sample:
                if "instruction" in sample:
                    sample["prompt"] = sample["instruction"]
                elif "text" in sample:
                    sample["prompt"] = sample["text"]
            
            # 确保有 instruction 字段（从 prompt 获取）
            if "instruction" not in sample:
                sample["instruction"] = sample.get("prompt", "")
            
            # 确保有 input_images 字段（从 images 或其他字段获取）
            if "input_images" not in sample:
                if "images" in sample:
                    sample["input_images"] = sample["images"]
                elif "image_paths" in sample:
                    sample["input_images"] = sample["image_paths"]
                else:
                    sample["input_images"] = []
            
            # 确保有 task 字段（用于识别任务类型）
            if "task" not in sample:
                sample["task"] = task
            
            # 确保有 idx 字段
            if "idx" not in sample:
                # 尝试从其他字段获取
                if "id" in sample:
                    sample["idx"] = sample["id"]
                elif "index" in sample:
                    sample["idx"] = sample["index"]
                else:
                    # 使用列表索引
                    sample["idx"] = samples.index(sample)
        
        return samples
    
    # 常规任务
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
    统一保存样本数据接口，支持所有任务类型
    
    Args:
        output_dir: 输出目录
        idx: 样本索引
        sample: 原始样本数据
        output_image_path: 生成的图像路径
        target_image_path: 目标图像路径（可选）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存JSON文件
    output_json = sample.copy()
    output_json["output_image"] = str(output_image_path)
    if target_image_path:
        output_json["target_image"] = target_image_path
    elif "output_image" in sample:
        # 如果原始样本中有 output_image，作为 target_image
        output_json["target_image"] = sample.get("output_image", "")
    
    # 如果指定了 seed，在 JSON 中记录
    if seed is not None:
        output_json["seed"] = seed
    
    # 如果指定了 seed，在 JSON 中记录
    if seed is not None:
        output_json["seed"] = seed
    
    # 确保有 idx（与 bagel/vis 等一致，便于评测与可视化）
    if "idx" not in output_json:
        output_json["idx"] = idx

    # 确保有 prompt 和 instruction 字段（兼容性）
    if "prompt" not in output_json and "instruction" in output_json:
        output_json["prompt"] = output_json["instruction"]
    if "instruction" not in output_json and "prompt" in output_json:
        output_json["instruction"] = output_json["prompt"]
    
    # 处理 input_images 字段：将 PIL Image 对象转换为路径字符串或移除
    if "input_images" in output_json:
        serializable_input_images = []
        for img_item in output_json["input_images"]:
            if isinstance(img_item, (str, Path)):
                # 已经是路径字符串
                serializable_input_images.append(str(img_item))
            elif hasattr(img_item, 'filename') and img_item.filename:
                # PIL Image 对象，尝试获取文件名
                serializable_input_images.append(str(img_item.filename))
            else:
                # 无法序列化的对象，跳过或使用占位符
                # 对于第三方任务，如果无法获取路径，使用空列表或占位符
                pass
        output_json["input_images"] = serializable_input_images
    
    # 清理其他可能包含不可序列化对象的字段
    # 创建一个可序列化的副本
    def make_json_serializable(obj):
        """递归地将对象转换为可 JSON 序列化的格式"""
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            # 尝试获取对象的字符串表示
            return str(obj)
        else:
            # 对于其他类型，尝试转换为字符串
            return str(obj)
    
    # 对 output_json 进行序列化处理
    output_json = make_json_serializable(output_json)
    
    # 保存JSON文件
    # 对于 geneval 和 dpg，如果指定了 seed，文件名包含 seed 信息
    if seed is not None:
        json_file = output_dir / f"{idx:08d}_seed{seed:02d}.json"
    else:
        json_file = output_dir / f"{idx:08d}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    
    # 设置文件权限
    os.chmod(json_file, 0o777)
