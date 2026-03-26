"""
最简格式转换工具

将final格式的数据转换为最简格式，只保留:
- task: 任务类型
- idx: 样本索引
- prompt: 提示词
- input_images: 输入图像列表
- output_image: 输出图像路径
"""

import json
from pathlib import Path
from typing import Dict, Any, List


def convert_to_minimal(sample: Dict[str, Any], task: str, idx: int) -> Dict[str, Any]:
    """
    将样本转换为最简格式
    
    Args:
        sample: 原始样本数据
        task: 任务类型
        idx: 样本索引
    
    Returns:
        最简格式的样本数据
    """
    # 尝试从多个可能的字段名获取 prompt
    prompt = (
        sample.get("instruction") or 
        sample.get("prompt") or 
        sample.get("text") or 
        ""
    )
    
    minimal = {
        "task": task,
        "idx": idx,
        "prompt": prompt,
        "input_images": sample.get("input_images", []),
        "output_image": sample.get("output_image", "")
    }
    
    return minimal


def convert_json_file(json_path: Path, task: str, idx: int) -> Dict[str, Any]:
    """
    从JSON文件读取并转换为最简格式
    
    Args:
        json_path: JSON文件路径
        task: 任务类型
        idx: 样本索引
    
    Returns:
        最简格式的样本数据
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        sample = json.load(f)
    
    return convert_to_minimal(sample, task, idx)

