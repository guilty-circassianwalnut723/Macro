"""
Minimal format conversion utility

Converts data from final format to minimal format, keeping only:
- task: task type
- idx: sample index
- prompt: prompt text
- input_images: input image list
- output_image: output image path
"""

import json
from pathlib import Path
from typing import Dict, Any, List


def convert_to_minimal(sample: Dict[str, Any], task: str, idx: int) -> Dict[str, Any]:
    """
    Convert a sample to minimal format
    
    Args:
        sample: original sample data
        task: task type
        idx: sample index
    
    Returns:
        sample data in minimal format
    """
    # Try to get prompt from multiple possible field names
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
    Read from JSON file and convert to minimal format
    
    Args:
        json_path: JSON file path
        task: task type
        idx: sample index
    
    Returns:
        sample data in minimal format
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        sample = json.load(f)
    
    return convert_to_minimal(sample, task, idx)

