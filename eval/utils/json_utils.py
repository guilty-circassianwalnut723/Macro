"""
JSON utility for parsing LLM outputs.
"""
import json
import re
from typing import Any, Optional


def mllm_output_to_dict(text: str) -> Optional[Any]:
    """
    Parse JSON from LLM output text.
    
    Handles cases where JSON is embedded in markdown code blocks or raw text.
    
    Args:
        text: LLM output text
    
    Returns:
        Parsed JSON object or None on failure
    """
    if not text:
        return None
    
    # Try to extract JSON from code blocks
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'(\{.*\})',
        r'(\[.*\])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    
    # Try parsing the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
