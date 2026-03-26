"""
OpenAI API utility for evaluation.
"""
import os
import json
import base64
import time
from typing import List, Optional
from pathlib import Path


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_gpt4o(
    image_paths: List[str],
    prompt: str,
    api_url: str,
    api_key: str,
    model: str = "gpt-4o",
    max_tokens: int = 1024,
    timeout: int = 60,
) -> Optional[str]:
    """
    Call GPT-4o API with images and prompt.
    
    Args:
        image_paths: List of image file paths
        prompt: Text prompt
        api_url: API endpoint URL
        api_key: API key
        model: Model name
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
    
    Returns:
        Response text or None on failure
    """
    try:
        import requests
        
        # Build messages with images
        content = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Warning: image not found: {img_path}")
                continue
            b64 = image_to_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        content.append({"type": "text", "text": prompt})
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"GPT API error: {e}")
        return None
