#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal任务评分模块

提供temporal任务的评分函数（GPT和Gemini）
"""

import os
import sys
import json
import time
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image

# Add Macro root to path for importing utils
script_dir = Path(__file__).resolve()
macro_dir = script_dir.parents[2]  # eval/score -> eval -> Macro
if str(macro_dir) not in sys.path:
    sys.path.insert(0, str(macro_dir))

# 尝试导入utils模块
from utils.openai_utils import ask_gpt4o
from utils.json_utils import mllm_output_to_dict
from api_generator.text_generator.gemini_api import GeminiAPIGenerator

# ============================================================================
# PROMPT 配置
# ============================================================================
EVALUATION_PROMPT = """You are a strict Video Continuity Specialist and QA Expert. Your task is to evaluate the quality of a generated frame in a temporal sequence.

**1. THE INSTRUCTION**
The specific action or change required for this frame:
"{instruction}"

**2. THE IMAGES**
You are provided with a sequence of images:
{image_descriptions}

**3. EVALUATION TASK**
Compare the **Generated Image** against the **Reference Images** (especially the last frame) and the **Instruction**.
Return a JSON object:
{{
  "context_consistency_score": <score>,
  "image_sequence_consistency_score": <score>,
  "overall_reasoning": "Concise analysis of flaws. (Max 100 words)"
}}

**4. SCORING CRITERIA**
Start at 10. Deduct points heavily for any failure.

**Metric 1: Context & Action Logic Score (0-10)**
Does the image correctly execute the text instruction and logically follow the previous frame's motion?

*   **10 (Perfect):** The specific action in the instruction is clearly performed. The motion is physically plausible and connects smoothly to the previous frame.
*   **8-9 (Good):** Instruction followed, but the motion feels slightly stiff or the magnitude of change is slightly off.
*   **5-7 (Moderate):** The instruction is partially ignored, OR the motion is abrupt/unnatural (teleporting objects, impossible physics).
*   **2-4 (Poor):** The instruction is mostly ignored, or the action contradicts the previous frame.
*   **0-1 (Failure):** Complete hallucination or irrelevant content.

**Focus on:**
- **Adherence:** Did the requested event actually happen?
- **Physics:** Is the movement logical relative to the last frame?

**Metric 2: Visual Consistency & Identity Score (0-10)**
Does the image strictly preserve the identity, style, and background details from the reference images?

*   **10 (Perfect):** Indistinguishable consistency. The character, object, and background details are identical to the reference.
*   **8-9 (Good):** Minor lighting or shading differences, but the subject's identity is clearly preserved.
*   **5-7 (Moderate):** Noticeable "Identity Drift" (face looks different, clothes change details) or background objects shift/disappear.
*   **2-4 (Poor):** Major inconsistencies. It looks like a different character or a different location.
*   **0-1 (Failure):** Completely different style or content.

**Focus on:**
- **Identity:** Do faces/objects look exactly the same? (Penalize heavily for face morphing).
- **Stability:** Does the background stay stable? (Penalize for flickering or random artifacts).

**Overall Reasoning Guide:**
State clearly if the score deduction is due to "Instruction Failure" (Metric 1) or "Visual Inconsistency" (Metric 2). Be critical of small details like changing facial features or disappearing props.
"""

# ============================================================================
# 配置常量
# ============================================================================
GPT_CONFIG = {
    "url": os.environ.get("OPENAI_URL", "https://api.openai.com/v1/chat/completions"),
    "key": os.environ.get("OPENAI_KEY", "")
}

GEMINI_CONFIG = {
    "api_key": os.environ.get("GEMINI_API_KEY", ""),
    "model_name": "gemini-2.0-flash-preview",
    "max_try": 100
}

MAX_RETRIES = 10
RETRY_DELAY = 2
TIMEOUT = 60


# ============================================================================
# Prompt构建函数
# ============================================================================
def build_prompt(instruction: str, image_descriptions: str) -> str:
    return EVALUATION_PROMPT.format(
        instruction=instruction,
        image_descriptions=image_descriptions
    )


def build_image_descriptions(reference_images: List[str], generated_image: str) -> str:
    descriptions = []
    
    if reference_images:
        for idx, ref_img in enumerate(reference_images, 1):
            descriptions.append(f"- <IMAGE_TOKEN> Reference Image {idx}: Previous frame {idx} in the temporal sequence")
    
    descriptions.append(f"- <IMAGE_TOKEN> Generated Image: The AI model's output for the next frame in the sequence")
    
    return "\n".join(descriptions)


# ============================================================================
# 评分函数
# ============================================================================
def evaluate_with_gpt(sample_data: Dict[str, Any], sample_id: str = "") -> Optional[Dict[str, Any]]:
    instruction = sample_data.get('instruction', '')
    reference_images = sample_data.get('input_images', [])
    generated_image = sample_data.get('output_image', '')
    
    if not all([instruction, generated_image]):
        print(f"[GPT] {sample_id}: Missing required data for GPT evaluation")
        return None
    
    all_images = reference_images + [generated_image] if reference_images else [generated_image]
    image_descriptions = build_image_descriptions(reference_images, generated_image)
    prompt = build_prompt(instruction, image_descriptions)
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1:
                print(f"[GPT] {sample_id}: 重试第 {attempt} 次（共 {MAX_RETRIES} 次）...")
            
            response = ask_gpt4o(all_images, prompt, GPT_CONFIG["url"], GPT_CONFIG["key"])
            if not response:
                print(f"[GPT] {sample_id}: 尝试 {attempt}/{MAX_RETRIES}: 空响应")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                return None
            
            result = mllm_output_to_dict(response)
            if result and isinstance(result, dict):
                if 'context_consistency_score' in result and 'image_sequence_consistency_score' in result:
                    if attempt > 1:
                        print(f"[GPT] {sample_id}: 第 {attempt} 次尝试成功")
                    return result
                else:
                    print(f"[GPT] {sample_id}: 尝试 {attempt}/{MAX_RETRIES}: 结果格式不正确 - {list(result.keys())}")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                        continue
                    return None
            else:
                print(f"[GPT] {sample_id}: 尝试 {attempt}/{MAX_RETRIES}: 解析失败 - {response[:200]}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                return None
        except Exception as e:
            print(f"[GPT] {sample_id}: 尝试 {attempt}/{MAX_RETRIES}: 错误 - {e}")
            if attempt < MAX_RETRIES:
                traceback.print_exc()
                time.sleep(RETRY_DELAY)
                continue
            else:
                traceback.print_exc()
                return None
    
    return None


def evaluate_with_gemini(sample_data: Dict[str, Any], generator: 'GeminiAPIGenerator', sample_id: str = "") -> Optional[Dict[str, Any]]:
    instruction = sample_data.get('instruction', '')
    reference_images = sample_data.get('input_images', [])
    generated_image = sample_data.get('output_image', '')
    
    if not all([instruction, generated_image]):
        print(f"[Gemini] {sample_id}: Missing required data for Gemini evaluation")
        return None
    
    all_images = []
    for img_path in (reference_images + [generated_image] if reference_images else [generated_image]):
        try:
            img = Image.open(img_path)
            all_images.append(img)
        except Exception as e:
            print(f"[Gemini] {sample_id}: Error loading image {img_path}: {e}")
            return None
    
    image_descriptions = build_image_descriptions(reference_images, generated_image)
    prompt = build_prompt(instruction, image_descriptions)
    
    response_format = {
        "context_consistency_score": "float",
        "image_sequence_consistency_score": "float",
        "overall_reasoning": "str"
    }
    
    try:
        response = generator.gen_response(
            prompt=prompt,
            response_format=response_format,
            images=all_images,
            think=False
        )
        
        if response is None:
            print(f"[Gemini] {sample_id}: 空响应")
            return None
        
        if isinstance(response, dict):
            if 'context_consistency_score' in response and 'image_sequence_consistency_score' in response:
                return response
            else:
                print(f"[Gemini] {sample_id}: 结果格式不正确 - {list(response.keys())}")
                return None
        else:
            print(f"[Gemini] {sample_id}: 意外的响应类型 - {type(response)}")
            return None
    except Exception as e:
        print(f"[Gemini] {sample_id}: 错误 - {e}")
        traceback.print_exc()
        return None


def is_score_valid(score_data: Any) -> bool:
    if score_data is None:
        return False
    if isinstance(score_data, dict):
        if 'context_consistency_score' in score_data and 'image_sequence_consistency_score' in score_data:
            return True
    return False
