#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Illustration任务评分模块

提供illustration任务的评分函数（GPT和Gemini）
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
EVALUATION_PROMPT = """You are an expert Visual Communication Specialist and Content Quality Auditor. You are capable of evaluating various types of visual content, ranging from artistic illustrations and diary entries to functional visuals like manuals, advertisements, and presentation slides.

**1. THE TASK CONTEXT (Read this first)**
This is a **Visual Content Generation Task**. The AI model was given a text description and reference images, and asked to generate an image output that is suitable to be placed after the context.

The text description given to the model:
"{instruction}"

**2. THE IMAGES**
You will see the following images:
{image_descriptions}

**3. EVALUATION TASK**
Evaluate the **Generated Image** against the **Text Description** and **Reference Images** (if any).
Your response must be a JSON object:
{{
  "text_consistency_score": <score>,
  "image_consistency_score": <score>,
  "overall_reasoning": "Critical analysis explaining the deductions. (Maximum 100 words)"
}}

**4. SCORING CRITERIA**
Start from a perfect score and deduct points for flaws.

**Metric 1: Text Consistency & Content Accuracy (0-10)**
Does the generated image accurately reflect the content, format, and intent described in the text?

*   **10 (Perfect):** The image perfectly matches the context in terms of **content, format, and style**, well integrated with the context.
*   **8-9 (High Quality):** The image follows the context well with only minor deviations in content, format, or style.
*   **5-7 (Moderate):** The image captures the main topic but fails in specific format or context requirements.
*   **2-4 (Poor):** The image fails to represent the core content or completely ignores the format instructions.
*   **0-1 (Failure):** The image is irrelevant to the text.

**Metric 2: Reference Consistency (0-10)**
Does the generated image correctly utilize the visual information provided in the reference images (if any)?

**Important:** Reference images are provided to assist in understanding the **subject matter** (e.g., what a specific product or character looks like). The generated image should maintain the identity/features of the subject from the reference, but **NOT necessarily the style**, unless the text explicitly asks to copy the style.

If reference images are provided and relevant to the context:
*   **10 (Perfect):** The generated image correctly features the **subjects/objects** from the reference images. Key visual features (color, shape, identifying marks) are preserved and integrated into the new context/format naturally.
*   **8-9 (High Quality):** The subject is clearly recognizable as the one from the reference, with only minor deviations in non-essential details.
*   **5-7 (Moderate):** The subject somewhat resembles the reference, but significant features are distorted or missing.
*   **2-4 (Poor):** The subject is barely recognizable or looks like a generic version rather than the specific one in the reference.
*   **0-1 (Failure):** The generated image ignores the reference images completely.

If no reference images are provided or not relevant to the context:
*   **10 (Perfect):** The image has perfect **internal consistency**. Logic, lighting, and spatial relationships make sense within the context of the requested format.
*   **8-9 (High Quality):** Good internal consistency with minor logical flaws.
*   **5-7 (Moderate):** Noticeable internal contradictions or broken visual logic.
*   **0-4 (Poor/Failure):** The image is incoherent or chaotic.

**Overall Reasoning Guide:**
Briefly explain the score.
- If the content is wrong, penalize Text Consistency.
- If the relevant subject/object looks different from the reference (e.g., wrong color, wrong features), penalize Image Consistency.
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
            descriptions.append(f"- <IMAGE_TOKEN> Reference Image {idx}: Previous image in the sequence")
    
    descriptions.append(f"- <IMAGE_TOKEN> Generated Image: The AI model's output for the illustration task")
    
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
                if 'text_consistency_score' in result and 'image_consistency_score' in result:
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
        "text_consistency_score": "float",
        "image_consistency_score": "float",
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
            if 'text_consistency_score' in response and 'image_consistency_score' in response:
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
        if 'text_consistency_score' in score_data and 'image_consistency_score' in score_data:
            return True
    return False
