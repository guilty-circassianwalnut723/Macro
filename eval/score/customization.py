#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customization任务评分模块

提供customization任务的评分函数（GPT和Gemini）
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

# 尝试导入utils模块，如果不存在则使用本地实现
from utils.openai_utils import ask_gpt4o
from utils.json_utils import mllm_output_to_dict
from api_generator.text_generator.gemini_api import GeminiAPIGenerator

# ============================================================================
# PROMPT 配置（放在脚本最上方，方便修改）
# ============================================================================
EVALUATION_PROMPT = """You are a meticulous Digital Art Critic and Quality Assurance Specialist. You are known for your extremely high standards and attention to microscopic details.

**1. THE INSTRUCTION (Read this first)**
The AI model was given this specific customization instruction:
"{instruction}"

**2. THE IMAGES**
You will be provided with a sequence of images.
*   **Reference Images:** The first few images showing the subject/object to be customized.
*   **Generated Image:** The **LAST** image in the sequence. This is the output you must evaluate.
{image_descriptions}

**3. EVALUATION TASK**
Evaluate the **Generated Image** (the last image) against the **Reference Images** and the **Instruction**.
Your response must be a JSON object:
{{
  "consistency_scores": [<score1>, <score2>, ..., <scoreN>],
  "following_score": <score>,
  "overall_reasoning": "Critical analysis explaining the deductions. First analyze the Instruction constraints, then the Identity retention. (Maximum 150 words)"
}}

**4. CRITICAL CONFLICT RESOLUTION**
*   **Style/Attribute Changes:** If the instruction explicitly asks to change a feature (e.g., "make him old", "turn into a cartoon", "change hair color"), **DO NOT penalize** the Consistency Score for these specific requested changes.
*   **Unintended Changes:** Penalize strictly if features change *without* being asked (e.g., the instruction is "add a hat", but the face structure changes completely).

**5. SCORING CRITERIA**
Start from a perfect score and deduct points for *any* flaw. Be unforgiving.

**Metric 1: Consistency Scores (0-10, one for each reference image)**
Does the generated image preserve the identity of subjects/objects from each reference image, considering the instruction constraints?

For each reference image, evaluate:
*   **10 (Perfect):** Identity is perfectly preserved. If style transfer was requested, the subject is instantly recognizable in the new style.
*   **8-9 (High Quality):** Identity is clearly recognizable. Minor details (e.g., ear shape, exact hair strand pattern) might differ slightly, or the likeness is 90% accurate.
*   **5-7 (Moderate):** Resembles the reference, but looks like a "sibling" or a "look-alike" rather than the exact same person/object. Key facial ratios are slightly off.
*   **2-4 (Poor):** Vague resemblance. Only general features (e.g., gender, hair color) match, but the specific identity is lost.
*   **0-1 (Failure):** Completely different person or object.

**Focus on:**
- **Faces:** Eye shape, nose structure, mouth width, jawline.
- **Objects:** distinctive markings, logos, specific geometry.
- **Clothing:** Should match unless the instruction implies changing the outfit or environment.

**Metric 2: Following Score (0-10)**
Does the generated image fulfill the editing instruction accurately?

*   **10 (Perfect):** All aspects of the instruction are executed perfectly. No hallucinations (unwanted elements) and no missing requirements.
*   **8-9 (High Quality):** Follows the main instruction well, but misses a tiny detail or the aesthetic quality of the edit is slightly unnatural.
*   **5-7 (Moderate):** Follows the general idea (e.g., "put on a hat") but gets the details wrong (e.g., wrong type of hat) or ignores secondary constraints.
*   **2-4 (Poor):** Misses the main point of the instruction or hallucinates significant unrelated content.
*   **0-1 (Failure):** Completely ignores the instruction.

**Overall Reasoning Guide:**
Provide a concise Chain of Thought.
1. State what the instruction demanded.
2. Check if those demands are met (Following).
3. Check if the non-edited parts match the Reference (Consistency).
4. Explain specific defects that led to score deductions.
"""

# ============================================================================
# 配置常量
# ============================================================================
# LLM配置
GPT_CONFIG = {
    "url": os.environ.get("OPENAI_URL", "https://api.openai.com/v1/chat/completions"),
    "key": os.environ.get("OPENAI_KEY", "")
}

GEMINI_CONFIG = {
    "api_key": os.environ.get("GEMINI_API_KEY", ""),
    "model_name": "gemini-2.0-flash-preview",
    "max_try": 100
}

# 重试配置
MAX_RETRIES = 10
RETRY_DELAY = 2
TIMEOUT = 60  # 60秒超时


# ============================================================================
# Prompt构建函数
# ============================================================================
def build_prompt(instruction: str, image_descriptions: str) -> str:
    """
    构建评分prompt
    
    Args:
        instruction: 指令文本
        image_descriptions: 图像描述文本
        
    Returns:
        完整的prompt
    """
    return EVALUATION_PROMPT.format(
        instruction=instruction,
        image_descriptions=image_descriptions
    )


def build_image_descriptions(reference_images: List[str], generated_image: str) -> str:
    """
    构建图像描述 - 优化版
    
    Args:
        reference_images: 参考图像路径列表
        generated_image: 生成图像路径
        
    Returns:
        图像描述文本
    """
    descriptions = []
    
    # 明确标记参考图
    for idx, ref_img in enumerate(reference_images, 1):
        descriptions.append(f"- <IMAGE_TOKEN> [Reference Image {idx}]: Input subject to preserve.")
    
    # 明确标记生成图
    descriptions.append(f"- <IMAGE_TOKEN> [Generated Image]: The final output to evaluate.")
    
    return "\n".join(descriptions)


# ============================================================================
# 评分函数
# ============================================================================
def evaluate_with_gpt(sample_data: Dict[str, Any], sample_id: str = "") -> Optional[Dict[str, Any]]:
    """
    使用GPT进行评分（支持多次重试，timeout=60s）
    
    Args:
        sample_data: 样本数据，包含 instruction, input_images, output_image 等字段
        sample_id: 样本ID（用于日志）
        
    Returns:
        评分结果字典，包含 consistency_scores, following_score, overall_reasoning
        如果失败返回None
    """
    instruction = sample_data.get('instruction', '')
    reference_images = sample_data.get('input_images', [])
    generated_image = sample_data.get('output_image', '')
    
    if not all([instruction, reference_images, generated_image]):
        print(f"[GPT] {sample_id}: Missing required data for GPT evaluation")
        return None
    
    # 准备图像列表
    all_images = reference_images + [generated_image]
    
    # 构建图像描述
    image_descriptions = build_image_descriptions(reference_images, generated_image)
    
    # 构建prompt
    prompt = build_prompt(instruction, image_descriptions)
    
    # 多次尝试调用GPT API
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
                # 验证结果格式
                if 'consistency_scores' in result and 'following_score' in result:
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
    """
    使用Gemini进行评分（GeminiAPIGenerator内部已处理重试，timeout=60s）
    
    Args:
        sample_data: 样本数据，包含 instruction, input_images, output_image 等字段
        generator: Gemini生成器实例
        sample_id: 样本ID（用于日志）
        
    Returns:
        评分结果字典，包含 consistency_scores, following_score, overall_reasoning
        如果失败返回None
    """
    instruction = sample_data.get('instruction', '')
    reference_images = sample_data.get('input_images', [])
    generated_image = sample_data.get('output_image', '')
    
    missing = []
    if not instruction:
        missing.append("instruction（或 prompt）")
    if not reference_images:
        missing.append("input_images（非空列表）")
    if not generated_image:
        missing.append("output_image（生成图路径）")
    if missing:
        print(f"[Gemini] {sample_id}: Missing required data for Gemini evaluation: 缺少 {', '.join(missing)}")
        return None
    
    # 准备图像列表（PIL Image对象）
    all_images = []
    for img_path in reference_images + [generated_image]:
        try:
            img = Image.open(img_path)
            all_images.append(img)
        except Exception as e:
            print(f"[Gemini] {sample_id}: Error loading image {img_path}: {e}")
            return None
    
    # 构建图像描述
    image_descriptions = build_image_descriptions(reference_images, generated_image)
    
    # 构建prompt
    prompt = build_prompt(instruction, image_descriptions)
    
    # 定义响应格式
    response_format = {
        "consistency_scores": "list[float]",
        "following_score": "float",
        "overall_reasoning": "str"
    }
    
    # 调用Gemini API（重试逻辑由GeminiAPIGenerator内部处理）
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
            # 验证结果格式
            if 'consistency_scores' in response and 'following_score' in response:
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
    """
    判断评分是否有效（区分失败和低分）
    
    Args:
        score_data: 评分数据
        
    Returns:
        是否有效
    """
    if score_data is None:
        return False  # 失败
    if isinstance(score_data, dict):
        # 检查是否有必需的字段
        if 'consistency_scores' in score_data and 'following_score' in score_data:
            return True
    return False
