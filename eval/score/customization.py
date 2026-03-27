#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customization task scoring module

Provides scoring functions (GPT and Gemini) for the customization task
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

# Try to import utils module; fall back to local implementation if not found
from utils.openai_utils import ask_gpt4o
from utils.json_utils import mllm_output_to_dict
from api_generator.text_generator.gemini_api import GeminiAPIGenerator

# ============================================================================
# PROMPT configuration (placed at top of script for easy modification)
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
# Configuration constants
# ============================================================================
# LLM configuration
GPT_CONFIG = {
    "url": os.environ.get("OPENAI_URL", "https://api.openai.com/v1/chat/completions"),
    "key": os.environ.get("OPENAI_KEY", "")
}

GEMINI_CONFIG = {
    "api_key": os.environ.get("GEMINI_API_KEY", ""),
    "model_name": "gemini-2.0-flash-preview",
    "max_try": 100
}

# Retry configuration
MAX_RETRIES = 10
RETRY_DELAY = 2
TIMEOUT = 60  # 60-second timeout


# ============================================================================
# Prompt construction functions
# ============================================================================
def build_prompt(instruction: str, image_descriptions: str) -> str:
    """
    Build a scoring prompt
    
    Args:
        instruction: instruction text
        image_descriptions: image description text
        
    Returns:
        complete prompt
    """
    return EVALUATION_PROMPT.format(
        instruction=instruction,
        image_descriptions=image_descriptions
    )


def build_image_descriptions(reference_images: List[str], generated_image: str) -> str:
    """
    Build image descriptions - optimized version
    
    Args:
        reference_images: list of reference image paths
        generated_image: generated image path
        
    Returns:
        image description text
    """
    descriptions = []
    
    # Clearly label reference images
    for idx, ref_img in enumerate(reference_images, 1):
        descriptions.append(f"- <IMAGE_TOKEN> [Reference Image {idx}]: Input subject to preserve.")
    
    # Clearly label the generated image
    descriptions.append(f"- <IMAGE_TOKEN> [Generated Image]: The final output to evaluate.")
    
    return "\n".join(descriptions)


# ============================================================================
# Scoring functions
# ============================================================================
def evaluate_with_gpt(sample_data: Dict[str, Any], sample_id: str = "") -> Optional[Dict[str, Any]]:
    """
    Score using GPT (supports multiple retries, timeout=60s)
    
    Args:
        sample_data: sample data containing instruction, input_images, output_image, etc.
        sample_id: sample ID (for logging)
        
    Returns:
        score result dict containing consistency_scores, following_score, overall_reasoning
        returns None on failure
    """
    instruction = sample_data.get('instruction', '')
    reference_images = sample_data.get('input_images', [])
    generated_image = sample_data.get('output_image', '')
    
    if not all([instruction, reference_images, generated_image]):
        print(f"[GPT] {sample_id}: Missing required data for GPT evaluation")
        return None
    
    # Prepare image list
    all_images = reference_images + [generated_image]
    
    # Build image descriptions
    image_descriptions = build_image_descriptions(reference_images, generated_image)
    
    # Build prompt
    prompt = build_prompt(instruction, image_descriptions)
    
    # Attempt to call GPT API multiple times
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1:
                print(f"[GPT] {sample_id}: Retry attempt {attempt} (of {MAX_RETRIES})...")
            
            response = ask_gpt4o(all_images, prompt, GPT_CONFIG["url"], GPT_CONFIG["key"])
            if not response:
                print(f"[GPT] {sample_id}: Attempt {attempt}/{MAX_RETRIES}: empty response")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                return None
            
            result = mllm_output_to_dict(response)
            if result and isinstance(result, dict):
                # Validate result format
                if 'consistency_scores' in result and 'following_score' in result:
                    if attempt > 1:
                        print(f"[GPT] {sample_id}: Attempt {attempt} succeeded")
                    return result
                else:
                    print(f"[GPT] {sample_id}: Attempt {attempt}/{MAX_RETRIES}: result format incorrect - {list(result.keys())}")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                        continue
                    return None
            else:
                print(f"[GPT] {sample_id}: Attempt {attempt}/{MAX_RETRIES}: parse failed - {response[:200]}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                return None
        except Exception as e:
            print(f"[GPT] {sample_id}: Attempt {attempt}/{MAX_RETRIES}: error - {e}")
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
    Score using Gemini (retry logic is handled internally by GeminiAPIGenerator, timeout=60s)
    
    Args:
        sample_data: sample data containing instruction, input_images, output_image, etc.
        generator: Gemini generator instance
        sample_id: sample ID (for logging)
        
    Returns:
        score result dict containing consistency_scores, following_score, overall_reasoning
        returns None on failure
    """
    instruction = sample_data.get('instruction', '')
    reference_images = sample_data.get('input_images', [])
    generated_image = sample_data.get('output_image', '')
    
    missing = []
    if not instruction:
        missing.append("instruction (or prompt)")
    if not reference_images:
        missing.append("input_images (non-empty list)")
    if not generated_image:
        missing.append("output_image (generated image path)")
    if missing:
        print(f"[Gemini] {sample_id}: Missing required data for Gemini evaluation: missing {\", \".join(missing)}")
        return None
    
    # Prepare image list (PIL Image objects)
    all_images = []
    for img_path in reference_images + [generated_image]:
        try:
            img = Image.open(img_path)
            all_images.append(img)
        except Exception as e:
            print(f"[Gemini] {sample_id}: Error loading image {img_path}: {e}")
            return None
    
    # Build image descriptions
    image_descriptions = build_image_descriptions(reference_images, generated_image)
    
    # Build prompt
    prompt = build_prompt(instruction, image_descriptions)
    
    # Define response format
    response_format = {
        "consistency_scores": "list[float]",
        "following_score": "float",
        "overall_reasoning": "str"
    }
    
    # Call Gemini API (retry logic handled internally by GeminiAPIGenerator)
    try:
        response = generator.gen_response(
            prompt=prompt,
            response_format=response_format,
            images=all_images,
            think=False
        )
        
        if response is None:
            print(f"[Gemini] {sample_id}: empty response")
            return None
        
        if isinstance(response, dict):
            # Validate result format
            if 'consistency_scores' in response and 'following_score' in response:
                return response
            else:
                print(f"[Gemini] {sample_id}: result format incorrect - {list(response.keys())}")
                return None
        else:
            print(f"[Gemini] {sample_id}: unexpected response type - {type(response)}")
            return None
    except Exception as e:
        print(f"[Gemini] {sample_id}: error - {e}")
        traceback.print_exc()
        return None


def is_score_valid(score_data: Any) -> bool:
    """
    Determine whether a score is valid (distinguishing failure from low score)
    
    Args:
        score_data: score data
        
    Returns:
        whether valid
    """
    if score_data is None:
        return False  # failed
    if isinstance(score_data, dict):
        # Check whether required fields are present
        if 'consistency_scores' in score_data and 'following_score' in score_data:
            return True
    return False
