#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial task scoring module

Provides scoring functions (GPT and Gemini) for the spatial task
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

# Try to import utils module
from utils.openai_utils import ask_gpt4o
from utils.json_utils import mllm_output_to_dict
from api_generator.text_generator.gemini_api import GeminiAPIGenerator

# ============================================================================
# PROMPT configuration
# ============================================================================
EVALUATION_PROMPT = """You are a meticulous 3D Quality Assurance Specialist and Digital Art Critic. You are known for your extremely high standards and attention to microscopic details.

**1. THE INSTRUCTION (Read this first)**
The AI model was given this specific spatial instruction:
"{instruction}"

**2. THE IMAGES**
You will see the following images:
{image_descriptions}

**3. EVALUATION TASK**
Compare the **Generated Image** against the **Reference Images** and the **Target Image** (Ground Truth).
Your response must be a JSON object:
{{
  "viewpoint_transformation_score": <score>,
  "content_consistency_score": <score>,
  "overall_reasoning": "Critical analysis explaining the deductions. (Maximum 100 words)"
}}

**4. SCORING CRITERIA (STRICT MODE)**
Start from a perfect score and deduct points for *any* flaw. Be unforgiving.

**Metric 1: Viewpoint Transformation Score (0-10)**
Did the model move the camera *exactly* according to the instruction?

*   **10 (Perfect):** The viewpoint matches the Target Image geometry **exactly**. No perceptible deviation in angle, elevation, or distance.
*   **8-9 (Acceptable):** The movement is correct, but there is a **very minor** deviation (e.g., slightly too zoomed in or a few degrees off).
*   **5-7 (Inaccurate):** The camera moved in the general correct direction, but the magnitude is clearly wrong (e.g., turned 90 degrees instead of 45, or elevation is noticeably incorrect).
*   **2-4 (Wrong Direction / Disoriented):** The model moved the camera, but in the **wrong direction** (e.g., left instead of right, up instead of down). This is a functional failure.
*   **0-1 (Lazy Copying / Failure):** The model **failed to transform** the view (image looks identical to input) OR the image is broken. **Zero tolerance for lazy copying.**

**Metric 2: Content Consistency Score (0-10)**
Does the image preserve the object's identity, texture fidelity, and structural logic?

*   **CRITICAL LOGIC:**
    1.  **Visible Regions:** Any detail visible in the Reference Images must be preserved **perfectly**. Blur, texture loss, or color shifting results in a severe penalty.
    2.  **Invisible Regions (Blind Spots):** While the model must generate unseen parts, the generated content must be **stylistically seamless** and **geometrically precise**. It must look like a high-end rendering, not a blurry guess.

*   **10 (Flawless):** Indistinguishable from the Ground Truth. Sharp textures, perfect geometry, no artifacts.
*   **8-9 (High Quality):** Preserves identity well. Minor loss of high-frequency detail (texture grain) or very slight lighting inconsistencies.
*   **5-7 (Mediocre):** Visible parts are preserved but look **softer/blurrier** than the reference. Invisible regions are different from the target image but are reasonable and plausible.
*   **2-4 (Distorted):** Significant structural warping. Straight lines become wavy. Textures are washed out. The object identity is barely maintained. The invisible regions are not reasonable and plausible.
*   **0-1 (Unusable):** Hallucination of a completely different object or severe visual noise.

**Overall Reasoning Guide:**
Be critical. Explicitly mention what defects caused the score reduction (e.g., "Texture on the armrest is blurry," "Camera angle is 15 degrees too shallow"). Do not praise the model for doing the bare minimum.
"""

# ============================================================================
# Configuration constants
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
# Prompt construction functions
# ============================================================================
def build_prompt(instruction: str, image_descriptions: str) -> str:
    return EVALUATION_PROMPT.format(
        instruction=instruction,
        image_descriptions=image_descriptions
    )


def build_image_descriptions(reference_images: List[str], target_image: str, generated_image: str) -> str:
    descriptions = []
    
    for idx, ref_img in enumerate(reference_images, 1):
        descriptions.append(f"- <IMAGE_TOKEN> Reference Image {idx}: Input image showing a specific viewpoint")
    
    descriptions.append(f"- <IMAGE_TOKEN> Target Image: Ground truth image showing the desired output viewpoint")
    descriptions.append(f"- <IMAGE_TOKEN> Generated Image: The AI model's output that should match the Target Image")
    
    return "\n".join(descriptions)


# ============================================================================
# Scoring functions
# ============================================================================
def evaluate_with_gpt(sample_data: Dict[str, Any], sample_id: str = "") -> Optional[Dict[str, Any]]:
    instruction = sample_data.get('instruction', '')
    reference_images = sample_data.get('input_images', [])
    target_image = sample_data.get('target_image', '')
    generated_image = sample_data.get('output_image', '')
    
    if not all([instruction, reference_images, target_image, generated_image]):
        print(f"[GPT] {sample_id}: Missing required data for GPT evaluation")
        return None
    
    all_images = reference_images + [target_image, generated_image]
    image_descriptions = build_image_descriptions(reference_images, target_image, generated_image)
    prompt = build_prompt(instruction, image_descriptions)
    
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
                if 'viewpoint_transformation_score' in result and 'content_consistency_score' in result:
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
    instruction = sample_data.get('instruction', '')
    reference_images = sample_data.get('input_images', [])
    target_image = sample_data.get('target_image', '')
    generated_image = sample_data.get('output_image', '')
    
    if not all([instruction, reference_images, target_image, generated_image]):
        print(f"[Gemini] {sample_id}: Missing required data for Gemini evaluation")
        return None
    
    all_images = []
    for img_path in reference_images + [target_image, generated_image]:
        try:
            img = Image.open(img_path)
            all_images.append(img)
        except Exception as e:
            print(f"[Gemini] {sample_id}: Error loading image {img_path}: {e}")
            return None
    
    image_descriptions = build_image_descriptions(reference_images, target_image, generated_image)
    prompt = build_prompt(instruction, image_descriptions)
    
    response_format = {
        "viewpoint_transformation_score": "float",
        "content_consistency_score": "float",
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
            print(f"[Gemini] {sample_id}: empty response")
            return None
        
        if isinstance(response, dict):
            if 'viewpoint_transformation_score' in response and 'content_consistency_score' in response:
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
    if score_data is None:
        return False
    if isinstance(score_data, dict):
        if 'viewpoint_transformation_score' in score_data and 'content_consistency_score' in score_data:
            return True
    return False
