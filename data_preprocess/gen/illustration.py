import os
#!/usr/bin/env python3
"""
Illustration data generation script

Features:
1. Read train data from split/illustration directory (train data only)
2. Rewrite text using gemini-3-pro-preview
3. Save to final/illustration/{train}/{image_count_category}/data and json directories
4. Support unique IDs to avoid duplicate generation
"""

import json
import random
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image
from dataclasses import dataclass

# Add utils path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from utils.common import (
    get_image_count_category,
    load_generated_ids,
    save_sample_data,
    generate_unique_id
)

# Ensure project root is in sys.path
MACRO_DIR = CURRENT_DIR.parent.parent
if str(MACRO_DIR) not in sys.path:
    sys.path.insert(0, str(MACRO_DIR))

from api_generator.text_generator.gemini_api import GeminiAPIGenerator

# ====== Configuration parameters ======
# Directory of raw image-text data (contains conversation info, used to build context)
DATA_DIR = MACRO_DIR / "data"  # Source data directory
SPLIT_DIR = DATA_DIR / "split" / "illustration"
FINAL_DIR = DATA_DIR / "final" / "illustration"
# Gemini API configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = "gemini-3-pro-preview"

# Generation config: {image_count_category: {train: count, eval: count}}
GEN_CONFIG = {
    "1-3": {"train": 30000, "eval": 500},
    "4-5": {"train": 30000, "eval": 500},
    "6-7": {"train": 30000, "eval": 500}, # 200
    ">=8": {"train": 47500, "eval": 500}, # 10000
}

# Thread configuration
MAX_WORKERS = 64
MAX_TRIES = 3

# Random seed configuration
RANDOM_SEED = 42

# Logging configuration
LOG_TO_SHELL = False  # Whether to output logs to shell

# Placeholder token
PLACEHOLDER_TOKEN = "<IMAGE_TOKEN>"
REWRITE_TOKEN = "<IMAGE TOKEN>"

# Prompt template
PROMPT_TEMPLATE = """
You are an expert evaluator of multimodal sequences. You will review a sequence of interleaved text and images that naturally leads to a final target image.

Context:
- Total images (including the target): {image_count}
- Context images (before the target): {context_image_count}
- The final image (image {image_count}) is the target outcome. Rely only on the provided content.

Sequence content:
```
{content}
```

Your Goal:
Evaluate this sequence as a training sample for a "descriptive image generation" model. In this task, images serve the semantic narrative. The model learns to predict the target image as a natural continuation of the preceding text and visual context, rather than just following explicit generation instructions.

Evaluation Tasks:

1. **Analyze Context Image Contributions:**
   For every context image (images 1..{context_image_count}), decide if it contributes meaningfully to the semantic or visual context required to understand or generate the target image.
   - **CRITICAL:** Do not limit "contribution" to direct visual overlapping (e.g., pixel-level similarity).
   - **INCLUDE:** Images that provide semantic grounding, style references, character designs (e.g., a movie poster providing costume details for a movie scene), or narrative setup. If an image helps establish *who, what, where, or the style* of the target, it contributes.
   - Return a boolean list `image_contributions` of length {context_image_count} (True = contributes, False = irrelevant/noise).

2. **Text Quality & Rewrite:**
   Assess the text. It should be rich in semantic information, coherent, and naturally lead into the target image.
   - If the text is already high-quality, informative, and flows naturally, set `rewritten_text` to null.
   - Otherwise, rewrite the text to improve clarity and information density without losing critical narrative details.
   - **Rewrite Rules:**
     - **Do NOT over-simplify.** The text must remain descriptive and retain the original semantic richness. Text is the primary carrier of meaning here.
     - Ensure the text flows naturally so that placing the target image at the very end feels like a logical conclusion or illustration of the text.
     - Use the placeholder `{rewrite_token}` exactly once for each context image marked `True` in task 1, in their original chronological order.
     - Do NOT include `{rewrite_token}` for images marked `False`.
     - Append a final `{rewrite_token}` at the very end of the text to represent the target image position.

3. **Score: Semantic Guidance (1-10):**
   Score how effectively the prior information (text + contributing context images) provides the necessary *semantic* blueprint for the target image.
   - 10 = The context provides rich character, setting, or stylistic information that makes the content of the target image clear and unambiguous.
   - 1 = The target image feels random or completely unrelated to the context.

4. **Score: Training Suitability (1-10):**
   Score how suitable this sample is for training a multimodal autoregressive model.
   - Consider: Is the transition from the final text to the target image natural? Is the target image a logical "next token" in this visual-textual sequence?
   - 10 = Perfectly aligned; predicting the target image after the text is a natural and easy task for a model.
   - 1 = Disjointed; the target image is a non-sequitur or contradicts the text.

Return a single valid JSON object (no Markdown) with keys:
{{
  "image_contributions": list[bool],
  "rewritten_text": null or string,
  "guidance_score": int (1-10),
  "training_score": int (1-10),
  "reasoning": string  // Explain why images were kept/dropped and why the text was/wasn't rewritten.
}}
"""

RESPONSE_FORMAT = {
    "image_contributions": "list[bool]",
    "rewritten_text": "null_or_str",
    "guidance_score": "int",
    "training_score": "int",
    "reasoning": "str",
}
# ======================

# ====== Thread-local storage ======
thread_local = threading.local()

# ====== Unique ID check lock ======
unique_id_lock = threading.Lock()
# =============================


@dataclass
class SequenceEntry:
    """Sequence entry"""
    entry_type: str  # "text" or "image"
    content: Optional[str] = None
    image_idx: Optional[int] = None
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict"""
        result = {"type": self.entry_type}
        if self.entry_type == "text":
            result["content"] = self.content
        else:
            result["image_idx"] = self.image_idx
            result["image_path"] = self.image_path
        return result


@dataclass
class SampleContext:
    """Sample context"""
    text_content: str
    image_paths: List[str]
    sequence: List[SequenceEntry]
    missing_images: List[str]

    @property
    def image_count(self) -> int:
        """Image count"""
        return len(self.image_paths)

    @property
    def context_image_count(self) -> int:
        """Context image count"""
        return len(self.image_paths) - 1 if len(self.image_paths) > 0 else 0


def get_or_create_generator(
    gemini_api_key: str,
    gemini_model_name: str,
    max_try: int
) -> GeminiAPIGenerator:
    """
    Get or create thread-local generator
    
    Args:
        gemini_api_key: Gemini API key
        gemini_model_name: Gemini model name
        max_try: max number of retries
    
    Returns:
        text generator
    """
    if not hasattr(thread_local, 'generator'):
        thread_local.generator = GeminiAPIGenerator(
            app_key=gemini_api_key,
            model_name=gemini_model_name,
            max_try=max_try,
            print_log=LOG_TO_SHELL
        )
    return thread_local.generator


def load_split_data(split_dir: Path, split_type: str) -> List[Dict]:
    """
    Load data from split directory
    
    Args:
        split_dir: split directory
        split_type: "train" or "eval"
    
    Returns:
        list of samples
    """
    all_samples = []
    
    # Try to load a single JSON file
    json_file = split_dir / f"{split_type}.json"
    if json_file.exists():
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"Warning: failed to load file {json_file}: {e}")
    
    # Try to load multiple JSON files (if data is split)
    json_files = sorted(split_dir.glob(f"{split_type}_*.json"))
    if json_files:
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_samples.extend(data)
            except Exception as e:
                print(f"Warning: failed to load file {json_file}: {e}")
                continue
        if all_samples:
            return all_samples
    
    # Compatibility with old format: try to load jsonl files
    jsonl_file = split_dir / f"{split_type}.jsonl"
    if jsonl_file.exists():
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        all_samples.append(sample)
                    except json.JSONDecodeError:
                        continue
            if all_samples:
                return all_samples
        except Exception as e:
            print(f"Warning: failed to load file {jsonl_file}: {e}")
    
    jsonl_files = sorted(split_dir.glob(f"{split_type}_*.jsonl"))
    if jsonl_files:
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            sample = json.loads(line)
                            all_samples.append(sample)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Warning: failed to load file {jsonl_file}: {e}")
                continue
        if all_samples:
            return all_samples
    
    # Compatibility with old format: try to load prefixed JSON files
    for json_file in split_dir.glob(f"{split_type}_*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_samples.extend(data)
        except Exception as e:
            print(f"Warning: failed to load file {json_file}: {e}")
            continue
    
    return all_samples


def load_original_sample(data_dir: Path, source_file: str, source_line: int) -> Optional[Dict[str, Any]]:
    """
    Load a sample from the original data file
    
    Args:
        data_dir: data directory
        source_file: source file path
        source_line: source file line number
    
    Returns:
        sample data or None
    """
    file_path = data_dir / source_file if not Path(source_file).is_absolute() else Path(source_file)
    if LOG_TO_SHELL:
        print(f"    [load original sample] Attempting to load: data_dir={data_dir}, source_file={source_file}, source_line={source_line}")
        print(f"    [load original sample] Full path: {file_path}")
    
    if not file_path.exists():
        if LOG_TO_SHELL:
            print(f"    [load original sample] File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                if line_num == source_line:
                    line = line.strip()
                    if not line:
                        if LOG_TO_SHELL:
                            print(f"    [load original sample] Line {source_line} is empty")
                        return None
                    sample = json.loads(line)
                    if LOG_TO_SHELL:
                        print(f"    [load original sample] Successfully loaded line {source_line}")
                        phrases = sample.get('phrases', [])
                        print(f"    [load original sample] Sample contains {len(phrases)} phrases")
                    return sample
        if LOG_TO_SHELL:
            print(f"    [load original sample] File has too few lines, cannot find line {source_line}")
    except Exception as e:
        if LOG_TO_SHELL:
            print(f"    [load original sample] Load failed: {e}")
        return None
    return None


def extract_phrases(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract phrases list (see gen.py implementation for reference)
    
    Args:
        sample: sample data
    
    Returns:
        list of phrases
    """
    phrases: List[Dict[str, Any]] = []
    
    # First try to get directly from the phrases field
    direct_phrases = sample.get('phrases', [])
    if isinstance(direct_phrases, list):
        phrases.extend(direct_phrases)
        if LOG_TO_SHELL:
            print(f"    [extract phrases] Extracted from phrases field: {len(phrases)}")
    
    # Extract phrases from conversation (see gen.py implementation)
    conversations = sample.get("conversation", [])
    if isinstance(conversations, dict):
        conversations = conversations.get("phrases", [])
    if LOG_TO_SHELL:
        print(f"    [extract phrases] conversations type: {type(conversations)}, length: {len(conversations) if isinstance(conversations, list) else 'N/A'}")
    
    for convo in conversations:
        if not isinstance(convo, dict):
            continue
        convo_phrases = convo.get("phrases", [])
        if isinstance(convo_phrases, list):
            phrases.extend(convo_phrases)
            if LOG_TO_SHELL:
                print(f"    [extract phrases] Extracted from conversation: +{len(convo_phrases)}")
    
    if LOG_TO_SHELL:
        print(f"    [extract phrases] Total extracted: {len(phrases)}")
    
    return phrases


def resolve_image_path(
    image_info: Dict[str, Any],
    image_root: Optional[Path],
    data_dir: Optional[Path],
) -> Optional[Path]:
    """
    Resolve image path (see gen.py implementation for reference)
    
    Args:
        image_info: image information
        image_root: image root directory
        data_dir: data directory
    
    Returns:
        resolved image path or None
    """
    candidate_paths: List[Path] = []
    
    # Try to get path from image field (see gen.py implementation)
    raw_path = image_info.get("image")
    if isinstance(raw_path, str) and raw_path.strip():
        path = Path(raw_path)
        if path.is_absolute():
            candidate_paths.append(path)
        else:
            if image_root:
                candidate_paths.append(image_root / raw_path)
            if data_dir:
                candidate_paths.append(data_dir / raw_path)
            candidate_paths.append(path)
    
    # Try to get path from img_path field (see gen.py implementation)
    relative_path = image_info.get("img_path")
    if isinstance(relative_path, str) and relative_path.strip():
        if image_root:
            candidate_paths.append(image_root / relative_path)
        if data_dir:
            candidate_paths.append(data_dir / relative_path)
        candidate_paths.append(Path(relative_path))
    
    # Compatibility with old format: try path or filepath fields
    path_str = image_info.get('path', '') or image_info.get('filepath', '')
    if path_str:
        path = Path(path_str)
        if path.is_absolute():
            candidate_paths.append(path)
        else:
            if image_root:
                candidate_paths.append(image_root / path_str)
            if data_dir:
                candidate_paths.append(data_dir / path_str)
            candidate_paths.append(path)
    
    if LOG_TO_SHELL:
        print(f"    [resolve path] Image info: {image_info}")
        print(f"    [resolve path] Number of candidate paths: {len(candidate_paths)}")
    
    # Try each candidate path
    for candidate in candidate_paths:
        if candidate.exists():
            if LOG_TO_SHELL:
                print(f"    [resolve path] Found valid path: {candidate}")
            return candidate
        elif LOG_TO_SHELL:
            print(f"    [resolve path] Path does not exist: {candidate}")
    
    # If all paths are missing, return the first candidate (if any) or None
    if candidate_paths:
        if LOG_TO_SHELL:
            print(f"    [resolve path] All paths missing, returning first candidate: {candidate_paths[0]}")
        return candidate_paths[0]
    
    if LOG_TO_SHELL:
        print(f"    [resolve path] Cannot resolve path, returning None")
    return None


def get_image_count_from_sample(sample: Dict[str, Any]) -> Optional[int]:
    """
    Get image count from split data sample
    
    Args:
        sample: split data sample
    
    Returns:
        image count or None (total image count including target image)
    """
    if LOG_TO_SHELL:
        print(f"    [get image count] Starting to get image count")
        print(f"    [get image count] sample image_count: {sample.get('image_count')}")
        print(f"    [get image count] sample actual_image_count: {sample.get('actual_image_count')}")
    
    # Get image_count directly from split data (total count including target image)
    result = sample.get('image_count')
    if LOG_TO_SHELL:
        print(f"    [get image count] image_count from split data: {result}")
    return result


def build_sample_context(
    sample: Dict[str, Any],
    original_sample: Dict[str, Any],
    image_count: int,
    data_dir: Path,
    image_root: Optional[Path],
    true_index: Optional[int] = None,
) -> SampleContext:
    """
    Build sample context
    
    Args:
        sample: sample data
        original_sample: original sample data
        image_count: actual number of images to use (including target image)
        data_dir: data directory
        image_root: image root directory
        true_index: true index (0-based)
    
    Returns:
        sample context
    """
    phrases = extract_phrases(original_sample)
    sequence = []
    text_parts = []
    image_paths = []
    missing_images = []
    
    if LOG_TO_SHELL:
        print(f"[build context] Starting to build sample context")
        print(f"  - Expected image count: {image_count}")
        print(f"  - true_index: {true_index}")
        print(f"  - Total phrases: {len(phrases)}")
        print(f"  - data_dir: {data_dir}")
        print(f"  - image_root: {image_root}")
    
    # See gen.py implementation, process phrases in order
    context_image_paths: List[str] = []  # Context images (excluding target image)
    target_image_path: Optional[str] = None
    
    # If true_index is provided, process only images before true_index (not including it, as it is the target)
    context_image_count = image_count - 1  # Number of context images (excluding target image)
    if true_index is not None:
        context_image_count = true_index
    
    if LOG_TO_SHELL:
        print(f"  - Context image count: {context_image_count}")
    
    # Collect context images and target image (see gen.py, process all phrases by image_idx)
    image_idx = 0
    for phrase_idx, phrase in enumerate(phrases):
        if LOG_TO_SHELL:
            print(f"  - Processing phrase[{phrase_idx}]: phrase structure={phrase}")
        
        # Process text (see gen.py implementation)
        text_info = phrase.get("text")
        if text_info:
            content = text_info.get("content") if isinstance(text_info, dict) else None
            if isinstance(content, str):
                normalized = " ".join(content.replace("\n", " ").split())
                if normalized:
                    text_parts.append(normalized)
                    sequence.append(SequenceEntry(entry_type='text', content=normalized))
                    if LOG_TO_SHELL:
                        print(f"    -> Added text: {normalized[:50]}...")
        
        # Process image (see gen.py implementation)
        image_info = phrase.get("image")
        if image_info:
            if LOG_TO_SHELL:
                print(f"    -> Image info: {image_info}, current image index: {image_idx}")
            
            resolved = resolve_image_path(image_info, image_root, data_dir)
            if resolved is None:
                missing_images.append(str(image_info))
                image_idx += 1
                if LOG_TO_SHELL:
                    print(f"    -> Image resolution failed: {image_info}")
                continue
            resolved = resolved.resolve()
            if not resolved.exists():
                missing_images.append(str(resolved))
                image_idx += 1
                if LOG_TO_SHELL:
                    print(f"    -> Image file not found: {resolved}")
                continue
            
            # Determine whether context or target image based on image index (see gen.py)
            if image_idx < context_image_count:
                # Context image
                context_image_paths.append(str(resolved))
                text_parts.append(PLACEHOLDER_TOKEN)
                sequence.append(SequenceEntry(
                    entry_type='image',
                    image_idx=len(context_image_paths) - 1,
                    image_path=str(resolved)
                ))
                if LOG_TO_SHELL:
                    print(f"    -> Added context image: {resolved}, image_idx={image_idx}")
            elif (true_index is not None and image_idx == true_index) or (true_index is None and image_idx == image_count - 1):
                # Target image (temporarily saved, added after the loop)
                target_image_path = str(resolved)
                if LOG_TO_SHELL:
                    print(f"    -> Found target image: {resolved}, image_idx={image_idx}")
            
            image_idx += 1
    
    # Build final image_paths: context images + target image (for LLM) (see gen.py)
    image_paths = context_image_paths.copy()
    if target_image_path:
        image_paths.append(target_image_path)
        text_parts.append(PLACEHOLDER_TOKEN)
        sequence.append(SequenceEntry(
            entry_type='image',
            image_idx=len(image_paths) - 1,
            image_path=target_image_path
        ))
        if LOG_TO_SHELL:
            print(f"    -> Added target image to final list: {target_image_path}")
    
    # Verify PLACEHOLDER_TOKEN count (see gen.py implementation)
    placeholder_token_count = len([item for item in text_parts if item == PLACEHOLDER_TOKEN])
    if LOG_TO_SHELL:
        print(f"    -> PLACEHOLDER_TOKEN count: {placeholder_token_count}, expected: {image_count}")
    if placeholder_token_count != image_count:
        if LOG_TO_SHELL:
            print(f"    [error] PLACEHOLDER_TOKEN count incorrect: {placeholder_token_count} != {image_count}")
    
    text_content = ' '.join(text_parts)
    
    if LOG_TO_SHELL:
        print(f"[build context] Done")
        print(f"  - Actual images found: {len(image_paths)}")
        print(f"  - Missing images: {len(missing_images)}")
        print(f"  - Text segments: {len(text_parts)}")
        print(f"  - Sequence entries: {len(sequence)}")
    
    return SampleContext(
        text_content=text_content,
        image_paths=image_paths,
        sequence=sequence,
        missing_images=missing_images
    )


def ensure_sequence_valid(context: SampleContext, expected_count: int) -> Tuple[bool, Optional[str]]:
    """
    Validate that the sequence is valid
    
    Args:
        context: sample context
        expected_count: expected image count
    
    Returns:
        (is_valid, error_message)
    """
    if LOG_TO_SHELL:
        print(f"    [validate sequence] Starting validation")
        print(f"      - Expected image count: {expected_count}")
        print(f"      - Actual image count: {context.image_count}")
        print(f"      - Image path list: {context.image_paths}")
        print(f"      - Missing images: {context.missing_images}")
        print(f"      - Sequence entry count: {len(context.sequence)}")
    
    if context.image_count != expected_count:
        error_msg = f"Image count mismatch: expected {expected_count}, actual {context.image_count}"
        if LOG_TO_SHELL:
            print(f"    [validate sequence] Validation failed: {error_msg}")
            print(f"      - Detailed analysis:")
            print(f"        * Image path list length: {len(context.image_paths)}")
            print(f"        * Existence of each image path: {[(p, Path(p).exists()) for p in context.image_paths]}")
        return False, error_msg
    
    if context.missing_images:
        error_msg = f"Missing images: {len(context.missing_images)}"
        if LOG_TO_SHELL:
            print(f"    [validate sequence] Validation failed: {error_msg}")
            print(f"      - Missing image details: {context.missing_images}")
        return False, error_msg
    
    for img_path in context.image_paths:
        if not Path(img_path).exists():
            error_msg = f"Image file not found: {img_path}"
            if LOG_TO_SHELL:
                print(f"    [validate sequence] Validation failed: {error_msg}")
            return False, error_msg
    
    if LOG_TO_SHELL:
        print(f"    [validate sequence] Validation passed")
    return True, None


def build_prompt(sample: Dict[str, Any], context: SampleContext) -> str:
    """
    Build prompt
    
    Args:
        sample: sample data
        context: sample context
    
    Returns:
        prompt string
    """
    content_parts = []
    for entry in context.sequence:
        if entry.entry_type == 'text':
            content_parts.append(entry.content)
        else:
            content_parts.append(PLACEHOLDER_TOKEN)
    
    content = ' '.join(content_parts)
    
    prompt = PROMPT_TEMPLATE.format(
        image_count=context.image_count,
        context_image_count=context.context_image_count,
        content=content,
        rewrite_token=REWRITE_TOKEN
    )
    
    return prompt


def normalize_bool_list(values: List[Any], expected_len: int) -> List[bool]:
    """
    Normalize boolean list
    
    Args:
        values: list of values
        expected_len: expected length
    
    Returns:
        normalized boolean list
    """
    if not isinstance(values, list):
        return [True] * expected_len
    
    result = []
    for v in values[:expected_len]:
        if isinstance(v, bool):
            result.append(v)
        elif isinstance(v, (int, float)):
            result.append(bool(v))
        else:
            result.append(bool(str(v).lower() in ['true', '1', 'yes']))
    
    while len(result) < expected_len:
        result.append(True)
    
    return result[:expected_len]


def normalize_int(value: Any, min_value: int = 1, max_value: int = 10) -> int:
    """
    Normalize integer
    
    Args:
        value: value
        min_value: minimum value
        max_value: maximum value
    
    Returns:
        normalized integer
    """
    try:
        if isinstance(value, (int, float)):
            result = int(value)
        else:
            result = int(str(value))
        return max(min_value, min(max_value, result))
    except (ValueError, TypeError):
        return (min_value + max_value) // 2


def validate_rewritten_text(rewritten: Optional[str], required_token_count: int) -> Optional[str]:
    """Validate the number of REWRITE_TOKEN occurrences in rewritten text
    
    Args:
        rewritten: rewritten text
        required_token_count: number of contributing context images (excluding target image)
    
    Returns:
        validated text; if token count is contributing_count+1 and the last token is at the end, strip it;
        returns None if token count does not meet requirements
    """
    if required_token_count <= 0:
        if rewritten is None:
            return None
        stripped_zero = rewritten.strip()
        if not stripped_zero:
            return None
        token_count_zero = stripped_zero.count(REWRITE_TOKEN)
        if token_count_zero != 0:
            return None  # No tokens should appear when there are no contributing images
        return stripped_zero

    if rewritten is None:
        return None
    if not isinstance(rewritten, str):
        return None
    stripped = rewritten.strip()
    if not stripped:
        return None
    
    token_count = stripped.count(REWRITE_TOKEN)
    
    # Check whether token count meets requirements
    if token_count == required_token_count:
        # Token count exactly equals contributing image count, return normally
        return stripped
    elif token_count == required_token_count + 1:
        # Token count is contributing count + 1 (includes target image), check if last token is at the end
        # Use rsplit to split once from the right, separator is REWRITE_TOKEN
        parts = stripped.rsplit(REWRITE_TOKEN, 1)
        if len(parts) == 2:
            # parts[0] is content before the last token, parts[1] is content after the last token
            before_last_token = parts[0].rstrip()
            after_last_token = parts[1].strip()
            
            # Check if last token is at the end (after_last_token should be empty or only punctuation)
            # If there is substantial content after the last token, validation fails
            if after_last_token and not all(c in '.,;:!?\'"' for c in after_last_token):
                # Last token is not at the end, validation fails
                return None
            
            # Remove the last token
            return before_last_token.strip()
        # If split fails (should not happen in theory), return None
        return None
    else:
        # Token count does not meet requirements, return None
        return None


def process_sample(
    sample: Dict[str, Any],
    data_dir: Path,
    image_root: Optional[Path],
    text_generator: GeminiAPIGenerator,
    final_dir: Path,
    split_type: str,
    image_count_category: str,
    idx: int,
    generated_ids: Set[str]
) -> bool:
    """
    Process a single sample
    
    Args:
        sample: split data sample
        data_dir: directory of raw image-text data (contains conversation info, used for context)
        image_root: image root directory
        text_generator: text generator
        final_dir: final directory
        split_type: "train" or "eval"
        image_count_category: image count category
        idx: sample index
        generated_ids: set of already-generated unique IDs
    
    Returns:
        whether processing succeeded
    """
    source_file = sample.get('source_file')
    source_line = sample.get('source_line')
    true_index = sample.get('true_index')
    
    if not source_file or source_line is None or true_index is None:
        return False
    
    # Generate unique ID
    # If SAVE_ORIGINAL_STRING=True, get the original string for saving
    from utils.common import SAVE_ORIGINAL_STRING
    unique_id_result = generate_unique_id("illustration",
                                         return_original=SAVE_ORIGINAL_STRING,
                                         source_file=source_file,
                                         source_line=source_line,
                                         true_index=true_index)
    
    # Extract unique ID for checking (use MD5 hash if tuple, or use string directly)
    unique_id = unique_id_result[0] if isinstance(unique_id_result, tuple) else unique_id_result
    
    # Check whether unique_id has already been generated (thread-safe)
    with unique_id_lock:
        if unique_id in generated_ids:
            return False
        # Add unique_id immediately to prevent other threads from generating the same
        generated_ids.add(unique_id)
    
    # Load original sample (from raw image-text data directory, contains conversation info)
    if LOG_TO_SHELL:
        print(f"[Sample {idx}] Starting processing")
        print(f"  - source_file: {source_file}")
        print(f"  - source_line: {source_line}")
        print(f"  - true_index: {true_index}")
        print(f"  - data_dir: {data_dir} (used for loading raw image-text data)")
    
    original_sample = load_original_sample(data_dir, source_file, source_line)
    if not original_sample:
        if LOG_TO_SHELL:
            print(f"[Sample {idx}] Failed to load original sample: data_dir={data_dir}, source_file={source_file}, source_line={source_line}")
        # Remove unique_id (if it was added)
        with unique_id_lock:
            generated_ids.discard(unique_id)
        return False
    
    if LOG_TO_SHELL:
        print(f"[Sample {idx}] Successfully loaded original sample")
        phrases = extract_phrases(original_sample)
        print(f"  - Original sample phrase count: {len(phrases)}")
        for i, phrase in enumerate(phrases[:10]):  # Only print first 10
            phrase_type = phrase.get('type', 'unknown')
            if not phrase_type:
                # Try to determine type from text or image field
                if phrase.get('text'):
                    phrase_type = 'text'
                elif phrase.get('image'):
                    phrase_type = 'image'
            print(f"    phrases[{i}]: type={phrase_type}")
    
    # Get image count (from split data)
    image_count = get_image_count_from_sample(sample)
    if not image_count:
        if LOG_TO_SHELL:
            print(f"[Sample {idx}] Cannot get image count")
            print(f"  - sample image_count: {sample.get('image_count')}")
            print(f"  - sample actual_image_count: {sample.get('actual_image_count')}")
        # Remove unique_id (if it was added)
        with unique_id_lock:
            generated_ids.discard(unique_id)
        return False
    
    if LOG_TO_SHELL:
        print(f"[Sample {idx}] Image count: {image_count}")
    
    # Build sample context
    context = build_sample_context(
        sample, original_sample, image_count, data_dir, image_root, true_index
    )
    
    # Validate sequence
    if LOG_TO_SHELL:
        print(f"[Sample {idx}] Starting sequence validation")
        print(f"  - unique_id: {unique_id}")
        print(f"  - source_file: {source_file}")
        print(f"  - source_line: {source_line}")
        print(f"  - true_index: {true_index}")
        print(f"  - Expected image count: {image_count}")
        print(f"  - Actual image count: {context.image_count}")
        print(f"  - Actual image paths: {context.image_paths}")
        print(f"  - Missing images: {context.missing_images}")
        print(f"  - Sequence entries: {len(context.sequence)}")
    
    is_valid, error_msg = ensure_sequence_valid(context, image_count)
    if not is_valid:
        if LOG_TO_SHELL:
            print(f"[Sample {idx}] Validation failed: {error_msg}")
            print(f"  - Details:")
            print(f"    * Expected image count: {image_count}")
            print(f"    * Actual image count: {context.image_count}")
            print(f"    * Image path list: {context.image_paths}")
            print(f"    * Missing images: {context.missing_images}")
            print(f"    * Sequence entries: {[e.entry_type for e in context.sequence]}")
        print(f"Sample {idx} validation failed: {error_msg}")
        # Remove unique_id (if it was added)
        with unique_id_lock:
            generated_ids.discard(unique_id)
        return False
    
    if LOG_TO_SHELL:
        print(f"[Sample {idx}] Validation passed")
    
    # Build prompt
    if LOG_TO_SHELL:
        print(f"[Sample {idx}] Building prompt")
    prompt = build_prompt(sample, context)
    
    # Load images
    if LOG_TO_SHELL:
        print(f"[Sample {idx}] Loading {len(context.image_paths)} images")
    images = []
    for i, img_path in enumerate(context.image_paths):
        if LOG_TO_SHELL:
            print(f"  - Loading image [{i+1}/{len(context.image_paths)}]: {img_path}")
        try:
            img_path_obj = Path(img_path)
            if not img_path_obj.exists():
                if LOG_TO_SHELL:
                    print(f"    [load image] Path not found: {img_path}")
                raise FileNotFoundError(f"Image file not found: {img_path}")
            img = Image.open(img_path)
            images.append(img)
            if LOG_TO_SHELL:
                print(f"    [load image] Successfully loaded: {img_path}, size={img.size}")
        except Exception as e:
            if LOG_TO_SHELL:
                print(f"    [load image] Load failed: {img_path}, error={e}")
            print(f"Sample {idx} failed to load image {img_path}: {e}")
            # Remove unique_id (if it was added)
            with unique_id_lock:
                generated_ids.discard(unique_id)
            return False
    
    if LOG_TO_SHELL:
        print(f"[Sample {idx}] Successfully loaded all {len(images)} images")
    
    # Call LLM to generate result
    try:
        response = text_generator.gen_response(
            prompt=prompt,
            images=images,
            response_format=RESPONSE_FORMAT
        )
        
        if not response:
            return False
        
        image_contributions = normalize_bool_list(
            response.get('image_contributions', []),
            context.context_image_count
        )
        # Count contributing context images (excluding target image)
        contributing_count = sum(1 for flag in image_contributions if flag)
        rewritten_text = validate_rewritten_text(
            response.get('rewritten_text'),
            contributing_count  # Only contributing context images, excluding target image
        )
        guidance_score = normalize_int(response.get('guidance_score', 5))
        training_score = normalize_int(response.get('training_score', 5))
        reasoning = response.get('reasoning', '')
        
    except Exception as e:
        print(f"Sample {idx} LLM call failed: {e}")
        # Remove unique_id (if it was added)
        with unique_id_lock:
            generated_ids.discard(unique_id)
        return False
    
    # Validate the rewritten text (must contain the correct number of <IMAGE TOKEN>)
    if rewritten_text is None:
        if LOG_TO_SHELL:
            print(f"[Sample {idx}] Rewritten text validation failed: token count mismatch or format error")
        # Remove unique_id (if it was added)
        with unique_id_lock:
            generated_ids.discard(unique_id)
        return False
    
    # Build final text (must use rewritten text; original text is no longer a fallback)
    instruction = rewritten_text
    
    # Convert REWRITE_TOKEN (<IMAGE TOKEN>) to <image n> format
    # Replace in order: first token -> <image 1>, second -> <image 2>, etc.
    # Only convert when rewritten_text exists (consistent with reference file)
    if rewritten_text and REWRITE_TOKEN in instruction:
        token_index = 1
        while REWRITE_TOKEN in instruction:
            instruction = instruction.replace(REWRITE_TOKEN, f"<image {token_index}>", 1)
            token_index += 1
    
    final_text = instruction
    
    # Save data
    json_data = {
        'source_file': source_file,
        'source_line': source_line,
        'true_index': true_index,
        'image_count': image_count,
        'text': final_text,
        'image_contributions': image_contributions,
        'guidance_score': guidance_score,
        'training_score': training_score,
        'reasoning': reasoning,
        'input_images': context.image_paths[:-1] if len(context.image_paths) > 1 else [],
        'output_image': context.image_paths[-1] if context.image_paths else None
    }
    
    image_files = {}
    for i, img_path in enumerate(context.image_paths):
        image_files[f"image_{i+1}.jpg"] = Path(img_path)
    
    success = save_sample_data(
        final_dir,
        split_type,
        image_count_category,
        idx,
        unique_id_result,  # May be a string or (md5_hash, original_string) tuple
        json_data,
        image_files
    )
    
    if not success:
        # Remove unique_id (if saving failed)
        with unique_id_lock:
            generated_ids.discard(unique_id)
    
    return success


def worker_task(
    idx: int,
    sample: Dict[str, Any],
    data_dir: Path,
    image_root: Optional[Path],
    final_dir: Path,
    split_type: str,
    image_count_category: str,
    stop_event: threading.Event,
    gemini_api_key: str,
    gemini_model_name: str,
    max_try: int,
    generated_ids: Set[str]
) -> bool:
    """
    Worker thread task
    
    Args:
        idx: sample index
        sample: split data sample
        data_dir: directory of raw image-text data (contains conversation info, used for context)
        image_root: image root directory
        final_dir: final directory
        split_type: "train" or "eval"
        image_count_category: image count category
        stop_event: stop event
        gemini_api_key: Gemini API key
        gemini_model_name: Gemini model name
        max_try: max number of retries
        generated_ids: set of already-generated unique IDs
    
    Returns:
        whether processing succeeded
    """
    if stop_event.is_set():
        return False
    
    # Get or create thread-local generator
    text_generator = get_or_create_generator(
        gemini_api_key, gemini_model_name, max_try
    )
    
    # Process sample
    result = process_sample(
        sample,
        data_dir,
        image_root,
        text_generator,
        final_dir,
        split_type,
        image_count_category,
        idx,
        generated_ids
    )
    return result


def process_split_data(
    split_dir: Path,
    final_dir: Path,
    split_type: str,
    image_count_category: str,
    target_count: int,
    generated_ids: Set[str]
) -> None:
    """
    Process split data and generate final data
    
    Args:
        split_dir: split directory
        final_dir: final directory
        split_type: "train" or "eval"
        image_count_category: image count category
        target_count: target generation count
        generated_ids: set of already-generated unique IDs
    """
    # Load split data
    samples = load_split_data(split_dir, split_type)
    
    # Filter samples matching image_count_category
    filtered_samples = []
    for sample in samples:
        actual_image_count = sample.get('actual_image_count')
        if actual_image_count is not None:
            category = get_image_count_category(actual_image_count)
            if category == image_count_category:
                filtered_samples.append(sample)
    
    print(f"Found {len(filtered_samples)} samples matching {image_count_category}")
    
    # Randomly shuffle samples for random selection
    random.seed(RANDOM_SEED)
    random.shuffle(filtered_samples)
    
    # Data directory and image root directory (using configured paths)
    data_dir = DATA_DIR  # Directory of raw image-text data (contains conversation info, for building context)
    image_root = None  # Image root directory (optional, set as needed)
    
    if LOG_TO_SHELL:
        print(f"  - data_dir: {data_dir} (used for loading raw image-text data)")
        print(f"  - image_root: {image_root}")
    
    # Create stop event
    stop_event = threading.Event()
    
    # Use thread pool for processing
    current_idx = len(generated_ids)
    completed_count = len(generated_ids)
    
    # Statistics
    total_submitted = completed_count  # Total submitted tasks (including completed)
    
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            with tqdm(
                total=target_count, 
                desc=f"{split_type}/{image_count_category}",
                unit="sample",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ) as pbar:
                pbar.update(completed_count)
                
                # Submit initial tasks
                sample_idx = 0
                while len(futures) < MAX_WORKERS * 2 and sample_idx < len(filtered_samples) and completed_count < target_count:
                    if stop_event.is_set():
                        break
                    
                    sample = filtered_samples[sample_idx]
                    future = executor.submit(
                        worker_task,
                        current_idx,
                        sample,
                        data_dir,
                        image_root,
                        final_dir,
                        split_type,
                        image_count_category,
                        stop_event,
                        GEMINI_API_KEY,
                        GEMINI_MODEL_NAME,
                        MAX_TRIES,
                        generated_ids
                    )
                    futures.append(future)
                    current_idx += 1
                    sample_idx += 1
                    total_submitted += 1
                
                # Process completed tasks and submit new ones
                while completed_count < target_count and not stop_event.is_set():
                    # Check completed tasks
                    done_futures = []
                    for future in futures:
                        if future.done():
                            try:
                                result = future.result()
                                if result:
                                    completed_count += 1
                                    pbar.update(1)
                                    
                                    # Update progress bar description with detailed statistics
                                    success_rate = (completed_count / total_submitted * 100) if total_submitted > 0 else 0
                                    pbar.set_description(
                                        f"{split_type}/{image_count_category} | "
                                        f"Done:{completed_count}/{target_count} | "
                                        f"Submitted:{total_submitted} | "
                                        f"SuccessRate:{success_rate:.1f}% | "
                                        f"Running:{len(futures)-len(done_futures)}"
                                    )
                                    
                                    # Set stop event if target count is reached
                                    if completed_count >= target_count:
                                        stop_event.set()
                                        break
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                            done_futures.append(future)
                    
                    # Remove completed tasks
                    for future in done_futures:
                        futures.remove(future)
                    
                    # Submit new tasks
                    while len(futures) < MAX_WORKERS * 2 and sample_idx < len(filtered_samples) and completed_count < target_count:
                        if stop_event.is_set():
                            break
                        
                        sample = filtered_samples[sample_idx]
                        future = executor.submit(
                            worker_task,
                            current_idx,
                            sample,
                            data_dir,
                            image_root,
                            final_dir,
                            split_type,
                            image_count_category,
                            stop_event,
                            GEMINI_API_KEY,
                            GEMINI_MODEL_NAME,
                            MAX_TRIES,
                            generated_ids
                        )
                        futures.append(future)
                        current_idx += 1
                        sample_idx += 1
                        total_submitted += 1
                    
                    # If no more tasks and all tasks are done, exit the loop
                    if sample_idx >= len(filtered_samples) and len(futures) == 0:
                        break
                    
                    # Brief sleep to avoid high CPU usage
                    time.sleep(0.1)
                
                # Wait for all remaining tasks to complete
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result and completed_count < target_count:
                            completed_count += 1
                            pbar.update(1)
                            
                            # Update progress bar description
                            success_rate = (completed_count / total_submitted * 100) if total_submitted > 0 else 0
                            pbar.set_description(
                                f"{split_type}/{image_count_category} | "
                                f"Done:{completed_count}/{target_count} | "
                                f"Submitted:{total_submitted} | "
                                f"SuccessRate:{success_rate:.1f}%"
                            )
                    except Exception as e:
                        print(f"Task execution exception: {e}")
    
    except KeyboardInterrupt:
        print("\nInterrupt signal received, stopping...")
        stop_event.set()
        raise
    
    print(f"\n{split_type}/{image_count_category} done: {completed_count}/{target_count}")


def main():
    """Main function"""
    print("=" * 80)
    print("Illustration data generation script")
    print("=" * 80)
    print(f"Split directory: {SPLIT_DIR}")
    print(f"Final directory: {FINAL_DIR}")
    print(f"Generation config: {GEN_CONFIG}")
    print("=" * 80)
    
    # Create final directory
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process train and eval data
    for split_type in ["train", "eval"]:
        print(f"\nProcessing {split_type} data...")
        
        for image_count_category, config in GEN_CONFIG.items():
            target_count = config.get(split_type, 0)
            
            if target_count <= 0:
                print(f"Skipping {split_type}/{image_count_category} data generation (target count is 0)")
                continue
            
            # Load already-generated unique IDs
            generated_ids = load_generated_ids(FINAL_DIR, split_type, image_count_category)
            print(f"Loaded {len(generated_ids)} already-generated sample IDs")
            
            # Process data
            process_split_data(
                split_dir=SPLIT_DIR,
                final_dir=FINAL_DIR,
                split_type=split_type,
                image_count_category=image_count_category,
                target_count=target_count,
                generated_ids=generated_ids
            )
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()

