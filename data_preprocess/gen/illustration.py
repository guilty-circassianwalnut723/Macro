import os
#!/usr/bin/env python3
"""
Illustration数据生成脚本

功能：
1. 从split/illustration目录读取train数据（仅处理train数据）
2. 使用gemini-3-pro-preview重写文本
3. 保存到final/illustration/{train}/{image_count_category}/data和json目录
4. 支持唯一识别编号，避免重复生成
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

# 添加utils路径
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from utils.common import (
    get_image_count_category,
    load_generated_ids,
    save_sample_data,
    generate_unique_id
)

# 确保项目根目录在路径中
MACRO_DIR = CURRENT_DIR.parent.parent
if str(MACRO_DIR) not in sys.path:
    sys.path.insert(0, str(MACRO_DIR))

from api_generator.text_generator.gemini_api import GeminiAPIGenerator

# ====== 配置参数 ======
# 原始图文数据所在目录（包含conversation信息的原始数据，用于构建上下文）
DATA_DIR = MACRO_DIR / "data"  # Source data directory
SPLIT_DIR = DATA_DIR / "split" / "illustration"
FINAL_DIR = DATA_DIR / "final" / "illustration"
# Gemini API 配置
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = "gemini-3-pro-preview"

# 生成配置：{image_count_category: {train: count, eval: count}}
GEN_CONFIG = {
    "1-3": {"train": 30000, "eval": 500},
    "4-5": {"train": 30000, "eval": 500},
    "6-7": {"train": 30000, "eval": 500}, # 200
    ">=8": {"train": 47500, "eval": 500}, # 10000
}

# 线程配置
MAX_WORKERS = 64
MAX_TRIES = 3

# 随机种子配置
RANDOM_SEED = 42

# 日志配置
LOG_TO_SHELL = False  # 是否输出日志到shell

# 占位符token
PLACEHOLDER_TOKEN = "<IMAGE_TOKEN>"
REWRITE_TOKEN = "<IMAGE TOKEN>"

# Prompt模板
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

# ====== 线程局部存储 ======
thread_local = threading.local()

# ====== 唯一ID检查锁 ======
unique_id_lock = threading.Lock()
# =============================


@dataclass
class SequenceEntry:
    """序列条目"""
    entry_type: str  # "text" or "image"
    content: Optional[str] = None
    image_idx: Optional[int] = None
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {"type": self.entry_type}
        if self.entry_type == "text":
            result["content"] = self.content
        else:
            result["image_idx"] = self.image_idx
            result["image_path"] = self.image_path
        return result


@dataclass
class SampleContext:
    """样本上下文"""
    text_content: str
    image_paths: List[str]
    sequence: List[SequenceEntry]
    missing_images: List[str]

    @property
    def image_count(self) -> int:
        """图像数量"""
        return len(self.image_paths)

    @property
    def context_image_count(self) -> int:
        """上下文图像数量"""
        return len(self.image_paths) - 1 if len(self.image_paths) > 0 else 0


def get_or_create_generator(
    gemini_api_key: str,
    gemini_model_name: str,
    max_try: int
) -> GeminiAPIGenerator:
    """
    获取或创建线程局部生成器
    
    Args:
        gemini_api_key: Gemini API密钥
        gemini_model_name: Gemini模型名称
        max_try: 最大重试次数
    
    Returns:
        文本生成器
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
    从split目录加载数据
    
    Args:
        split_dir: split目录
        split_type: "train" 或 "eval"
    
    Returns:
        样本列表
    """
    all_samples = []
    
    # 尝试加载单个json文件
    json_file = split_dir / f"{split_type}.json"
    if json_file.exists():
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"警告: 加载文件失败 {json_file}: {e}")
    
    # 尝试加载多个json文件（如果数据被拆分）
    json_files = sorted(split_dir.glob(f"{split_type}_*.json"))
    if json_files:
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_samples.extend(data)
            except Exception as e:
                print(f"警告: 加载文件失败 {json_file}: {e}")
                continue
        if all_samples:
            return all_samples
    
    # 兼容旧格式：尝试加载jsonl文件
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
            print(f"警告: 加载文件失败 {jsonl_file}: {e}")
    
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
                print(f"警告: 加载文件失败 {jsonl_file}: {e}")
                continue
        if all_samples:
            return all_samples
    
    # 兼容旧格式：尝试加载JSON文件（带前缀的）
    for json_file in split_dir.glob(f"{split_type}_*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_samples.extend(data)
        except Exception as e:
            print(f"警告: 加载文件失败 {json_file}: {e}")
            continue
    
    return all_samples


def load_original_sample(data_dir: Path, source_file: str, source_line: int) -> Optional[Dict[str, Any]]:
    """
    从原始数据文件中加载样本
    
    Args:
        data_dir: 数据目录
        source_file: 源文件路径
        source_line: 源文件行号
    
    Returns:
        样本数据或None
    """
    file_path = data_dir / source_file if not Path(source_file).is_absolute() else Path(source_file)
    if LOG_TO_SHELL:
        print(f"    [加载原始样本] 尝试加载: data_dir={data_dir}, source_file={source_file}, source_line={source_line}")
        print(f"    [加载原始样本] 完整路径: {file_path}")
    
    if not file_path.exists():
        if LOG_TO_SHELL:
            print(f"    [加载原始样本] 文件不存在: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                if line_num == source_line:
                    line = line.strip()
                    if not line:
                        if LOG_TO_SHELL:
                            print(f"    [加载原始样本] 第{source_line}行为空")
                        return None
                    sample = json.loads(line)
                    if LOG_TO_SHELL:
                        print(f"    [加载原始样本] 成功加载第{source_line}行")
                        phrases = sample.get('phrases', [])
                        print(f"    [加载原始样本] 样本包含{len(phrases)}个phrases")
                    return sample
        if LOG_TO_SHELL:
            print(f"    [加载原始样本] 文件行数不足，找不到第{source_line}行")
    except Exception as e:
        if LOG_TO_SHELL:
            print(f"    [加载原始样本] 加载失败: {e}")
        return None
    return None


def extract_phrases(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    提取phrases列表（参考gen.py的实现）
    
    Args:
        sample: 样本数据
    
    Returns:
        phrases列表
    """
    phrases: List[Dict[str, Any]] = []
    
    # 首先尝试直接从phrases字段获取
    direct_phrases = sample.get('phrases', [])
    if isinstance(direct_phrases, list):
        phrases.extend(direct_phrases)
        if LOG_TO_SHELL:
            print(f"    [提取phrases] 从phrases字段提取: {len(phrases)}")
    
    # 从conversation中提取phrases（参考gen.py的实现）
    conversations = sample.get("conversation", [])
    if isinstance(conversations, dict):
        conversations = conversations.get("phrases", [])
    if LOG_TO_SHELL:
        print(f"    [提取phrases] conversations类型: {type(conversations)}, 长度: {len(conversations) if isinstance(conversations, list) else 'N/A'}")
    
    for convo in conversations:
        if not isinstance(convo, dict):
            continue
        convo_phrases = convo.get("phrases", [])
        if isinstance(convo_phrases, list):
            phrases.extend(convo_phrases)
            if LOG_TO_SHELL:
                print(f"    [提取phrases] 从conversation中提取: +{len(convo_phrases)}")
    
    if LOG_TO_SHELL:
        print(f"    [提取phrases] 总计提取: {len(phrases)}")
    
    return phrases


def resolve_image_path(
    image_info: Dict[str, Any],
    image_root: Optional[Path],
    data_dir: Optional[Path],
) -> Optional[Path]:
    """
    解析图像路径（参考gen.py的实现）
    
    Args:
        image_info: 图像信息
        image_root: 图像根目录
        data_dir: 数据目录
    
    Returns:
        解析后的图像路径或None
    """
    candidate_paths: List[Path] = []
    
    # 尝试从image字段获取路径（参考gen.py的实现）
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
    
    # 尝试从img_path字段获取路径（参考gen.py的实现）
    relative_path = image_info.get("img_path")
    if isinstance(relative_path, str) and relative_path.strip():
        if image_root:
            candidate_paths.append(image_root / relative_path)
        if data_dir:
            candidate_paths.append(data_dir / relative_path)
        candidate_paths.append(Path(relative_path))
    
    # 兼容旧格式：尝试从path或filepath字段获取路径
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
        print(f"    [解析路径] 图像信息: {image_info}")
        print(f"    [解析路径] 候选路径数量: {len(candidate_paths)}")
    
    # 尝试每个候选路径
    for candidate in candidate_paths:
        if candidate.exists():
            if LOG_TO_SHELL:
                print(f"    [解析路径] 找到有效路径: {candidate}")
            return candidate
        elif LOG_TO_SHELL:
            print(f"    [解析路径] 路径不存在: {candidate}")
    
    # 如果所有路径都不存在，返回第一个候选路径（如果存在）或None
    if candidate_paths:
        if LOG_TO_SHELL:
            print(f"    [解析路径] 所有路径都不存在，返回第一个候选路径: {candidate_paths[0]}")
        return candidate_paths[0]
    
    if LOG_TO_SHELL:
        print(f"    [解析路径] 无法解析路径，返回None")
    return None


def get_image_count_from_sample(sample: Dict[str, Any]) -> Optional[int]:
    """
    从split数据中获取图像数量
    
    Args:
        sample: split数据样本
    
    Returns:
        图像数量或None（总图像数量，包括目标图像）
    """
    if LOG_TO_SHELL:
        print(f"    [获取图像数量] 开始获取图像数量")
        print(f"    [获取图像数量] sample中的image_count: {sample.get('image_count')}")
        print(f"    [获取图像数量] sample中的actual_image_count: {sample.get('actual_image_count')}")
    
    # 直接从split数据中获取image_count（总图像数量，包括目标图像）
    result = sample.get('image_count')
    if LOG_TO_SHELL:
        print(f"    [获取图像数量] 从split数据获取image_count: {result}")
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
    构建样本上下文
    
    Args:
        sample: 样本数据
        original_sample: 原始样本数据
        image_count: 实际使用的图像数量（包括目标图像）
        data_dir: 数据目录
        image_root: 图像根目录
        true_index: true索引（0-based）
    
    Returns:
        样本上下文
    """
    phrases = extract_phrases(original_sample)
    sequence = []
    text_parts = []
    image_paths = []
    missing_images = []
    
    if LOG_TO_SHELL:
        print(f"[构建上下文] 开始构建样本上下文")
        print(f"  - 期望图像数量: {image_count}")
        print(f"  - true_index: {true_index}")
        print(f"  - phrases总数: {len(phrases)}")
        print(f"  - data_dir: {data_dir}")
        print(f"  - image_root: {image_root}")
    
    # 参考gen.py的实现，按顺序处理phrases
    context_image_paths: List[str] = []  # 上下文图像（不包括target image）
    target_image_path: Optional[str] = None
    
    # 如果提供了true_index，则只处理true_index之前的图像（不包括true_index本身，因为它是目标图像）
    context_image_count = image_count - 1  # 上下文图像数量（不包括target image）
    if true_index is not None:
        context_image_count = true_index
    
    if LOG_TO_SHELL:
        print(f"  - 上下文图像数量: {context_image_count}")
    
    # 收集上下文图像和target image（参考gen.py的实现，处理所有phrases，根据image_idx判断）
    image_idx = 0
    for phrase_idx, phrase in enumerate(phrases):
        if LOG_TO_SHELL:
            print(f"  - 处理phrase[{phrase_idx}]: phrase结构={phrase}")
        
        # 处理文本（参考gen.py的实现）
        text_info = phrase.get("text")
        if text_info:
            content = text_info.get("content") if isinstance(text_info, dict) else None
            if isinstance(content, str):
                normalized = " ".join(content.replace("\n", " ").split())
                if normalized:
                    text_parts.append(normalized)
                    sequence.append(SequenceEntry(entry_type='text', content=normalized))
                    if LOG_TO_SHELL:
                        print(f"    -> 添加文本: {normalized[:50]}...")
        
        # 处理图像（参考gen.py的实现）
        image_info = phrase.get("image")
        if image_info:
            if LOG_TO_SHELL:
                print(f"    -> 图像信息: {image_info}, 当前图像索引: {image_idx}")
            
            resolved = resolve_image_path(image_info, image_root, data_dir)
            if resolved is None:
                missing_images.append(str(image_info))
                image_idx += 1
                if LOG_TO_SHELL:
                    print(f"    -> 图像解析失败: {image_info}")
                continue
            resolved = resolved.resolve()
            if not resolved.exists():
                missing_images.append(str(resolved))
                image_idx += 1
                if LOG_TO_SHELL:
                    print(f"    -> 图像文件不存在: {resolved}")
                continue
            
            # 根据图像索引判断是上下文图像还是目标图像（参考gen.py的实现）
            if image_idx < context_image_count:
                # 上下文图像
                context_image_paths.append(str(resolved))
                text_parts.append(PLACEHOLDER_TOKEN)
                sequence.append(SequenceEntry(
                    entry_type='image',
                    image_idx=len(context_image_paths) - 1,
                    image_path=str(resolved)
                ))
                if LOG_TO_SHELL:
                    print(f"    -> 添加上下文图像: {resolved}, image_idx={image_idx}")
            elif (true_index is not None and image_idx == true_index) or (true_index is None and image_idx == image_count - 1):
                # target image（暂时保存，循环后添加）
                target_image_path = str(resolved)
                if LOG_TO_SHELL:
                    print(f"    -> 找到目标图像: {resolved}, image_idx={image_idx}")
            
            image_idx += 1
    
    # 构建最终的image_paths：上下文图像 + target image（用于传递给LLM）（参考gen.py的实现）
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
            print(f"    -> 添加目标图像到最终列表: {target_image_path}")
    
    # 验证PLACEHOLDER_TOKEN数量（参考gen.py的实现）
    placeholder_token_count = len([item for item in text_parts if item == PLACEHOLDER_TOKEN])
    if LOG_TO_SHELL:
        print(f"    -> PLACEHOLDER_TOKEN数量: {placeholder_token_count}, 期望: {image_count}")
    if placeholder_token_count != image_count:
        if LOG_TO_SHELL:
            print(f"    [错误] PLACEHOLDER_TOKEN数量不正确: {placeholder_token_count} != {image_count}")
    
    text_content = ' '.join(text_parts)
    
    if LOG_TO_SHELL:
        print(f"[构建上下文] 完成")
        print(f"  - 实际找到图像数量: {len(image_paths)}")
        print(f"  - 缺失图像数量: {len(missing_images)}")
        print(f"  - 文本片段数量: {len(text_parts)}")
        print(f"  - 序列条目数量: {len(sequence)}")
    
    return SampleContext(
        text_content=text_content,
        image_paths=image_paths,
        sequence=sequence,
        missing_images=missing_images
    )


def ensure_sequence_valid(context: SampleContext, expected_count: int) -> Tuple[bool, Optional[str]]:
    """
    确保序列有效
    
    Args:
        context: 样本上下文
        expected_count: 期望的图像数量
    
    Returns:
        (is_valid, error_message)
    """
    if LOG_TO_SHELL:
        print(f"    [验证序列] 开始验证")
        print(f"      - 期望图像数量: {expected_count}")
        print(f"      - 实际图像数量: {context.image_count}")
        print(f"      - 图像路径列表: {context.image_paths}")
        print(f"      - 缺失图像: {context.missing_images}")
        print(f"      - 序列条目数: {len(context.sequence)}")
    
    if context.image_count != expected_count:
        error_msg = f"图像数量不匹配: 期望 {expected_count}, 实际 {context.image_count}"
        if LOG_TO_SHELL:
            print(f"    [验证序列] 验证失败: {error_msg}")
            print(f"      - 详细分析:")
            print(f"        * 图像路径列表长度: {len(context.image_paths)}")
            print(f"        * 每个图像路径存在性: {[(p, Path(p).exists()) for p in context.image_paths]}")
        return False, error_msg
    
    if context.missing_images:
        error_msg = f"缺失图像: {len(context.missing_images)} 个"
        if LOG_TO_SHELL:
            print(f"    [验证序列] 验证失败: {error_msg}")
            print(f"      - 缺失图像详情: {context.missing_images}")
        return False, error_msg
    
    for img_path in context.image_paths:
        if not Path(img_path).exists():
            error_msg = f"图像文件不存在: {img_path}"
            if LOG_TO_SHELL:
                print(f"    [验证序列] 验证失败: {error_msg}")
            return False, error_msg
    
    if LOG_TO_SHELL:
        print(f"    [验证序列] 验证通过")
    return True, None


def build_prompt(sample: Dict[str, Any], context: SampleContext) -> str:
    """
    构建prompt
    
    Args:
        sample: 样本数据
        context: 样本上下文
    
    Returns:
        prompt字符串
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
    规范化布尔列表
    
    Args:
        values: 值列表
        expected_len: 期望长度
    
    Returns:
        规范化后的布尔列表
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
    规范化整数
    
    Args:
        value: 值
        min_value: 最小值
        max_value: 最大值
    
    Returns:
        规范化后的整数
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
    """验证重写文本中的 REWRITE_TOKEN 数量
    
    Args:
        rewritten: 重写的文本
        required_token_count: 贡献的上下文图像数量（不包括目标图像）
    
    Returns:
        验证后的文本，如果token数量比contributing_count多1且最后一个token在末尾，则去掉最后一个token
        如果token数量不符合要求，返回None
    """
    if required_token_count <= 0:
        if rewritten is None:
            return None
        stripped_zero = rewritten.strip()
        if not stripped_zero:
            return None
        token_count_zero = stripped_zero.count(REWRITE_TOKEN)
        if token_count_zero != 0:
            return None  # 无贡献图片时不应有token
        return stripped_zero

    if rewritten is None:
        return None
    if not isinstance(rewritten, str):
        return None
    stripped = rewritten.strip()
    if not stripped:
        return None
    
    token_count = stripped.count(REWRITE_TOKEN)
    
    # 判断token数量是否符合要求
    if token_count == required_token_count:
        # token数量正好等于贡献图像数量，正常返回
        return stripped
    elif token_count == required_token_count + 1:
        # token数量比贡献图像数量多1（包含了目标图像），需要检查最后一个token是否在末尾
        # 使用rsplit从右往左分割一次，分隔符是REWRITE_TOKEN
        parts = stripped.rsplit(REWRITE_TOKEN, 1)
        if len(parts) == 2:
            # parts[0]是最后token之前的内容，parts[1]是最后token之后的内容
            before_last_token = parts[0].rstrip()
            after_last_token = parts[1].strip()
            
            # 检查最后一个token是否在末尾（即after_last_token应该为空或只有标点符号）
            # 如果最后一个token后面还有实质性内容，则认为验证失败
            if after_last_token and not all(c in '.,;:!?\'"' for c in after_last_token):
                # 最后一个token不在末尾，验证失败
                return None
            
            # 去掉最后一个token
            return before_last_token.strip()
        # 如果分割失败（理论上不应该发生），返回None
        return None
    else:
        # token数量不符合要求，返回None
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
    处理单个样本
    
    Args:
        sample: split数据样本
        data_dir: 原始图文数据所在目录（包含conversation信息的原始数据，用于构建上下文）
        image_root: 图像根目录
        text_generator: 文本生成器
        final_dir: final目录
        split_type: "train" 或 "eval"
        image_count_category: 图像数量类别
        idx: 样本索引
        generated_ids: 已生成的唯一ID集合
    
    Returns:
        是否处理成功
    """
    source_file = sample.get('source_file')
    source_line = sample.get('source_line')
    true_index = sample.get('true_index')
    
    if not source_file or source_line is None or true_index is None:
        return False
    
    # 生成唯一ID
    # 如果 SAVE_ORIGINAL_STRING=True，需要获取原始字符串以便保存
    from utils.common import SAVE_ORIGINAL_STRING
    unique_id_result = generate_unique_id("illustration",
                                         return_original=SAVE_ORIGINAL_STRING,
                                         source_file=source_file,
                                         source_line=source_line,
                                         true_index=true_index)
    
    # 提取用于检查的唯一ID（如果是元组，使用 MD5 哈希；如果是字符串，直接使用）
    unique_id = unique_id_result[0] if isinstance(unique_id_result, tuple) else unique_id_result
    
    # 检查 unique_id 是否已生成（线程安全）
    with unique_id_lock:
        if unique_id in generated_ids:
            return False
        # 立即添加 unique_id，防止其他线程同时生成
        generated_ids.add(unique_id)
    
    # 加载原始样本（从原始图文数据目录加载，包含conversation信息）
    if LOG_TO_SHELL:
        print(f"[样本 {idx}] 开始处理")
        print(f"  - source_file: {source_file}")
        print(f"  - source_line: {source_line}")
        print(f"  - true_index: {true_index}")
        print(f"  - data_dir: {data_dir} (用于加载原始图文数据)")
    
    original_sample = load_original_sample(data_dir, source_file, source_line)
    if not original_sample:
        if LOG_TO_SHELL:
            print(f"[样本 {idx}] 加载原始样本失败: data_dir={data_dir}, source_file={source_file}, source_line={source_line}")
        # 移除 unique_id（如果添加了）
        with unique_id_lock:
            generated_ids.discard(unique_id)
        return False
    
    if LOG_TO_SHELL:
        print(f"[样本 {idx}] 成功加载原始样本")
        phrases = extract_phrases(original_sample)
        print(f"  - 原始样本phrases数量: {len(phrases)}")
        for i, phrase in enumerate(phrases[:10]):  # 只打印前10个
            phrase_type = phrase.get('type', 'unknown')
            if not phrase_type:
                # 尝试从text或image字段判断类型
                if phrase.get('text'):
                    phrase_type = 'text'
                elif phrase.get('image'):
                    phrase_type = 'image'
            print(f"    phrases[{i}]: type={phrase_type}")
    
    # 获取图像数量（从split数据获取）
    image_count = get_image_count_from_sample(sample)
    if not image_count:
        if LOG_TO_SHELL:
            print(f"[样本 {idx}] 无法获取图像数量")
            print(f"  - sample中的image_count: {sample.get('image_count')}")
            print(f"  - sample中的actual_image_count: {sample.get('actual_image_count')}")
        # 移除 unique_id（如果添加了）
        with unique_id_lock:
            generated_ids.discard(unique_id)
        return False
    
    if LOG_TO_SHELL:
        print(f"[样本 {idx}] 图像数量: {image_count}")
    
    # 构建样本上下文
    context = build_sample_context(
        sample, original_sample, image_count, data_dir, image_root, true_index
    )
    
    # 验证序列
    if LOG_TO_SHELL:
        print(f"[样本 {idx}] 开始验证序列")
        print(f"  - unique_id: {unique_id}")
        print(f"  - source_file: {source_file}")
        print(f"  - source_line: {source_line}")
        print(f"  - true_index: {true_index}")
        print(f"  - 期望图像数量: {image_count}")
        print(f"  - 实际图像数量: {context.image_count}")
        print(f"  - 实际图像路径: {context.image_paths}")
        print(f"  - 缺失图像: {context.missing_images}")
        print(f"  - 序列条目: {len(context.sequence)}")
    
    is_valid, error_msg = ensure_sequence_valid(context, image_count)
    if not is_valid:
        if LOG_TO_SHELL:
            print(f"[样本 {idx}] 验证失败: {error_msg}")
            print(f"  - 详细信息:")
            print(f"    * 期望图像数量: {image_count}")
            print(f"    * 实际图像数量: {context.image_count}")
            print(f"    * 图像路径列表: {context.image_paths}")
            print(f"    * 缺失图像: {context.missing_images}")
            print(f"    * 序列条目: {[e.entry_type for e in context.sequence]}")
        print(f"样本 {idx} 验证失败: {error_msg}")
        # 移除 unique_id（如果添加了）
        with unique_id_lock:
            generated_ids.discard(unique_id)
        return False
    
    if LOG_TO_SHELL:
        print(f"[样本 {idx}] 验证通过")
    
    # 构建prompt
    if LOG_TO_SHELL:
        print(f"[样本 {idx}] 构建prompt")
    prompt = build_prompt(sample, context)
    
    # 加载图像
    if LOG_TO_SHELL:
        print(f"[样本 {idx}] 开始加载图像，共{len(context.image_paths)}张")
    images = []
    for i, img_path in enumerate(context.image_paths):
        if LOG_TO_SHELL:
            print(f"  - 加载图像[{i+1}/{len(context.image_paths)}]: {img_path}")
        try:
            img_path_obj = Path(img_path)
            if not img_path_obj.exists():
                if LOG_TO_SHELL:
                    print(f"    [加载图像] 路径不存在: {img_path}")
                raise FileNotFoundError(f"图像文件不存在: {img_path}")
            img = Image.open(img_path)
            images.append(img)
            if LOG_TO_SHELL:
                print(f"    [加载图像] 成功加载: {img_path}, 尺寸={img.size}")
        except Exception as e:
            if LOG_TO_SHELL:
                print(f"    [加载图像] 加载失败: {img_path}, 错误={e}")
            print(f"样本 {idx} 加载图像失败 {img_path}: {e}")
            # 移除 unique_id（如果添加了）
            with unique_id_lock:
                generated_ids.discard(unique_id)
            return False
    
    if LOG_TO_SHELL:
        print(f"[样本 {idx}] 成功加载所有{len(images)}张图像")
    
    # 调用LLM生成结果
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
        # 计算贡献的上下文图像数量（不包括目标图像）
        contributing_count = sum(1 for flag in image_contributions if flag)
        rewritten_text = validate_rewritten_text(
            response.get('rewritten_text'),
            contributing_count  # 只包括贡献的上下文图像数量，不包括目标图像
        )
        guidance_score = normalize_int(response.get('guidance_score', 5))
        training_score = normalize_int(response.get('training_score', 5))
        reasoning = response.get('reasoning', '')
        
    except Exception as e:
        print(f"样本 {idx} LLM调用失败: {e}")
        # 移除 unique_id（如果添加了）
        with unique_id_lock:
            generated_ids.discard(unique_id)
        return False
    
    # 验证重写文本是否有效（必须包含正确数量的 <IMAGE TOKEN>）
    if rewritten_text is None:
        if LOG_TO_SHELL:
            print(f"[样本 {idx}] 重写文本验证失败: token数量不匹配或格式错误")
        # 移除 unique_id（如果添加了）
        with unique_id_lock:
            generated_ids.discard(unique_id)
        return False
    
    # 构建最终文本（必须使用重写文本，不再使用原始文本作为备选）
    instruction = rewritten_text
    
    # 将REWRITE_TOKEN（<IMAGE TOKEN>）转换为<image n>格式
    # 按照顺序替换：第一个token替换为<image 1>，第二个替换为<image 2>，以此类推
    # 只在有rewritten_text时进行转换（与参考文件保持一致）
    if rewritten_text and REWRITE_TOKEN in instruction:
        token_index = 1
        while REWRITE_TOKEN in instruction:
            instruction = instruction.replace(REWRITE_TOKEN, f"<image {token_index}>", 1)
            token_index += 1
    
    final_text = instruction
    
    # 保存数据
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
        unique_id_result,  # 可能是字符串或 (md5_hash, original_string) 元组
        json_data,
        image_files
    )
    
    if not success:
        # 移除 unique_id（如果保存失败）
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
    工作线程任务
    
    Args:
        idx: 样本索引
        sample: split数据样本
        data_dir: 原始图文数据所在目录（包含conversation信息的原始数据，用于构建上下文）
        image_root: 图像根目录
        final_dir: final目录
        split_type: "train" 或 "eval"
        image_count_category: 图像数量类别
        stop_event: 停止事件
        gemini_api_key: Gemini API密钥
        gemini_model_name: Gemini模型名称
        max_try: 最大重试次数
        generated_ids: 已生成的唯一ID集合
    
    Returns:
        是否处理成功
    """
    if stop_event.is_set():
        return False
    
    # 获取或创建线程局部生成器
    text_generator = get_or_create_generator(
        gemini_api_key, gemini_model_name, max_try
    )
    
    # 处理样本
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
    处理split数据并生成最终数据
    
    Args:
        split_dir: split目录
        final_dir: final目录
        split_type: "train" 或 "eval"
        image_count_category: 图像数量类别
        target_count: 目标生成数量
        generated_ids: 已生成的唯一ID集合
    """
    # 加载split数据
    samples = load_split_data(split_dir, split_type)
    
    # 过滤出符合image_count_category的样本
    filtered_samples = []
    for sample in samples:
        actual_image_count = sample.get('actual_image_count')
        if actual_image_count is not None:
            category = get_image_count_category(actual_image_count)
            if category == image_count_category:
                filtered_samples.append(sample)
    
    print(f"找到 {len(filtered_samples)} 个符合 {image_count_category} 的样本")
    
    # 随机打乱样本顺序，实现随机选择
    random.seed(RANDOM_SEED)
    random.shuffle(filtered_samples)
    
    # 数据目录和图像根目录（使用配置中的路径）
    data_dir = DATA_DIR  # 原始图文数据所在目录（包含conversation信息的原始数据，用于构建上下文）
    image_root = None  # 图像根目录（可选，根据实际情况设置）
    
    if LOG_TO_SHELL:
        print(f"  - data_dir: {data_dir} (用于加载原始图文数据)")
        print(f"  - image_root: {image_root}")
    
    # 创建停止事件
    stop_event = threading.Event()
    
    # 使用线程池处理
    current_idx = len(generated_ids)
    completed_count = len(generated_ids)
    
    # 统计信息
    total_submitted = completed_count  # 已提交的任务数（包括已完成的）
    
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
                
                # 提交初始任务
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
                
                # 处理完成的任务并提交新任务
                while completed_count < target_count and not stop_event.is_set():
                    # 检查完成的任务
                    done_futures = []
                    for future in futures:
                        if future.done():
                            try:
                                result = future.result()
                                if result:
                                    completed_count += 1
                                    pbar.update(1)
                                    
                                    # 更新进度条描述，显示详细统计信息
                                    success_rate = (completed_count / total_submitted * 100) if total_submitted > 0 else 0
                                    pbar.set_description(
                                        f"{split_type}/{image_count_category} | "
                                        f"完成:{completed_count}/{target_count} | "
                                        f"提交:{total_submitted} | "
                                        f"成功率:{success_rate:.1f}% | "
                                        f"运行中:{len(futures)-len(done_futures)}"
                                    )
                                    
                                    # 如果达到目标数量，设置停止事件
                                    if completed_count >= target_count:
                                        stop_event.set()
                                        break
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                            done_futures.append(future)
                    
                    # 移除已完成的任务
                    for future in done_futures:
                        futures.remove(future)
                    
                    # 提交新任务
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
                    
                    # 如果没有更多任务且所有任务都完成，退出循环
                    if sample_idx >= len(filtered_samples) and len(futures) == 0:
                        break
                    
                    # 短暂休眠，避免CPU占用过高
                    time.sleep(0.1)
                
                # 等待所有剩余任务完成
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result and completed_count < target_count:
                            completed_count += 1
                            pbar.update(1)
                            
                            # 更新进度条描述
                            success_rate = (completed_count / total_submitted * 100) if total_submitted > 0 else 0
                            pbar.set_description(
                                f"{split_type}/{image_count_category} | "
                                f"完成:{completed_count}/{target_count} | "
                                f"提交:{total_submitted} | "
                                f"成功率:{success_rate:.1f}%"
                            )
                    except Exception as e:
                        print(f"任务执行异常: {e}")
    
    except KeyboardInterrupt:
        print("\n收到中断信号，正在停止...")
        stop_event.set()
        raise
    
    print(f"\n{split_type}/{image_count_category} 完成: {completed_count}/{target_count}")


def main():
    """主函数"""
    print("=" * 80)
    print("Illustration数据生成脚本")
    print("=" * 80)
    print(f"Split目录: {SPLIT_DIR}")
    print(f"Final目录: {FINAL_DIR}")
    print(f"生成配置: {GEN_CONFIG}")
    print("=" * 80)
    
    # 创建final目录
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 处理train和eval数据
    for split_type in ["train", "eval"]:
        print(f"\n处理 {split_type} 数据...")
        
        for image_count_category, config in GEN_CONFIG.items():
            target_count = config.get(split_type, 0)
            
            if target_count <= 0:
                print(f"跳过 {split_type}/{image_count_category} 数据生成（目标数量为0）")
                continue
            
            # 加载已生成的唯一识别编号
            generated_ids = load_generated_ids(FINAL_DIR, split_type, image_count_category)
            print(f"已加载 {len(generated_ids)} 个已生成的样本ID")
            
            # 处理数据
            process_split_data(
                split_dir=SPLIT_DIR,
                final_dir=FINAL_DIR,
                split_type=split_type,
                image_count_category=image_count_category,
                target_count=target_count,
                generated_ids=generated_ids
            )
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()

