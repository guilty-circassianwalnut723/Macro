#!/usr/bin/env python3
"""
Qwen-Image-Edit 推理脚本

从filter目录加载数据，使用Qwen-Image-Edit模型进行推理，并保存结果
支持多GPU并行推理和断点续传
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import numpy as np

from PIL import Image
import torch
from diffsynth.pipelines.qwen_image import QwenImagePipeline
from diffsynth.core import ModelConfig
from diffsynth import load_state_dict

# 添加qwen source路径
SCRIPT_DIR = Path(__file__).parent
QWEN_DIR = SCRIPT_DIR.parent
MACRO_DIR = QWEN_DIR.parent  # Macro root directory
QWEN_SOURCE_PATH = QWEN_DIR / "source"
if str(QWEN_SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(QWEN_SOURCE_PATH))

# 导入本地utils
from utils import (
    load_data_for_task,  # 统一的数据加载接口
    SUPPORTED_TASKS, 
    THIRDPARTY_TASKS,
    IMAGE_NUM_CATEGORIES, 
    check_sample_exists,
    save_sample
)


def load_pipeline(
    model_configs: List[Dict[str, str]],
    processor_config: Dict[str, str],
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    device_id: Optional[int] = None,
    model_base_path: Optional[str] = None,
    transformer_path: Optional[str] = None,
    tokenizer_config: Optional[Dict[str, str]] = None,
    lora_path: Optional[str] = None,
):
    """
    加载Qwen-Image-Edit Pipeline
    
    Args:
        model_configs: 模型配置列表，每个元素包含 model_id 和 origin_file_pattern
        processor_config: 处理器配置，包含 model_id 和 origin_file_pattern
        device: 设备类型 ("cuda" 或 "cpu")
        torch_dtype: torch数据类型 ("float32", "float16", "bfloat16")
        device_id: 指定的GPU设备ID（0-based），如果为None，使用默认设备
        model_base_path: 模型基础路径，如果指定，会设置 DIFFSYNTH_MODEL_BASE_PATH 环境变量
        transformer_path: 训练后的 transformer 权重路径（可选），如果指定，会加载训练后的权重到 pipe.dit（全量微调）
        tokenizer_config: tokenizer 配置（可选），包含 model_id 和 origin_file_pattern
        lora_path: LoRA 权重路径（可选），如果指定，会加载 LoRA 权重到 pipe.dit
                   注意：transformer_path 和 lora_path 互斥，只能指定其中一个
    
    Returns:
        pipeline: QwenImagePipeline实例
    """
    # 检查 transformer_path 和 lora_path 互斥
    if transformer_path and lora_path:
        raise ValueError("transformer_path 和 lora_path 互斥，只能指定其中一个。"
                        "transformer_path 用于全量微调权重，lora_path 用于 LoRA 权重。")
    
    # 设置本地模型路径环境变量（如果指定）
    if model_base_path:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = model_base_path
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        print(f"设置本地模型路径: {model_base_path}")
        print(f"跳过 ModelScope 下载，使用本地模型")
    
    # 转换torch_dtype
    if torch_dtype == "float32":
        dtype = torch.float32
    elif torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")
    
    # 确定设备
    if device_id is not None:
        device_str = f"cuda:{device_id}"
    else:
        device_str = device
    
    # 设置环境变量
    if model_base_path:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = model_base_path
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    
    # 构建ModelConfig列表
    model_config_objs = []
    for config in model_configs:
        model_config_objs.append(
            ModelConfig(
                model_id=config["model_id"],
                origin_file_pattern=config["origin_file_pattern"]
            )
        )
    
    # 构建processor配置
    processor_config_obj = ModelConfig(
        model_id=processor_config["model_id"],
        origin_file_pattern=processor_config["origin_file_pattern"]
    )
    
    # 构建tokenizer配置（如果指定）
    tokenizer_config_obj = None
    if tokenizer_config:
        tokenizer_config_obj = ModelConfig(
            model_id=tokenizer_config["model_id"],
            origin_file_pattern=tokenizer_config["origin_file_pattern"]
        )
    
    # 加载pipeline
    print(f"正在加载 Qwen-Image-Edit Pipeline (设备: {device_str}, dtype: {dtype})...")
    pipeline = QwenImagePipeline.from_pretrained(
        torch_dtype=dtype,
        device=device_str,
        model_configs=model_config_objs,
        processor_config=processor_config_obj,
        tokenizer_config=tokenizer_config_obj,
    )
    
    # 如果指定了训练后的 transformer 路径，加载训练后的权重
    if transformer_path:
        # 处理相对路径（相对于 model_base_path）
        if not os.path.isabs(transformer_path) and model_base_path:
            transformer_path = os.path.join(model_base_path, transformer_path)
        
        # 检查文件是否存在
        if not os.path.exists(transformer_path):
            raise FileNotFoundError(f"训练后的 transformer 权重文件不存在: {transformer_path}")
        
        print(f"加载训练后的 transformer 权重: {transformer_path}")
        state_dict = load_state_dict(transformer_path)
        
        # 处理权重键名前缀（训练时可能保存了 "pipe.dit." 前缀）
        # 如果 state_dict 的键名包含 "pipe.dit." 前缀，需要移除
        if state_dict and any(key.startswith("pipe.dit.") for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                # 移除 "pipe.dit." 前缀
                if key.startswith("pipe.dit."):
                    new_key = key[len("pipe.dit."):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
            print(f"已移除权重键名的 'pipe.dit.' 前缀")
        
        pipeline.dit.load_state_dict(state_dict)
        print(f"训练后的 transformer 权重加载完成（全量微调）")
    
    # 如果指定了 LoRA 权重路径，加载 LoRA 权重
    # 参考: examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit-2511.py
    if lora_path:
        # 处理相对路径（相对于 model_base_path）
        if not os.path.isabs(lora_path) and model_base_path:
            lora_path = os.path.join(model_base_path, lora_path)
        
        # 检查文件是否存在
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA 权重文件不存在: {lora_path}")
        
        print(f"加载 LoRA 权重: {lora_path}")
        # 使用 QwenImagePipeline 的 load_lora 方法
        pipeline.load_lora(pipeline.dit, lora_path)
        print(f"LoRA 权重加载完成")
    
    print(f"Pipeline 加载完成")
    return pipeline


def process_sample(
    sample: Dict[str, Any],
    output_dir: Path,
    pipeline: QwenImagePipeline,
    inference_hyper: Dict[str, Any],
    gpu_id: int,
    seed: Optional[int] = None
) -> Tuple[bool, Optional[str]]:
    """
    处理单个样本
    
    Args:
        sample: 样本数据
        output_dir: 输出目录
        pipeline: 推理pipeline（每个线程使用独立的实例，无需锁保护）
        inference_hyper: 推理超参数
        gpu_id: GPU ID（用于日志）
        seed: 随机种子（可选）
    
    Returns:
        (成功标志, 错误信息)
    """
    try:
        idx = sample.get("idx", 0)
        seed = sample.get("seed", None)
        task = sample.get("task", "")
        
        # 断点续传检查
        # 对于 geneval 和 dpg，如果指定了 seed，检查特定 seed 的文件是否存在
        if seed is not None and task in ["geneval", "dpg"]:
            base_idx = sample.get("base_idx", idx)
            image_file = output_dir / f"{base_idx:08d}_seed{seed:02d}.jpg"
            json_file = output_dir / f"{base_idx:08d}_seed{seed:02d}.json"
            if image_file.exists() and json_file.exists():
                # 检查图像是否可读
                try:
                    with Image.open(image_file) as img:
                        img.verify()
                    with Image.open(image_file) as img:
                        img.convert("RGB").load()
                    return (True, None)
                except:
                    pass
        else:
            if check_sample_exists(output_dir, idx):
                return (True, None)
        
        prompt = sample.get("prompt", "")
        input_images = sample.get("input_images", [])
        output_image_path = sample.get("output_image", "")
        
        # 对于 omnicontext 任务，如果 input_images 包含 PIL Image 对象，先保存到本地
        if task == "omnicontext" or (not task and any(isinstance(img, Image.Image) for img in input_images)):
            has_pil_images = any(isinstance(img, Image.Image) for img in input_images)
            if has_pil_images:
                try:
                    SCRIPT_DIR = Path(__file__).parent
                    MACRO_DIR = SCRIPT_DIR.parent.parent
                    INFERENCE_UTILS_DIR = MACRO_DIR / "inference_utils"
                    if str(INFERENCE_UTILS_DIR) not in sys.path:
                        sys.path.insert(0, str(INFERENCE_UTILS_DIR))
                    
                    from thirdparty_tasks import save_omnicontext_input_images
                    
                    saved_paths = save_omnicontext_input_images(
                        sample_idx=idx,
                        input_images=input_images,
                        prompt=prompt
                    )
                    sample["input_images"] = saved_paths
                    input_images = saved_paths
                    print(f"[GPU {gpu_id}] 已保存 OmniContext 输入图像到本地: {len(saved_paths)} 张")
                except Exception as e:
                    print(f"[GPU {gpu_id}] 保存 OmniContext 输入图像失败: {e}")
        
        # 加载输入图像
        # 对于 geneval 和 dpg 任务，是纯文本生成图像任务，不需要输入图像
        is_text_to_image_task = task in ["geneval", "dpg"]
        
        edit_images = []
        if not is_text_to_image_task:
            for img_item in input_images:
                if isinstance(img_item, (str, Path)):
                    img_path = str(img_item)
                    if os.path.exists(img_path):
                        edit_images.append(Image.open(img_path).convert("RGB"))
                    else:
                        print(f"[GPU {gpu_id}] 警告: 图像文件不存在: {img_path}")
                elif isinstance(img_item, Image.Image):
                    edit_images.append(img_item.convert("RGB"))
                else:
                    print(f"[GPU {gpu_id}] 警告: 不支持的图像类型: {type(img_item)}")
            
            if len(edit_images) == 0:
                print(f"[GPU {gpu_id}] 警告: 样本 {idx} 没有输入图像，跳过")
                return (False, f"跳过样本 {idx}: 没有有效的输入图像")
        
        # 如果指定了 seed，设置随机种子
        # 对于 geneval 和 dpg，使用 base_seed (42) + seed 作为实际 seed
        # 仅对当前线程使用的 GPU 设 seed，避免多线程下 manual_seed_all 导致竞态/段错误
        if seed is not None:
            actual_seed = 42 + seed
            random.seed(actual_seed)
            np.random.seed(actual_seed)
            torch.manual_seed(actual_seed)
            if torch.cuda.is_available():
                with torch.cuda.device(gpu_id):
                    torch.cuda.manual_seed(actual_seed)
            # 更新 inference_hyper 中的 seed
            inference_hyper = inference_hyper.copy()
            inference_hyper['seed'] = actual_seed
        
        # 推理（每个线程使用独立的 pipeline 实例，无需锁保护）
        # 对于 t2i 任务（geneval, dpg），edit_image 应该为 None（而不是空列表）
        # 因为 QwenImageUnit_PromptEmbedder 在 edit_image=None 时会调用 encode_prompt（纯文本编码）
        # 而 edit_image=[] 时会调用 encode_prompt_edit_multi，无法处理空列表
        # print(f"Prompt: {prompt}, num images: {len(edit_images)}")
        if is_text_to_image_task:
            # 对于纯文本生成任务，传递 None 而不是空列表
            generated_image = pipeline(
                prompt,
                edit_image=None,  # None 表示纯文本生成
                **inference_hyper
            )
        else:
            # 注意: edit_image 必须是列表（非空）或 None
            generated_image = pipeline(
                prompt,
                edit_image=edit_images if edit_images else None,  # 如果为空列表，传递 None
                **inference_hyper
            )
        
        # 保存图像
        # 对于 geneval 和 dpg，如果指定了 seed，使用 base_idx 作为文件名前缀
        if seed is not None and task in ["geneval", "dpg"]:
            base_idx = sample.get("base_idx", idx)
            output_image_file = output_dir / f"{base_idx:08d}_seed{seed:02d}.jpg"
            save_idx = base_idx
        else:
            output_image_file = output_dir / f"{idx:08d}.jpg"
            save_idx = idx
        
        generated_image.save(output_image_file)
        # 设置文件权限为777（所有人可读写执行）
        os.chmod(output_image_file, 0o777)
        
        # 确保 sample 字典中包含 base_idx、idx、task_type、tag、include（与 bagel 一致，便于评测/可视化）
        sample_with_base_idx = sample.copy()
        if seed is not None and task in ["geneval", "dpg"]:
            sample_with_base_idx["base_idx"] = base_idx
        sample_with_base_idx["idx"] = save_idx  # 确保保存的 JSON 中 idx 与文件名一致
        if task == "omnicontext" and "task_type" not in sample_with_base_idx:
            sample_with_base_idx["task_type"] = sample.get("task_type", "")
        # geneval/dpg 评测需要 tag 和 include，确保保留
        if task in ["geneval", "dpg"] and "tag" in sample:
            sample_with_base_idx["tag"] = sample["tag"]
        if task in ["geneval", "dpg"] and "include" in sample:
            sample_with_base_idx["include"] = sample["include"]
        
        # 使用统一的保存接口
        save_sample(
            output_dir=output_dir,
            idx=save_idx,
            sample=sample_with_base_idx,
            output_image_path=str(output_image_file),
            target_image_path=output_image_path,
            seed=seed  # 传递 seed 参数
        )
        
        return (True, None)
        
    except Exception as e:
        sample_idx = sample.get("idx", "unknown")
        error_msg = f"处理样本 {sample_idx} 时出错: {e}"
        import traceback
        traceback.print_exc()
        return (False, error_msg)


def run_inference(
    model_configs: List[Dict[str, str]],
    processor_config: Dict[str, str],
    task: str,
    image_num_category: str = "all",
    output_dir: str = "./outputs",
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    inference_hyper: Optional[Dict[str, Any]] = None,
    num_workers: int = 1,
    use_refine_prompt: bool = False,  # 对于 geneval，是否使用 refine prompt
    model_base_path: Optional[str] = None,  # 模型基础路径，用于本地模型加载
    transformer_path: Optional[str] = None,  # 训练后的 transformer 权重路径（全量微调）
    tokenizer_config: Optional[Dict[str, str]] = None,  # tokenizer 配置
    lora_path: Optional[str] = None,  # LoRA 权重路径（LoRA 微调）
    data_root: Optional[str] = None,  # 数据根目录，如果为None，使用默认路径
) -> int:
    """
    运行推理（多GPU并行，支持断点续传）
    
    Args:
        model_configs: 模型配置列表
        processor_config: 处理器配置
        task: 任务类型 (customization, illustration, spatial, temporal, all)
        image_num_category: 图像数量类别 (1-3, 4-5, 6-7, >=8, all)
        output_dir: 输出目录
        device: 设备类型
        torch_dtype: torch数据类型
        inference_hyper: 推理超参数
        num_workers: 并行工作线程数量（总模型数量），默认1。模型会按顺序分配到所有可用GPU上
    """
    # 默认推理超参数
    if inference_hyper is None:
        inference_hyper = dict(
            seed=42,
            num_inference_steps=40,
            height=768,
            width=768,
            edit_image_auto_resize=True,
            zero_cond_t=True,  # Qwen-Image-Edit-2511 的特殊参数
        )
    
    # 设置随机种子以确保可复现性
    seed = inference_hyper.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"设置随机种子: {seed}")
    
    # 确定可用的GPU数量
    if device == "cuda":
        available_gpus = torch.cuda.device_count()
        if available_gpus <= 0:
            raise RuntimeError("没有可用的GPU")
    else:
        available_gpus = 1
    
    # 将num_workers个模型按顺序分配到所有GPU上（轮询分配）
    actual_num_workers = num_workers
    
    print(f"可用GPU数量: {available_gpus}")
    print(f"将加载 {actual_num_workers} 个模型，按顺序分配到所有GPU上")
    
    # 一次性加载所有模型到GPU（按顺序轮询分配到所有GPU）
    print(f"正在加载 {actual_num_workers} 个模型...")
    pipelines_data = []
    for model_idx in range(actual_num_workers):
        # 按顺序轮询分配到GPU
        if device == "cuda":
            gpu_id = model_idx % available_gpus
        else:
            gpu_id = None
        
        print(f"加载模型 {model_idx + 1}/{actual_num_workers} 到 GPU {gpu_id if gpu_id is not None else 'CPU'}...")
        pipeline = load_pipeline(
            model_configs=model_configs,
            processor_config=processor_config,
            device=device,
            torch_dtype=torch_dtype,
            device_id=gpu_id,
            model_base_path=model_base_path,
            transformer_path=transformer_path,
            tokenizer_config=tokenizer_config,
            lora_path=lora_path,
        )
        
        pipelines_data.append({
            'model_idx': model_idx,
            'gpu_id': gpu_id if gpu_id is not None else 0,
            'pipeline': pipeline,  # 每个模型实例独立，无需锁保护
        })
        print(f"模型 {model_idx + 1} 加载完成（GPU {gpu_id if gpu_id is not None else 'CPU'}）")
    
    print("所有模型加载完成！")
    
    # 确定要处理的tasks
    if task == "all":
        tasks_to_process = SUPPORTED_TASKS
    elif task in THIRDPARTY_TASKS:
        tasks_to_process = [task]
    else:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}. Supported tasks: {SUPPORTED_TASKS + THIRDPARTY_TASKS}")
        tasks_to_process = [task]
    
    # 处理每个task
    for current_task in tasks_to_process:
        print(f"\n处理任务: {current_task}")
        if current_task not in THIRDPARTY_TASKS:
            print(f"图像数量类别: {image_num_category}")
        
        # 使用统一的数据加载接口
        try:
            samples = load_data_for_task(
                task=current_task,
                image_num_category=image_num_category if image_num_category != "all" and current_task not in THIRDPARTY_TASKS else None,
                data_root=Path(data_root) if data_root else None,
                use_refine_prompt=use_refine_prompt
            )
            print(f"加载了 {len(samples)} 个样本")
        except Exception as e:
            print(f"加载数据失败: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        if len(samples) == 0:
            print(f"没有找到样本，跳过")
            continue
        
        # 创建输出目录
        # output_dir 已经包含了 task 和 category 信息（由 run.py 传入）
        # 所以直接使用 output_dir 作为 task_output_dir
        task_output_dir = Path(output_dir)
        task_output_dir.mkdir(parents=True, exist_ok=True)
        # 设置目录权限为777（所有人可读写执行）
        os.chmod(task_output_dir, 0o777)
        
        # 对于 geneval 和 dpg，需要为每个 prompt 生成多个 seed 的结果
        # 每个 prompt 生成 4 个样本，使用 seed [0, 1, 2, 3]
        NUM_SEEDS_PER_PROMPT = 4
        
        # 检查是否是 geneval 或 dpg 任务
        is_multi_seed_task = current_task in ["geneval", "dpg"]
        
        # 扩展样本列表：对于 geneval 和 dpg，每个 prompt 需要生成多个 seed 的结果
        if is_multi_seed_task:
            expanded_samples = []
            for sample in samples:
                base_idx = sample.get("idx", 0)
                for seed in range(NUM_SEEDS_PER_PROMPT):
                    expanded_sample = sample.copy()
                    expanded_sample["seed"] = seed
                    expanded_sample["base_idx"] = base_idx  # 保存原始 idx
                    expanded_sample["idx"] = base_idx * NUM_SEEDS_PER_PROMPT + seed  # 新的 idx
                    expanded_samples.append(expanded_sample)
            samples = expanded_samples
            print(f"扩展为 {len(samples)} 个样本（每个 prompt {NUM_SEEDS_PER_PROMPT} 个 seed）")
        
        # 检查断点续传
        skipped_count = 0
        for sample in samples:
            idx = sample.get("idx", 0)
            seed = sample.get("seed", None)
            # 对于多 seed 任务，检查特定 seed 的文件是否存在
            if is_multi_seed_task and seed is not None:
                image_file = task_output_dir / f"{sample['base_idx']:08d}_seed{seed:02d}.jpg"
                json_file = task_output_dir / f"{sample['base_idx']:08d}_seed{seed:02d}.json"
                if image_file.exists() and json_file.exists():
                    # 检查图像是否可读
                    try:
                        with Image.open(image_file) as img:
                            img.verify()
                        with Image.open(image_file) as img:
                            img.convert("RGB").load()
                        skipped_count += 1
                    except:
                        pass
            else:
                if check_sample_exists(task_output_dir, idx):
                    skipped_count += 1
        
        if skipped_count > 0:
            print(f"发现 {skipped_count} 个已生成的样本，将跳过")
        
        # 使用线程池并行处理样本
        print(f"开始并行处理 {len(samples)} 个样本（使用 {actual_num_workers} 个模型）...")
        
        # 将样本分配给不同的模型（轮询分配）
        samples_with_model = []
        for i, sample in enumerate(samples):
            model_idx = i % actual_num_workers
            samples_with_model.append((sample, model_idx))
        
        # 使用线程池执行
        success_count = 0
        error_count = 0
        
        with ThreadPoolExecutor(max_workers=actual_num_workers) as executor:
            futures = []
            for sample, model_idx in samples_with_model:
                pipeline_data = pipelines_data[model_idx]
                seed = sample.get("seed", None)  # 获取 seed
                future = executor.submit(
                    process_sample,
                    sample,
                    task_output_dir,
                    pipeline_data['pipeline'],
                    inference_hyper,
                    pipeline_data['gpu_id'],  # 使用GPU ID用于日志
                    seed=seed  # 传递 seed 参数
                )
                futures.append((future, sample.get("idx", 0)))
            
            # 等待所有线程任务完成并收集结果（只有全部完成后才判断是否写 .finish）
            for future, idx in futures:
                try:
                    success, error_msg = future.result()
                    if success:
                        if error_msg is None:
                            success_count += 1
                        else:
                            # 跳过的情况
                            success_count += 1
                    else:
                        error_count += 1
                        if error_msg:
                            print(f"样本 {idx}: {error_msg}")
                except Exception as e:
                    error_count += 1
                    print(f"样本 {idx} 处理异常: {e}")
        
        print(f"\n任务 {current_task} 处理完成:")
        print(f"  成功: {success_count + skipped_count} (其中 {skipped_count} 个跳过)")
        print(f"  失败: {error_count}")
        
        # 仅当所有线程都成功（无失败）时才写 .finish
        total_samples = len(samples)
        total_success = success_count + skipped_count
        all_success = (error_count == 0)
        
        # 对于 geneval 和 dpg，需要先创建 bench 目录，然后才创建 .finish 文件
        bench_created = True
        if all_success and current_task in ["geneval", "dpg"]:
            print(f"\n为 {current_task} 任务并行创建 bench 目录（使用 {actual_num_workers} 个线程）...")
            try:
                # 添加 inference_utils 路径
                inference_utils_dir = MACRO_DIR / "inference_utils"
                if str(inference_utils_dir) not in sys.path:
                    sys.path.insert(0, str(inference_utils_dir))
                from thirdparty_tasks import create_bench_directory_parallel
                
                # 使用线程池并行创建 bench 目录
                bench_results = []
                with ThreadPoolExecutor(max_workers=actual_num_workers) as executor:
                    futures = []
                    for worker_idx in range(actual_num_workers):
                        future = executor.submit(
                            create_bench_directory_parallel,
                            current_task,
                            task_output_dir,
                            process_index=worker_idx,
                            num_processes=actual_num_workers
                        )
                        futures.append(future)
                    
                    for future in futures:
                        try:
                            result = future.result()
                            bench_results.append(result)
                        except Exception as e:
                            print(f"Bench 目录创建线程出错: {e}")
                            bench_results.append(False)
                
                bench_created = all(bench_results)
                if bench_created:
                    print(f"Bench 目录创建完成")
                else:
                    print(f"警告: Bench 目录创建失败")
            except Exception as e:
                print(f"创建 bench 目录失败: {e}")
                import traceback
                traceback.print_exc()
                bench_created = False
        
        # 只有在所有线程都成功且（非 geneval/dpg 任务或 bench 目录创建成功）时才创建 .finish 文件
        if all_success and bench_created:
            finish_file = task_output_dir / ".finish"
            try:
                finish_file.touch()
                os.chmod(finish_file, 0o777)
                print(f"✓ 所有样本生成成功，已创建完成标记文件: {finish_file}")
            except Exception as e:
                print(f"警告: 创建完成标记文件失败: {e}")
        else:
            print(f"✗ 有样本生成失败，未创建完成标记文件")
            print(f"  总样本数: {total_samples}, 成功: {total_success}, 失败: {error_count}")
    
    print("\n所有任务处理完成！")
    
    # 返回退出码：0 表示成功，非0 表示失败
    # 检查是否有任何任务失败（通过检查 .finish 文件）
    has_failure = False
    for current_task in tasks_to_process:
        task_output_dir = Path(output_dir)
        finish_file = task_output_dir / ".finish"
        if not finish_file.exists():
            has_failure = True
            print(f"✗ 任务 {current_task} 未完成（缺少 .finish 文件）")
    
    return 0 if not has_failure else 1


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit模型推理脚本（支持多GPU并行和断点续传）")
    
    parser.add_argument("--model_configs", type=str, required=True, action='append',
                       help="模型配置，格式: model_id:origin_file_pattern (可以多个，每次使用 --model_configs 添加一个)")
    parser.add_argument("--processor_config", type=str, required=True,
                       help="处理器配置，格式: model_id:origin_file_pattern")
    parser.add_argument("--tokenizer_config", type=str, default=None,
                       help="Tokenizer 配置（可选），格式: model_id:origin_file_pattern")
    parser.add_argument("--task", type=str, default="all", 
                       choices=SUPPORTED_TASKS + ["all"],
                       help="任务类型")
    parser.add_argument("--use_refine_prompt", action="store_true", default=False,
                       help="For geneval task, whether to use refine prompt.")
    parser.add_argument("--image_num_category", type=str, default="all",
                       choices=IMAGE_NUM_CATEGORIES + ["all"],
                       help="图像数量类别")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="设备类型")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                       choices=["float32", "float16", "bfloat16"],
                       help="torch数据类型")
    parser.add_argument("--num_workers", type=int, default=1,
                       help="并行工作线程数量（总模型数量），默认1。模型会按顺序分配到所有可用GPU上")
    parser.add_argument("--model_base_path", type=str, default=None,
                       help="模型基础路径，如果指定，会设置 DIFFSYNTH_MODEL_BASE_PATH 环境变量并使用本地模型")
    parser.add_argument("--transformer_path", type=str, default=None,
                       help="训练后的 transformer 权重路径（全量微调，可选），如果指定，会加载训练后的权重到 pipe.dit")
    parser.add_argument("--lora_path", type=str, default=None,
                       help="LoRA 权重路径（LoRA 微调，可选），如果指定，会加载 LoRA 权重到 pipe.dit。与 --transformer_path 互斥")
    parser.add_argument("--data_root", type=str, default=None,
                       help="数据根目录（包含 filter/{task}/eval 结构），如果为None，使用默认路径")
    
    # 推理超参数
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子，确保可复现性")
    parser.add_argument("--num_inference_steps", type=int, default=40,
                       help="推理步数")
    parser.add_argument("--height", type=int, default=768,
                       help="输出图像高度")
    parser.add_argument("--width", type=int, default=768,
                       help="输出图像宽度")
    parser.add_argument("--edit_image_auto_resize", action="store_true", default=True,
                       help="自动调整输入图像尺寸")
    parser.add_argument("--zero_cond_t", action="store_true", default=True,
                       help="Qwen-Image-Edit-2511 的特殊参数")
    parser.add_argument("--max_input_pixels", type=str, default=None,
                       help="最大输入像素数，可以是 int 或 JSON 列表字符串，用于动态调整输入图像分辨率")
    
    args = parser.parse_args()
    
    # 解析模型配置
    model_configs = []
    for config_str in args.model_configs:
        parts = config_str.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"模型配置格式错误: {config_str}，应为 model_id:origin_file_pattern")
        model_configs.append({
            "model_id": parts[0],
            "origin_file_pattern": parts[1]
        })
    
    # 解析处理器配置
    parts = args.processor_config.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"处理器配置格式错误: {args.processor_config}，应为 model_id:origin_file_pattern")
    processor_config = {
        "model_id": parts[0],
        "origin_file_pattern": parts[1]
    }
    
    # 解析 tokenizer 配置（可选）
    tokenizer_config = None
    if args.tokenizer_config:
        parts = args.tokenizer_config.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Tokenizer 配置格式错误: {args.tokenizer_config}，应为 model_id:origin_file_pattern")
        tokenizer_config = {
            "model_id": parts[0],
            "origin_file_pattern": parts[1]
        }
    
    # 解析 max_input_pixels（可能是 JSON 字符串）
    max_input_pixels = None
    if args.max_input_pixels:
        try:
            import json
            max_input_pixels = json.loads(args.max_input_pixels)
        except json.JSONDecodeError:
            # 如果不是 JSON，尝试解析为 int
            try:
                max_input_pixels = int(args.max_input_pixels)
            except ValueError:
                raise ValueError(f"无法解析 max_input_pixels: {args.max_input_pixels}")
    
    # 默认动态分辨率
    if max_input_pixels is None:
        max_input_pixels = [1048576, 1048576, 589824, 589824, 589824, 262144, 262144, 262144, 262144, 262144]
    
    # 构建推理超参数
    inference_hyper = dict(
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        edit_image_auto_resize=args.edit_image_auto_resize,
        zero_cond_t=args.zero_cond_t,
    )
    
    # 如果指定了 max_input_pixels，添加到 inference_hyper
    if max_input_pixels is not None:
        inference_hyper['max_input_pixels'] = max_input_pixels
    
    # 运行推理
    exit_code = run_inference(
        model_configs=model_configs,
        processor_config=processor_config,
        task=args.task,
        image_num_category=args.image_num_category,
        output_dir=args.output_dir,
        device=args.device,
        torch_dtype=args.torch_dtype,
        inference_hyper=inference_hyper,
        num_workers=args.num_workers,
        use_refine_prompt=getattr(args, 'use_refine_prompt', False),
        model_base_path=getattr(args, 'model_base_path', None),
        transformer_path=getattr(args, 'transformer_path', None),
        tokenizer_config=tokenizer_config,
        lora_path=getattr(args, 'lora_path', None),
        data_root=getattr(args, 'data_root', None),
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
