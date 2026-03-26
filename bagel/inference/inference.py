#!/usr/bin/env python3
"""
推理脚本

从filter目录加载数据，使用Bagel模型进行推理，并保存结果
支持多GPU并行推理和断点续传
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import random
import numpy as np

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

# 添加bagel source路径
SCRIPT_DIR = Path(__file__).parent
BAGEL_DIR = SCRIPT_DIR.parent
MACRO_DIR = BAGEL_DIR.parent
BAGEL_SOURCE_PATH = BAGEL_DIR / "source"
if str(BAGEL_SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(BAGEL_SOURCE_PATH))

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer

# 导入本地utils
from utils import (
    load_data_for_task,  # 统一的数据加载接口
    SUPPORTED_TASKS,
    IMAGE_NUM_CATEGORIES,
    check_sample_exists,
    save_sample
)

# 固定的基础模型路径（包含llm_config.json, vit_config.json, ae.safetensors等）
BASE_MODEL_PATH = Path(os.environ.get("BAGEL_BASE_MODEL_PATH", "/path/to/BAGEL-7B-MoT"))


def load_model(
    model_path: str,
    base_model_path: Optional[str] = None,
    vae_transform_size: tuple = (1024, 512),
    vit_transform_size: tuple = (980, 224),
    max_mem_per_gpu: str = "40GiB",
    offload_folder: str = "/tmp/offload",
    device_id: Optional[int] = None,
    max_input_pixels = None,  # 可以是 int 或 list，用于动态调整输入图像分辨率
):
    """
    加载Bagel模型到指定GPU（不分片）

    Args:
        model_path: 包含ema.safetensors的模型路径（可以是目录或safetensors文件路径）
        base_model_path: 基础模型路径（包含llm_config.json, vit_config.json, ae.safetensors），
                        如果为None，使用默认的BASE_MODEL_PATH
        vae_transform_size: VAE transform尺寸 (height, width)
        vit_transform_size: ViT transform尺寸 (height, width)
        max_mem_per_gpu: 每个GPU的最大内存
        offload_folder: offload文件夹路径
        device_id: 指定的GPU设备ID（0-based），如果为None，使用accelerate自动分配
        max_input_pixels: 最大输入像素数，可以是 int 或 list，用于动态调整输入图像分辨率

    Returns:
        (model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)
    """
    model_path = Path(model_path)

    # 确定基础模型路径
    if base_model_path is None:
        base_model_path = BASE_MODEL_PATH
    else:
        base_model_path = Path(base_model_path)

    # 确定ema.safetensors路径
    if model_path.suffix == ".safetensors":
        ema_checkpoint_path = model_path
    else:
        # 如果是目录，查找ema.safetensors
        ema_checkpoint_path = model_path / "ema.safetensors"
        if not ema_checkpoint_path.exists():
            # 如果不存在，假设整个目录就是模型目录
            ema_checkpoint_path = model_path

    # LLM config preparing (从基础模型路径加载)
    llm_config = Qwen2Config.from_json_file(base_model_path / "llm_config.json")
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config preparing (从基础模型路径加载)
    vit_config = SiglipVisionConfig.from_json_file(base_model_path / "vit_config.json")
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE loading (从基础模型路径加载)
    vae_model, vae_config = load_ae(local_path=str(base_model_path / "ae.safetensors"))

    # Bagel config preparing
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # Tokenizer Preparing (从基础模型路径加载)
    tokenizer = Qwen2Tokenizer.from_pretrained(str(base_model_path))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Image Transform Preparing
    # 如果指定了 max_input_pixels，传递给 ImageTransform
    vae_transform = ImageTransform(vae_transform_size[0], vae_transform_size[1], 16, max_pixels=max_input_pixels)
    vit_transform = ImageTransform(vit_transform_size[0], vit_transform_size[1], 14, max_pixels=max_input_pixels)

    # Device map
    if device_id is not None:
        # 指定GPU设备，将所有模块放在同一个GPU上（不分片）
        device_map = {"": f"cuda:{device_id}"}
    else:
        # 自动分配GPU（原来的逻辑）
        device_map = infer_auto_device_map(
            model,
            max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )

        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]

        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

    # Load checkpoint (从model_path加载ema.safetensors)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(ema_checkpoint_path),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder=offload_folder
    )

    model = model.eval()

    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


def process_sample(
    sample: Dict[str, Any],
    output_dir: Path,
    inferencer: InterleaveInferencer,
    inference_hyper: Dict[str, Any],
    gpu_id: int,
    lock: threading.Lock,
    seed: Optional[int] = None
) -> Tuple[bool, Optional[str]]:
    """
    处理单个样本（线程安全）

    Args:
        sample: 样本数据
        output_dir: 输出目录
        inferencer: 推理器（使用线程锁保护）
        inference_hyper: 推理超参数
        gpu_id: GPU ID（用于日志）
        lock: 线程锁

    Returns:
        (成功标志, 错误信息)
    """
    try:
        idx = sample.get("idx", 0)
        task = sample.get("task", "")

        # 断点续传检查
        if check_sample_exists(output_dir, idx):
            return (True, None)

        prompt = sample.get("prompt", "")
        input_images = sample.get("input_images", [])
        output_image_path = sample.get("output_image", "")

        # 加载输入图像
        images = []
        for img_item in input_images:
            if isinstance(img_item, (str, Path)):
                img_path = str(img_item)
                if os.path.exists(img_path):
                    images.append(Image.open(img_path).convert("RGB"))
                else:
                    print(f"[GPU {gpu_id}] 警告: 图像文件不存在: {img_path}")
            elif isinstance(img_item, Image.Image):
                images.append(img_item.convert("RGB"))
            else:
                print(f"[GPU {gpu_id}] 警告: 不支持的图像类型: {type(img_item)}")

        if len(images) == 0:
            return (False, f"跳过样本 {idx}: 没有有效的输入图像")

        # 使用线程锁保护推理
        with lock:
            # 如果 inference_hyper 中指定了 image_shapes，使用指定的尺寸；否则使用最后一张输入图像的尺寸
            output_dict = inferencer(image=images, text=prompt, **inference_hyper)
            generated_image = output_dict['image']

        # 保存图像
        output_image_file = output_dir / f"{idx:08d}.jpg"
        generated_image.save(output_image_file)
        # 设置文件权限为777（所有人可读写执行）
        os.chmod(output_image_file, 0o777)

        # 使用统一的保存接口
        save_sample(
            output_dir=output_dir,
            idx=idx,
            sample=sample,
            output_image_path=str(output_image_file),
            target_image_path=output_image_path,
        )

        return (True, None)

    except Exception as e:
        sample_idx = sample.get("idx", "unknown")
        error_msg = f"处理样本 {sample_idx} 时出错: {e}"
        import traceback
        traceback.print_exc()
        return (False, error_msg)


def run_inference(
    model_path: str,
    task: str,
    image_num_category: str = "all",
    output_dir: str = "./outputs",
    base_model_path: Optional[str] = None,
    vae_transform: Optional[tuple] = None,
    vit_transform: Optional[tuple] = None,
    inference_hyper: Optional[Dict[str, Any]] = None,
    max_mem_per_gpu: str = "40GiB",
    offload_folder: str = "/tmp/offload",
    num_workers: int = 8,
    max_input_pixels = None,  # 可以是 int 或 list，用于动态调整输入图像分辨率
    data_root: Optional[str] = None,  # 数据根目录，如果为None，使用默认路径
) -> int:
    # 默认transform尺寸
    if vae_transform is None:
        vae_transform = (1024, 512)
    if vit_transform is None:
        vit_transform = (980, 224)

    # 默认推理超参数
    if inference_hyper is None:
        inference_hyper = dict(
            seed=42,
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            cfg_interval=[0.0, 1.0],
            timestep_shift=3.0,
            num_timesteps=50,
            cfg_renorm_min=0.0,
            cfg_renorm_type="text_channel",
        )

    # 设置随机种子以确保可复现性
    seed = inference_hyper.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"设置随机种子: {seed}")

    # 从inference_hyper中移除seed（如果存在），因为已经设置了全局种子
    inference_hyper = {k: v for k, v in inference_hyper.items() if k != 'seed'}

    # 确定可用的GPU数量
    available_gpus = torch.cuda.device_count()

    if available_gpus <= 0:
        raise RuntimeError("没有可用的GPU")

    actual_num_workers = num_workers

    print(f"可用GPU数量: {available_gpus}")
    print(f"将加载 {actual_num_workers} 个模型，按顺序分配到所有GPU上")

    # 一次性加载所有模型到GPU（按顺序轮询分配到所有GPU）
    print(f"正在加载 {actual_num_workers} 个模型...")
    models_data = []
    for model_idx in range(actual_num_workers):
        gpu_id = model_idx % available_gpus

        print(f"加载模型 {model_idx + 1}/{actual_num_workers} 到 GPU {gpu_id}...")
        model, vae_model, tokenizer, vae_tf, vit_tf, new_token_ids = load_model(
            model_path=model_path,
            base_model_path=base_model_path,
            vae_transform_size=vae_transform,
            vit_transform_size=vit_transform,
            max_mem_per_gpu=max_mem_per_gpu,
            offload_folder=f"{offload_folder}_{gpu_id}_{model_idx}",
            device_id=gpu_id,
            max_input_pixels=max_input_pixels,
        )

        inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_tf,
            vit_transform=vit_tf,
            new_token_ids=new_token_ids
        )

        models_data.append({
            'model_idx': model_idx,
            'gpu_id': gpu_id,
            'inferencer': inferencer,
            'lock': threading.Lock(),
        })
        print(f"模型 {model_idx + 1} 加载完成（GPU {gpu_id}）")

    print("所有模型加载完成！")

    # 确定要处理的tasks
    if task == "all":
        tasks_to_process = SUPPORTED_TASKS
    else:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}. Supported tasks: {SUPPORTED_TASKS}")
        tasks_to_process = [task]

    # 处理每个task
    for current_task in tasks_to_process:
        print(f"\n处理任务: {current_task}")
        print(f"图像数量类别: {image_num_category}")

        # 使用统一的数据加载接口
        try:
            samples = load_data_for_task(
                task=current_task,
                image_num_category=image_num_category if image_num_category != "all" else None,
                data_root=Path(data_root) if data_root else None,
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
        task_output_dir = Path(output_dir)
        task_output_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(task_output_dir, 0o777)

        # 检查断点续传
        skipped_count = 0
        for sample in samples:
            idx = sample.get("idx", 0)
            if check_sample_exists(task_output_dir, idx):
                skipped_count += 1

        if skipped_count > 0:
            print(f"发现 {skipped_count} 个已生成的样本，将跳过")

        print(f"开始并行处理 {len(samples)} 个样本（使用 {actual_num_workers} 个模型）...")

        samples_with_model = []
        for i, sample in enumerate(samples):
            model_idx = i % actual_num_workers
            samples_with_model.append((sample, model_idx))

        success_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=actual_num_workers) as executor:
            futures = []
            for sample, model_idx in samples_with_model:
                model_data = models_data[model_idx]
                future = executor.submit(
                    process_sample,
                    sample,
                    task_output_dir,
                    model_data['inferencer'],
                    inference_hyper,
                    model_data['gpu_id'],
                    model_data['lock'],
                )
                futures.append((future, sample.get("idx", 0)))

            for future, idx in futures:
                try:
                    success, error_msg = future.result()
                    if success:
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

        all_success = (error_count == 0)

        if all_success:
            finish_file = task_output_dir / ".finish"
            try:
                finish_file.touch()
                os.chmod(finish_file, 0o777)
                print(f"✓ 所有样本生成成功，已创建完成标记文件: {finish_file}")
            except Exception as e:
                print(f"警告: 创建完成标记文件失败: {e}")
        else:
            total_success = success_count + skipped_count
            print(f"✗ 有样本生成失败，未创建完成标记文件")
            print(f"  总样本数: {len(samples)}, 成功: {total_success}, 失败: {error_count}")

    print("\n所有任务处理完成！")

    has_failure = False
    for current_task in tasks_to_process:
        task_output_dir = Path(output_dir)
        finish_file = task_output_dir / ".finish"
        if not finish_file.exists():
            has_failure = True
            break

    return 0 if not has_failure else 1


def main():
    parser = argparse.ArgumentParser(description="Bagel模型推理脚本（支持多GPU并行和断点续传）")

    parser.add_argument("--model_path", type=str, required=True,
                       help="包含ema.safetensors的模型路径（可以是目录或safetensors文件路径）")
    parser.add_argument("--base_model_path", type=str, default=None,
                       help="基础模型路径（包含llm_config.json, vit_config.json, ae.safetensors），如果为None，使用默认路径")
    parser.add_argument("--data_root", type=str, default=None,
                       help="数据根目录（包含 filter/{task}/eval 结构），如果为None，使用默认路径")
    parser.add_argument("--task", type=str, default="all",
                       choices=SUPPORTED_TASKS + ["all"],
                       help="任务类型")
    parser.add_argument("--image_num_category", type=str, default="all",
                       choices=IMAGE_NUM_CATEGORIES + ["all"],
                       help="图像数量类别")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="输出目录")
    parser.add_argument("--vae_transform", type=str, default="1024,512",
                       help="VAE transform尺寸，格式: height,width")
    parser.add_argument("--vit_transform", type=str, default="980,224",
                       help="ViT transform尺寸，格式: height,width")
    parser.add_argument("--max_mem_per_gpu", type=str, default="40GiB",
                       help="每个GPU的最大内存")
    parser.add_argument("--offload_folder", type=str, default="/tmp/offload",
                       help="offload文件夹路径")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="并行工作线程数量（总模型数量），默认8。模型会按顺序分配到所有可用GPU上")

    # 推理超参数
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--cfg_text_scale", type=float, default=4.0,
                       help="CFG text scale")
    parser.add_argument("--cfg_img_scale", type=float, default=2.0,
                       help="CFG image scale")
    parser.add_argument("--cfg_interval", type=str, default="0.0,1.0",
                       help="CFG interval，格式: start,end")
    parser.add_argument("--timestep_shift", type=float, default=3.0,
                       help="Timestep shift")
    parser.add_argument("--num_timesteps", type=int, default=50,
                       help="Number of timesteps")
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0,
                       help="CFG renorm min")
    parser.add_argument("--cfg_renorm_type", type=str, default="text_channel",
                       choices=["global", "text_channel"],
                       help="CFG renorm type")
    parser.add_argument("--max_input_pixels", type=str, default=None,
                       help="最大输入像素数，可以是 int 或 JSON 列表字符串，用于动态调整输入图像分辨率")
    parser.add_argument("--image_shapes", type=str, default=None,
                       help="生成图像的尺寸，格式: height,width (例如: 768,768)。如果未指定，则使用最后一张输入图像的尺寸")

    args = parser.parse_args()

    # 解析transform尺寸
    vae_transform = tuple(map(int, args.vae_transform.split(",")))
    vit_transform = tuple(map(int, args.vit_transform.split(",")))

    # 解析 max_input_pixels（可能是 JSON 字符串）
    max_input_pixels = None
    if args.max_input_pixels:
        try:
            max_input_pixels = json.loads(args.max_input_pixels)
        except json.JSONDecodeError:
            try:
                max_input_pixels = int(args.max_input_pixels)
            except ValueError:
                raise ValueError(f"无法解析 max_input_pixels: {args.max_input_pixels}")

    # 默认动态分辨率
    if max_input_pixels is None:
        max_input_pixels = [1048576, 1048576, 589824, 589824, 589824, 262144, 262144, 262144, 262144, 262144]

    # 解析CFG interval
    cfg_interval = list(map(float, args.cfg_interval.split(",")))

    # 解析 image_shapes（如果指定）
    image_shapes = None
    if args.image_shapes:
        try:
            height, width = map(int, args.image_shapes.split(","))
            image_shapes = (height, width)
        except ValueError:
            raise ValueError(f"无法解析 image_shapes: {args.image_shapes}，格式应为 height,width (例如: 768,768)")

    # 构建推理超参数
    inference_hyper = dict(
        seed=args.seed,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=cfg_interval,
        timestep_shift=args.timestep_shift,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=args.cfg_renorm_min,
        cfg_renorm_type=args.cfg_renorm_type,
    )

    # 如果指定了 image_shapes，添加到 inference_hyper
    if image_shapes is not None:
        inference_hyper['image_shapes'] = image_shapes

    # 运行推理
    exit_code = run_inference(
        model_path=args.model_path,
        task=args.task,
        image_num_category=args.image_num_category,
        output_dir=args.output_dir,
        base_model_path=args.base_model_path,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        inference_hyper=inference_hyper,
        max_mem_per_gpu=args.max_mem_per_gpu,
        offload_folder=args.offload_folder,
        num_workers=args.num_workers,
        max_input_pixels=max_input_pixels,
        data_root=args.data_root,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
