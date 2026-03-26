#!/usr/bin/env python3
"""
OmniGen checkpoint 转换脚本

将训练的 checkpoint (model.bin) 转换为 HuggingFace 格式的 transformer 目录

使用方式:
    python convert_checkpoint.py --model_checkpoint_dir /path/to/checkpoint-{step}

该脚本会自动查找:
    - 配置文件: {exp_dir}/X2I2.yml
    - 模型文件: {checkpoint_dir}/model.bin
    - 输出目录: {checkpoint_dir}/transformer
"""

import dotenv
dotenv.load_dotenv(override=True)

import argparse
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

import torch
from accelerate import init_empty_weights

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

# 添加evaluation目录到路径
# 需要将包含evaluation的父目录添加到sys.path，而不是evaluation目录本身
SCRIPT_DIR = Path(__file__).parent.absolute()
MACRO_DIR = SCRIPT_DIR.parent.parent
macro_dir = str(MACRO_DIR)
if macro_dir not in sys.path:
    sys.path.insert(0, macro_dir)

# 添加 omnigen/source 目录到 sys.path，以便导入 omnigen2 模块
# 必须在导入 omnigen2 之前添加
source_dir = str(MACRO_DIR / "omnigen" / "source")
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)
    # 调试信息：确认路径已添加
    if Path(source_dir).exists():
        pass
    else:
        print(f"[WARNING] 路径不存在: {source_dir}")
        print(f"[DEBUG] MACRO_DIR: {MACRO_DIR}")
        print(f"[DEBUG] 检查 omnigen 目录: {MACRO_DIR / 'omnigen'}")
        print(f"[DEBUG] 检查 source 目录: {MACRO_DIR / 'omnigen' / 'source'}")

from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline


def main(args):
    checkpoint_dir = Path(args.model_checkpoint_dir)
    
    # 检查 checkpoint 目录是否存在
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint 目录不存在: {checkpoint_dir}")
    
    # 查找 model.bin
    model_path = checkpoint_dir / "model.bin"
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 查找配置文件 (向上查找 exp 目录)
    # checkpoint_dir 通常是 .../exp/checkpoint-{step}
    # 配置文件在 .../exp/X2I2.yml
    exp_dir = checkpoint_dir.parent
    config_path = exp_dir / "X2I2.yml"
    
    if not config_path.exists():
        # 如果不在 exp 目录下，尝试在 checkpoint_dir 的父目录的父目录查找
        exp_dir = checkpoint_dir.parent.parent
        config_path = exp_dir / "X2I2.yml"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {exp_dir}/X2I2.yml")
    
    # 输出目录
    save_path = checkpoint_dir / "transformer"
    
    print(f"转换配置:")
    print(f"  配置文件: {config_path}")
    print(f"  模型文件: {model_path}")
    print(f"  输出目录: {save_path}")
    
    # 如果 transformer 目录已存在，跳过转换
    if save_path.exists():
        print(f"Transformer 目录已存在: {save_path}，跳过转换")
        return
    
    # 加载配置
    conf = OmegaConf.load(config_path)
    arch_opt = conf.model.arch_opt
    
    # 首先加载 checkpoint 检查实际参数形状
    print(f"加载 checkpoint: {model_path}")
    state_dict = torch.load(model_path, mmap=True, weights_only=True)
    
    # 检查 image_index_embedding 形状并调整 arch_opt
    if "image_index_embedding" in state_dict:
        checkpoint_max_ref_images = state_dict["image_index_embedding"].shape[0]
        config_max_ref_images = arch_opt.get("max_ref_images", 5)
        if checkpoint_max_ref_images != config_max_ref_images:
            print(f"Warning: Checkpoint has max_ref_images={checkpoint_max_ref_images}, "
                  f"but config specifies {config_max_ref_images}. "
                  f"Using checkpoint value: {checkpoint_max_ref_images}")
            arch_opt.max_ref_images = checkpoint_max_ref_images
    
    arch_opt = OmegaConf.to_object(arch_opt)
    # 将列表转换为元组
    for key, value in arch_opt.items():
        if isinstance(value, list):
            arch_opt[key] = tuple(value)
    
    # 初始化空的 transformer
    print(f"初始化 transformer...")
    with init_empty_weights():
        transformer = OmniGen2Transformer2DModel(**arch_opt)
        
        # 如果使用 LoRA，添加适配器
        if conf.train.get('lora_ft', False):
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
            
            lora_config = LoraConfig(
                r=conf.train.lora_rank,
                lora_alpha=conf.train.lora_rank,
                lora_dropout=conf.train.lora_dropout,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            transformer.add_adapter(lora_config)
    
    # 加载状态字典
    print(f"加载状态字典...")
    missing, unexpect = transformer.load_state_dict(
        state_dict, assign=True, strict=False
    )
    if missing:
        print(f"缺失的参数: {missing}")
    if unexpect:
        print(f"意外的参数: {unexpect}")
    
    # 保存 transformer
    print(f"保存 transformer 到: {save_path}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    if conf.train.get('lora_ft', False):
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        OmniGen2Pipeline.save_lora_weights(
            save_directory=str(save_path),
            transformer_lora_layers=transformer_lora_layers,
        )
        print(f"已保存 LoRA 权重")
    else:
        transformer.save_pretrained(str(save_path))
        print(f"已保存完整 transformer")
    
    print(f"转换完成!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 OmniGen checkpoint 转换为 HuggingFace 格式"
    )
    parser.add_argument(
        "--model_checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint 目录路径 (例如: .../exp/checkpoint-5000)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
