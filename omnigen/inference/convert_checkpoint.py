#!/usr/bin/env python3
"""
OmniGen checkpoint conversion script

Converts a trained checkpoint (model.bin) to a HuggingFace format transformer directory

Usage:
    python convert_checkpoint.py --model_checkpoint_dir /path/to/checkpoint-{step}

This script automatically finds:
    - Config file: {exp_dir}/X2I2.yml
    - Model file: {checkpoint_dir}/model.bin
    - Output directory: {checkpoint_dir}/transformer
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

# Add evaluation directory to path
# Need to add the parent directory containing evaluation to sys.path, not the evaluation directory itself
SCRIPT_DIR = Path(__file__).parent.absolute()
MACRO_DIR = SCRIPT_DIR.parent.parent
macro_dir = str(MACRO_DIR)
if macro_dir not in sys.path:
    sys.path.insert(0, macro_dir)

# Add omnigen/source directory to sys.path for importing omnigen2 module
# Must be added before importing omnigen2
source_dir = str(MACRO_DIR / "omnigen" / "source")
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)
    # Debug info: confirm path has been added
    if Path(source_dir).exists():
        pass
    else:
        print(f"[WARNING] Path does not exist: {source_dir}")
        print(f"[DEBUG] MACRO_DIR: {MACRO_DIR}")
        print(f"[DEBUG] Checking omnigen directory: {MACRO_DIR / 'omnigen'}")
        print(f"[DEBUG] Checking source directory: {MACRO_DIR / 'omnigen' / 'source'}")

from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline


def main(args):
    checkpoint_dir = Path(args.model_checkpoint_dir)

    # Check if checkpoint directory exists
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    # Find model.bin
    model_path = checkpoint_dir / "model.bin"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    # Find config file (search up to exp directory)
    # checkpoint_dir is typically .../exp/checkpoint-{step}
    # Config file is at .../exp/X2I2.yml
    exp_dir = checkpoint_dir.parent
    config_path = exp_dir / "X2I2.yml"

    if not config_path.exists():
        # If not in exp directory, try searching in parent's parent directory
        exp_dir = checkpoint_dir.parent.parent
        config_path = exp_dir / "X2I2.yml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file does not exist: {exp_dir}/X2I2.yml")

    # Output directory
    save_path = checkpoint_dir / "transformer"

    print(f"Conversion configuration:")
    print(f"  Config file: {config_path}")
    print(f"  Model file: {model_path}")
    print(f"  Output directory: {save_path}")

    # If transformer directory already exists, skip conversion
    if save_path.exists():
        print(f"Transformer directory already exists: {save_path}, skipping conversion")
        return

    # Load configuration
    conf = OmegaConf.load(config_path)
    arch_opt = conf.model.arch_opt

    # Load checkpoint first to check actual parameter shapes
    print(f"Loading checkpoint: {model_path}")
    state_dict = torch.load(model_path, mmap=True, weights_only=True)

    # Check image_index_embedding shape and adjust arch_opt
    if "image_index_embedding" in state_dict:
        checkpoint_max_ref_images = state_dict["image_index_embedding"].shape[0]
        config_max_ref_images = arch_opt.get("max_ref_images", 5)
        if checkpoint_max_ref_images != config_max_ref_images:
            print(f"Warning: Checkpoint has max_ref_images={checkpoint_max_ref_images}, "
                  f"but config specifies {config_max_ref_images}. "
                  f"Using checkpoint value: {checkpoint_max_ref_images}")
            arch_opt.max_ref_images = checkpoint_max_ref_images

    arch_opt = OmegaConf.to_object(arch_opt)
    # Convert lists to tuples
    for key, value in arch_opt.items():
        if isinstance(value, list):
            arch_opt[key] = tuple(value)

    # Initialize empty transformer
    print(f"Initializing transformer...")
    with init_empty_weights():
        transformer = OmniGen2Transformer2DModel(**arch_opt)

        # If using LoRA, add adapter
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

    # Load state dictionary
    print(f"Loading state dictionary...")
    missing, unexpect = transformer.load_state_dict(
        state_dict, assign=True, strict=False
    )
    if missing:
        print(f"Missing parameters: {missing}")
    if unexpect:
        print(f"Unexpected parameters: {unexpect}")

    # Save transformer
    print(f"Saving transformer to: {save_path}")
    save_path.mkdir(parents=True, exist_ok=True)

    if conf.train.get('lora_ft', False):
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        OmniGen2Pipeline.save_lora_weights(
            save_directory=str(save_path),
            transformer_lora_layers=transformer_lora_layers,
        )
        print(f"LoRA weights saved")
    else:
        transformer.save_pretrained(str(save_path))
        print(f"Full transformer saved")

    print(f"Conversion complete!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert OmniGen checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--model_checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint directory path (e.g., .../exp/checkpoint-5000)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
