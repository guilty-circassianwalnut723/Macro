#!/usr/bin/env python3
"""
Convert FSDP sharded model checkpoints to safetensors format.

Uses torch.distributed.checkpoint to directly load sharded weights.

Usage:
    python convert_fsdp_to_safetensors.py \
        --checkpoint_dir /path/to/checkpoint/0005000 \
        --output_file /path/to/checkpoint/0005000/ema.safetensors \
        --use_ema
"""

import os
import sys
import socket
import inspect
import argparse
import logging
from copy import deepcopy
from pathlib import Path

import torch

# Add bagel source path (parent directory contains bagel/source)
SCRIPT_DIR = Path(__file__).parent
BAGEL_DIR = SCRIPT_DIR.parent
BAGEL_SOURCE_PATH = BAGEL_DIR / "source"
if str(BAGEL_SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(BAGEL_SOURCE_PATH))

from safetensors.torch import save_file
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel,
    Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from data.data_utils import add_special_tokens

# Base model path (containing llm_config.json, vit_config.json, ae.safetensors)
BASE_MODEL_PATH = Path(os.environ.get("BAGEL_BASE_MODEL_PATH", "/path/to/BAGEL-7B-MoT"))


def setup_logger():
    """Set up logger."""
    logger = logging.getLogger("convert_fsdp_to_safetensors")
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[convert] %(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
    return logger


def _init_single_process_group():
    """Initialize single-process distributed group."""
    import torch.distributed as dist
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

    def _find_free_port(start_port: int = 29521, max_tries: int = 100) -> str:
        """Find an available port starting from start_port."""
        port = start_port
        for _ in range(max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                try:
                    s.bind(("127.0.0.1", port))
                    return str(port)
                except OSError:
                    port += 1
        raise RuntimeError(f"Cannot find available port (start: {start_port}, tries: {max_tries})")

    os.environ.setdefault("MASTER_PORT", _find_free_port(29521, 100))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


def _load_distcp_into_state_dict(target_sd: dict, ema_dir: str, logger: logging.Logger):
    """Load weights from FSDP sharded checkpoint into state_dict."""
    from torch.distributed.checkpoint import FileSystemReader
    reader = FileSystemReader(ema_dir)

    try:
        from torch.distributed.checkpoint import load as tdc_load
        params = set(inspect.signature(tdc_load).parameters.keys())
        if "no_dist" in params:
            logger.info("Using TDC load(..., no_dist=True).")
            tdc_load(target_sd, storage_reader=reader, no_dist=True)
            return
        else:
            logger.info("TDC load() without no_dist -> init single-process group.")
            _init_single_process_group()
            tdc_load(target_sd, storage_reader=reader)
            return
    except Exception as e:
        logger.warning(f"TDC load() failed with: {e}. Falling back to load_state_dict().")

    from torch.distributed.checkpoint import load_state_dict as tdc_load_state_dict
    _init_single_process_group()
    tdc_load_state_dict(target_sd, reader)


def build_bagel(base_model_path: Path, logger: logging.Logger):
    """Build Bagel model structure."""
    logger.info(f"Building model structure from base model path: {base_model_path}")

    # LLM config
    llm_config = Qwen2Config.from_json_file(base_model_path / "llm_config.json")
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    language_model = Qwen2ForCausalLM(llm_config)

    # ViT config
    vit_config = SiglipVisionConfig.from_json_file(base_model_path / "vit_config.json")
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vit_model = SiglipVisionModel(vit_config)

    # VAE config
    _, vae_config = load_ae(local_path=str(base_model_path / "ae.safetensors"))

    # Bagel config
    cfg = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        latent_patch_size=2,
        max_latent_size=64,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        interpolate_pos=False,
        timestep_shift=1.0,
    )

    model = Bagel(language_model, vit_model, cfg)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    return model


def prepare_tokenizer_and_resize(model: Bagel, base_model_path: Path, logger: logging.Logger):
    """Prepare tokenizer and resize model embeddings."""
    tokenizer = Qwen2Tokenizer.from_pretrained(str(base_model_path))
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)
        logger.info(f"Resized token embeddings to {len(tokenizer)} (added {num_new_tokens} special tokens).")

    return tokenizer, new_token_ids


def convert_fsdp_to_safetensors(
    checkpoint_dir: str,
    output_file: str,
    base_model_path: str = None,
    use_ema: bool = True,
    pop_pos_embeds: bool = False,
):
    """
    Convert FSDP sharded checkpoint to safetensors format.

    Args:
        checkpoint_dir: Checkpoint directory (containing ema or model subdirectory)
        output_file: Output safetensors file path
        base_model_path: Base model path (containing llm_config.json, vit_config.json, ae.safetensors).
                        If None, uses the default BASE_MODEL_PATH.
        use_ema: Whether to convert EMA model
        pop_pos_embeds: Whether to remove position embeddings
    """
    logger = setup_logger()

    checkpoint_dir = Path(checkpoint_dir)
    output_file = Path(output_file)

    # Determine base model path
    if base_model_path is None:
        base_model_path = BASE_MODEL_PATH
    else:
        base_model_path = Path(base_model_path)

    # Check if output file already exists
    if output_file.exists():
        logger.info(f"Output file already exists: {output_file}, skipping conversion")
        return

    # Determine input directory
    if use_ema:
        ema_dir = checkpoint_dir / "ema"
    else:
        ema_dir = checkpoint_dir / "model"

    if not ema_dir.is_dir():
        raise FileNotFoundError(f"Shard directory not found: {ema_dir}")

    logger.info(f"Loading weights from shard directory: {ema_dir}")
    logger.info(f"Output file: {output_file}")

    # Build model
    model = build_bagel(base_model_path, logger)
    model.eval()

    # Prepare tokenizer
    prepare_tokenizer_and_resize(model, base_model_path, logger)

    # Create target state_dict
    target_sd = deepcopy(model.state_dict())

    # Save initial weights (for checking which keys were not covered by checkpoint)
    initial_sd = deepcopy(model.state_dict())

    # Load weights from sharded checkpoint
    logger.info("Loading sharded weights...")
    _load_distcp_into_state_dict(target_sd, str(ema_dir), logger)

    # Check which keys used initial weights (not covered by checkpoint)
    logger.info("Checking weight sources...")
    keys_using_initial_weights = []
    for key in target_sd.keys():
        if key not in initial_sd:
            continue
        try:
            initial_val = initial_sd[key]
            current_val = target_sd[key]

            if torch.is_tensor(initial_val) and torch.is_tensor(current_val):
                if initial_val.shape == current_val.shape:
                    if torch.allclose(initial_val, current_val, atol=1e-6, rtol=1e-5):
                        keys_using_initial_weights.append(key)
            elif initial_val == current_val:
                keys_using_initial_weights.append(key)
        except Exception as e:
            logger.debug(f"Error comparing key {key}: {e}")
            pass

    if keys_using_initial_weights:
        logger.warning(f"Found {len(keys_using_initial_weights)} keys using initial weights (may not be covered by checkpoint):")
        for key in sorted(keys_using_initial_weights):
            logger.warning(f"  - {key}")
        logger.warning("If these keys should be trained, please check if the checkpoint is complete.")
    else:
        logger.info("All keys loaded from checkpoint (no initial weights used).")

    # Remove position embeddings if needed
    if pop_pos_embeds:
        for key in ["latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"]:
            if key in target_sd:
                target_sd.pop(key)
                logger.info(f"Removed {key}")

    # Convert to CPU dict and save
    logger.info("Saving as safetensors format...")
    cpu_state_dict = {}
    for key, value in target_sd.items():
        if hasattr(value, 'cpu'):
            cpu_value = value.cpu()
            if cpu_value.is_floating_point():
                cpu_value = cpu_value.to(torch.bfloat16)
            cpu_state_dict[key] = cpu_value
        else:
            cpu_state_dict[key] = value

    save_file(cpu_state_dict, str(output_file))
    logger.info(f"Conversion complete! Saved to: {output_file}")
    logger.info(f"Total {len(cpu_state_dict)} weight keys")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FSDP sharded model to safetensors format"
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Checkpoint directory path (containing ema or model subdirectory)")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output safetensors file path")
    parser.add_argument("--base_model_path", type=str, default=None,
                       help="Base model path (for model configs: llm_config.json, vit_config.json, ae.safetensors). "
                            "If None, uses BAGEL_BASE_MODEL_PATH env var or default path.")
    parser.add_argument("--use_ema", action="store_true", default=True,
                       help="Whether to convert EMA model (default True)")
    parser.add_argument("--use_model", action="store_true", default=False,
                       help="Whether to convert regular model (if set, use_ema is ignored)")
    parser.add_argument("--pop_pos_embeds", action="store_true", default=False,
                       help="Whether to remove position embeddings")

    args = parser.parse_args()

    use_ema = not args.use_model

    try:
        convert_fsdp_to_safetensors(
            checkpoint_dir=args.checkpoint_dir,
            output_file=args.output_file,
            base_model_path=args.base_model_path,
            use_ema=use_ema,
            pop_pos_embeds=args.pop_pos_embeds,
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
