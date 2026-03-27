#!/usr/bin/env python3
"""
Bagel configuration processing script - reads config.yaml and generates files needed for training.

Preserves the original training configuration format:
- dataset_config.yaml: training data configuration
- merged_parquet_info.json: merged parquet info
- run.sh / run_local.sh: training launch scripts

Removes hope-related logic; keeps only local execution (supports multi-node multi-GPU).

Usage:
    python process_config.py --exp_name <experiment_name>
    python process_config.py --list
    python process_config.py --all
"""

import os
import sys
import json
import argparse
import yaml
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# Directory configuration (using relative paths)
SCRIPT_DIR = Path(__file__).parent.resolve()
MACRO_DIR = SCRIPT_DIR.parent
CONFIG_FILE = SCRIPT_DIR / "config.yaml"
SOURCE_DIR = SCRIPT_DIR / "source"
EXPS_DIR = SCRIPT_DIR / "exps"
DATA_DIR = MACRO_DIR / "data"
CKPTS_DIR = MACRO_DIR / "ckpts"


def load_config() -> dict:
    """Load configuration file."""
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def check_t2i_data_prepared(data_dir: Path) -> bool:
    """Check whether T2I data has been prepared (Parquet format)."""
    parquet_info_path = data_dir / "parquet_info.json"
    if not parquet_info_path.exists():
        return False
    try:
        with open(parquet_info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        has_file_keys = any(k.endswith('.parquet') for k in info.keys())
        return has_file_keys
    except Exception:
        return False


def prepare_t2i_data(t2i_jsonl_path: str, output_dir: Path,
                     max_records_per_file: int = 1000) -> Tuple[Optional[Path], int, int]:
    """Prepare T2I data: convert from JSONL to Parquet format."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("Error: pyarrow is required: pip install pyarrow")
        return None, 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already prepared
    parquet_info_path = output_dir / "parquet_info.json"
    if parquet_info_path.exists():
        with open(parquet_info_path, 'r', encoding='utf-8') as f:
            existing_info = json.load(f)
        has_file_keys = any(k.endswith('.parquet') for k in existing_info.keys())
        if has_file_keys:
            print(f"T2I data already exists, skipping preparation: {output_dir}")
            return output_dir, existing_info.get('original_count', 0), existing_info.get('num_total_samples', 0)
        else:
            print("T2I parquet_info.json format is invalid, regenerating")
            for old_file in output_dir.glob("t2i_*.parquet"):
                old_file.unlink()

    print(f"Preparing T2I data: {t2i_jsonl_path} -> {output_dir}")
    
    t2i_path = Path(t2i_jsonl_path)
    if not t2i_path.exists():
        # Try path relative to MACRO_DIR
        t2i_path = MACRO_DIR / t2i_jsonl_path

    if not t2i_path.exists():
        print(f"Warning: T2I JSONL file not found: {t2i_jsonl_path}")
        return None, 0, 0
    
    data_list = []
    original_count = 0
    
    with open(t2i_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            original_count += 1
            
            try:
                data = json.loads(line)
                # Support two formats
                if 'messages' in data:
                    messages = data.get('messages', [])
                    instruction = ""
                    output_image_path = ""
                    for msg in messages:
                        if msg['role'] == 'user':
                            instruction = msg['content']
                        elif msg['role'] == 'assistant':
                            content = msg['content']
                            if '<img_start>' in content and '<img_end>' in content:
                                start_idx = content.find('>') + 1
                                end_idx = content.find('<img_end>')
                                if start_idx < end_idx:
                                    output_image_path = content[start_idx:end_idx]
                                    if output_image_path.startswith('<fixres_'):
                                        fixres_end = output_image_path.find('>')
                                        if fixres_end != -1:
                                            output_image_path = output_image_path[fixres_end + 1:]
                else:
                    instruction = data.get('instruction', data.get('prompt', ''))
                    output_image_path = data.get('output_image', data.get('image', ''))
                
                if instruction and output_image_path:
                    # Convert to absolute path
                    if not os.path.isabs(output_image_path):
                        output_image_path = str(MACRO_DIR / output_image_path)
                        
                    captions_dict = {'caption': instruction}
                    data_list.append({
                        'image_path': output_image_path,
                        'captions': json.dumps(captions_dict, ensure_ascii=False)
                    })
            except Exception:
                continue
    
    converted_count = len(data_list)
    print(f"  Processing complete: {original_count} original rows, {converted_count} valid records")

    if not data_list:
        return None, original_count, converted_count

    # Write Parquet files
    parquet_files = []
    parquet_file_info = {}
    num_files = (len(data_list) + max_records_per_file - 1) // max_records_per_file
    
    for file_idx in range(num_files):
        start_idx = file_idx * max_records_per_file
        end_idx = min(start_idx + max_records_per_file, len(data_list))
        batch_data = data_list[start_idx:end_idx]
        
        parquet_filename = f"t2i_{file_idx:04d}.parquet"
        parquet_filepath = output_dir / parquet_filename
        parquet_filepath_str = str(parquet_filepath)
        
        schema = pa.schema([
            pa.field('image_path', pa.string()),
            pa.field('captions', pa.string()),
        ])
        
        table = pa.Table.from_arrays(
            [[d['image_path'] for d in batch_data],
             [d['captions'] for d in batch_data]],
            schema=schema
        )
        pq.write_table(table, parquet_filepath, row_group_size=1000)
        parquet_files.append(parquet_filepath_str)
        
        pf_out = pq.ParquetFile(parquet_filepath)
        parquet_file_info[parquet_filepath_str] = {
            'num_row_groups': pf_out.num_row_groups,
            'num_rows': pf_out.metadata.num_rows
        }
    
    parquet_info = {
        'num_files': num_files,
        'num_total_samples': converted_count,
        'original_count': original_count,
        'parquet_files': parquet_files
    }
    parquet_info.update(parquet_file_info)
    
    with open(output_dir / 'parquet_info.json', 'w', encoding='utf-8') as f:
        json.dump(parquet_info, f, indent=2, ensure_ascii=False)
    
    print(f"T2I data preparation complete: {num_files} files, {converted_count} records")
    return output_dir, original_count, converted_count


def adjust_data_to_target_num(data_list: List[dict], target_num: int) -> List[dict]:
    """Adjust data list to the target count."""
    if not data_list:
        return []
    
    original_count = len(data_list)
    
    if target_num <= 0:
        return data_list
    
    if target_num <= original_count:
        return data_list[:target_num]
    else:
        result = []
        while len(result) < target_num:
            result.extend(data_list)
        return result[:target_num]


def convert_multiref_to_parquet(json_dir: Path, output_dir: Path,
                                 target_num: int = 0,
                                 max_records_per_file: int = 1000,
                                 force_regenerate: bool = False) -> Tuple[int, int, int]:
    """Convert MultiRef JSON data to Parquet format."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("Error: pyarrow is required")
        return 0, 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already converted
    parquet_info_path = output_dir / 'parquet_info.json'
    
    if not force_regenerate and parquet_info_path.exists():
        with open(parquet_info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        old_target_num = info.get('target_num', 0)
        has_file_keys = any(k.endswith('.parquet') for k in info.keys())
        if old_target_num == target_num and has_file_keys:
            num_files = info.get('num_files', 0)
            num_samples = info.get('num_total_samples', 0)
            original_count = info.get('original_samples', num_samples)
            print(f"    Parquet data already exists: {num_files} files, {num_samples} samples")
            return num_files, original_count, num_samples
        else:
            if not has_file_keys:
                print("    parquet_info.json format is invalid, regenerating")
            else:
                print(f"    target_num changed ({old_target_num} -> {target_num}), regenerating")
            for old_file in output_dir.glob("chunk_*.parquet"):
                old_file.unlink()
    
    if not json_dir.exists():
        print(f"    Warning: JSON directory not found: {json_dir}")
        return 0, 0, 0

    # Read JSON files
    raw_data_list = []
    json_files = sorted(json_dir.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            input_images = data.get('input_images', [])
            output_image = data.get('output_image', '')
            instruction = data.get('prompt', data.get('instruction', ''))
            
            if not instruction:
                continue
            
            image_list = []
            for img_path in input_images:
                if not os.path.isabs(img_path):
                    img_path = str(MACRO_DIR / img_path)
                image_list.append(img_path)
                
            if output_image:
                if not os.path.isabs(output_image):
                    output_image = str(MACRO_DIR / output_image)
                image_list.append(output_image)
            
            instruction_list = [instruction]
            
            raw_data_list.append({
                'image_list': image_list,
                'instruction_list': instruction_list
            })
        except Exception:
            continue
    
    if not raw_data_list:
        return 0, 0, 0
    
    original_count = len(raw_data_list)
    
    # Adjust data count
    data_list = adjust_data_to_target_num(raw_data_list, target_num)
    converted_count = len(data_list)

    if converted_count == 0:
        return 0, original_count, 0

    # Write Parquet
    num_files = (converted_count + max_records_per_file - 1) // max_records_per_file
    parquet_files = []
    parquet_file_info = {}
    
    for file_idx in range(num_files):
        start_idx = file_idx * max_records_per_file
        end_idx = min(start_idx + max_records_per_file, converted_count)
        batch_data = data_list[start_idx:end_idx]
        
        parquet_filename = f"chunk_{file_idx:04d}.parquet"
        parquet_filepath = output_dir / parquet_filename
        parquet_filepath_str = str(parquet_filepath)
        
        schema = pa.schema([
            pa.field('image_list', pa.list_(pa.string())),
            pa.field('instruction_list', pa.list_(pa.string())),
        ])
        
        image_list_col = [d['image_list'] for d in batch_data]
        instruction_list_col = [d['instruction_list'] for d in batch_data]
        
        table = pa.Table.from_arrays(
            [image_list_col, instruction_list_col],
            schema=schema
        )
        pq.write_table(table, parquet_filepath, row_group_size=100)
        parquet_files.append(parquet_filepath_str)
        
        pf_out = pq.ParquetFile(parquet_filepath)
        parquet_file_info[parquet_filepath_str] = {
            'num_row_groups': pf_out.num_row_groups,
            'num_rows': pf_out.metadata.num_rows
        }
    
    parquet_info = {
        'num_files': num_files,
        'num_total_samples': converted_count,
        'original_samples': original_count,
        'target_num': target_num,
        'parquet_files': parquet_files
    }
    parquet_info.update(parquet_file_info)
    
    with open(parquet_info_path, 'w', encoding='utf-8') as f:
        json.dump(parquet_info, f, indent=2, ensure_ascii=False)
    
    print(f"    Conversion complete: {original_count} -> {converted_count} samples, {num_files} files")
    return num_files, original_count, converted_count


def prepare_multiref_data(multiref_data_root: Path, data_config: Dict,
                          parquet_dir: Path,
                          force_regenerate: bool = False) -> Tuple[List[Dict], List[Dict]]:
    """Prepare multi-reference image data."""
    multi_ref_data_dirs = []
    statistics_list = []
    
    for task, categories in data_config.items():
        for category, cat_config in categories.items():
            src_json_dir = multiref_data_root / task / "train" / category
            dest_parquet_dir = parquet_dir / task / category
            
            target_num = cat_config.get('data_num', 0)
            
            print(f"  Processing: {task}/{category}, target count: {target_num}")
            
            num_files, original_count, converted_count = convert_multiref_to_parquet(
                src_json_dir, dest_parquet_dir,
                target_num=target_num,
                force_regenerate=force_regenerate
            )
            
            if num_files > 0:
                multi_ref_data_dirs.append({
                    'path': str(dest_parquet_dir),
                    'task': task,
                    'category': category,
                    'num_files': num_files,
                    'num_samples': converted_count
                })
                statistics_list.append({
                    'task': task,
                    'category': category,
                    'original_count': original_count,
                    'converted_count': converted_count
                })
    
    return multi_ref_data_dirs, statistics_list


def create_dataset_config_yaml(exp_name: str, exp_config: dict, global_config: dict,
                                exp_dir: Path, t2i_data_dir: Optional[Path],
                                multi_ref_data_dirs: List[Dict]) -> Path:
    """Create dataset_config.yaml training configuration."""
    use_t2i = exp_config.get('use_t2i', False)
    vae_size = exp_config.get('vae_size', global_config.get('default_vae_size', [768, 512]))
    vit_size = exp_config.get('vit_size', global_config.get('default_vit_size', [336, 224]))
    t2i_ratio = exp_config.get('t2i_ratio', global_config.get('default_t2i_ratio', 0.2))
    multiref_ratio = 1.0 - t2i_ratio if use_t2i else 1.0
    
    max_input_pixels = exp_config.get('max_input_pixels', 
                                       global_config.get('default_max_input_pixels'))
    max_output_pixels = exp_config.get('max_output_pixels',
                                        global_config.get('default_max_output_pixels', 589824))
    
    vae_max_size, vae_min_size = vae_size
    vit_max_size, vit_min_size = vit_size
    
    # Build data_dirs and num_used_data
    data_dirs_yaml = ""
    num_used_data_yaml = ""
    parquet_info_paths = []
    
    for data_dir_info in multi_ref_data_dirs:
        abs_path = str(Path(data_dir_info['path']).resolve())
        data_dirs_yaml += f"  - {abs_path}\n"
        num_used_data_yaml += f"  - {data_dir_info['num_files']}\n"
        parquet_info_path = Path(data_dir_info['path']) / 'parquet_info.json'
        if parquet_info_path.exists():
            parquet_info_paths.append(str(parquet_info_path))
    
    # Merge parquet_info
    parquet_info_path_yaml = ""
    if parquet_info_paths:
        merged_parquet_info = {}
        for info_path in parquet_info_paths:
            if Path(info_path).exists():
                with open(info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                if isinstance(info, dict):
                    merged_parquet_info.update(info)
        
        if merged_parquet_info:
            merged_info_path = exp_dir / 'merged_parquet_info.json'
            with open(merged_info_path, 'w', encoding='utf-8') as f:
                json.dump(merged_parquet_info, f, indent=2, ensure_ascii=False)
            parquet_info_path_yaml = str(merged_info_path.resolve())
            print(f"Merged {len(parquet_info_paths)} parquet_info.json files into: {merged_info_path}")
    
    multi_ref_weight = int(multiref_ratio * 10)
    
    # Generate max_input_pixels YAML
    if max_input_pixels is not None:
        if isinstance(max_input_pixels, (list, tuple)):
            max_input_pixels_yaml = f"    max_input_pixels: [{', '.join(str(p) for p in max_input_pixels)}]"
        else:
            max_input_pixels_yaml = f"    max_input_pixels: {max_input_pixels}"
        max_input_pixels_line = f"\n{max_input_pixels_yaml}"
    else:
        max_input_pixels_line = ""
    
    config_content = f"""multi_ref:
  dataset_names: []
  data_dirs:
{data_dirs_yaml}  parquet_info_path: {parquet_info_path_yaml}
  num_used_data:
{num_used_data_yaml}  image_transform_args:
    image_stride: 16
    max_image_size: {vae_max_size}
    min_image_size: {vae_min_size}
    max_pixels: {max_output_pixels}{max_input_pixels_line}
  vit_image_transform_args:
    image_stride: 14
    max_image_size: {vit_max_size}
    min_image_size: {vit_min_size}{max_input_pixels_line}
  is_mandatory: true
  weight: {multi_ref_weight}
"""
    
    # Add T2I configuration
    if use_t2i and t2i_data_dir and t2i_data_dir.exists():
        t2i_parquet_info = t2i_data_dir / 'parquet_info.json'
        t2i_num_files = 10
        
        if t2i_parquet_info.exists():
            with open(t2i_parquet_info, 'r', encoding='utf-8') as f:
                t2i_info = json.load(f)
            t2i_num_files = t2i_info.get('num_files', 10)
        
        t2i_weight = int(t2i_ratio * 10)
        abs_t2i_dir = str(t2i_data_dir.resolve())
        abs_t2i_info = str(t2i_parquet_info.resolve())
        
        config_content += f"""
t2i_pretrain_path:
  dataset_names: []
  data_dirs:
  - {abs_t2i_dir}
  num_used_data:
  - {t2i_num_files}
  image_transform_args:
    image_stride: 16
    max_image_size: {vae_max_size}
    min_image_size: {vae_min_size}
  is_mandatory: true
  weight: {t2i_weight}
"""
    
    config_path = exp_dir / 'dataset_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"Created dataset_config.yaml: {config_path}")
    return config_path


def create_run_scripts(exp_name: str, exp_config: dict, global_config: dict, exp_dir: Path) -> None:
    """Create run.sh script."""

    # Use relative paths
    model_path = str(CKPTS_DIR / "BAGEL-7B-MoT")
    llm_path = str(CKPTS_DIR / "Qwen2.5-7B-Instruct")
    
    # Get training parameters
    learning_rate = exp_config.get('learning_rate', global_config.get('default_learning_rate', 2e-5))
    num_workers = exp_config.get('num_workers', global_config.get('default_num_workers', 1))

    # Create run.sh
    run_content = f'''#!/bin/bash
# Bagel training script - experiment name: {exp_name}
# Auto-generated by process_config.py

set -e

export TORCH_CPP_LOG_LEVEL=ERROR
export NCCL_DEBUG=WARN
ulimit -n 80000

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd {SOURCE_DIR}

export PYTHONPATH={SOURCE_DIR}:$PYTHONPATH
export WANDB_DIR="${{SCRIPT_DIR}}/wandb"

# Model paths (using relative paths)
MODEL_PATH="{model_path}"
LLM_PATH="{llm_path}"
DATASET_CONFIG="${{SCRIPT_DIR}}/dataset_config.yaml"
RESULTS_DIR="${{SCRIPT_DIR}}/results"
CHECKPOINT_DIR="${{RESULTS_DIR}}/checkpoints"

# Training parameters
RESUME_FROM="${{MODEL_PATH}}"
SHARDING_STRATEGY="FULL_SHARD"
LR={learning_rate}
LOG_EVERY=10
NUM_WORKER={num_workers}
SAVE_EVERY=5000
EXPECTED_NUM_TOKENS=32768
MAX_NUM_TOKENS=36864
MAX_NUM_TOKENS_PER_SAMPLE=32768

# Parse cluster info and set environment variables (if available)
if [ -n "$AFO_ENV_CLUSTER_SPEC" ]; then
    cluster_spec=${{AFO_ENV_CLUSTER_SPEC}}
    role=$(jq -r .role <<< "$cluster_spec")
    [ "$role" = "worker" ] || {{ echo "Error: $role vs worker" >&2; exit 1; }}

    node_rank=$(jq -r .index <<< "$cluster_spec")
    nnodes=$(jq -r ".worker | length" <<< "$cluster_spec")
    nproc_per_node=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')

    master=$(jq -r '.worker[0]' <<< "$cluster_spec")
    IFS=":" read -r master_addr ports <<< "$master"
    IFS="," read -ra master_ports <<< "$ports"
    master_port=${{master_ports[0]}}
else
    # Default values for local execution
    node_rank=0
    nnodes=1
    nproc_per_node=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ' || echo "1")
    master_addr="127.0.0.1"
    master_port=29500
fi

echo "========================================"
echo "Bagel Training - {exp_name}"
echo "========================================"
echo "Model path: $MODEL_PATH"
echo "LLM path: $LLM_PATH"
echo "Dataset config: $DATASET_CONFIG"
echo "Output dir: $RESULTS_DIR"
echo "Nodes: $nnodes, GPUs per node: $nproc_per_node"
echo "========================================"

mkdir -p "$RESULTS_DIR"
mkdir -p "$WANDB_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Run training
torchrun \\
  --nnodes=$nnodes \\
  --node_rank=$node_rank \\
  --nproc_per_node=$nproc_per_node \\
  --master_addr=$master_addr \\
  --master_port=$master_port \\
  train/pretrain_unified_navit.py \\
  --dataset_config_file ${{DATASET_CONFIG}} \\
  --model_path ${{MODEL_PATH}} \\
  --llm_path ${{LLM_PATH}} \\
  --layer_module Qwen2MoTDecoderLayer \\
  --max_latent_size 64 \\
  --resume-from ${{RESUME_FROM}} \\
  --finetune_from_hf True \\
  --auto_resume True \\
  --resume-model-only True \\
  --finetune-from-ema True \\
  --sharding_strategy ${{SHARDING_STRATEGY}} \\
  --log_every ${{LOG_EVERY}} \\
  --lr ${{LR}} \\
  --wandb_offline True \\
  --wandb_name "{exp_name}" \\
  --num_worker ${{NUM_WORKER}} \\
  --freeze_vit True \\
  --visual_gen True \\
  --save_every ${{SAVE_EVERY}} \\
  --vit_cond_dropout_prob 0.0 \\
  --expected_num_tokens ${{EXPECTED_NUM_TOKENS}} \\
  --max_num_tokens ${{MAX_NUM_TOKENS}} \\
  --max_num_tokens_per_sample ${{MAX_NUM_TOKENS_PER_SAMPLE}} \\
  --use_flex True \\
  --results_dir ${{RESULTS_DIR}} \\
  --checkpoint_dir ${{CHECKPOINT_DIR}}

echo "========================================"
echo "Training complete!"
echo "Output dir: $RESULTS_DIR"
echo "========================================"
'''
    
    run_sh_path = exp_dir / 'run.sh'
    with open(run_sh_path, 'w', encoding='utf-8') as f:
        f.write(run_content)
    run_sh_path.chmod(0o755)
    print(f"Created run.sh: {run_sh_path}")


def process_experiment(exp_name: str, remake_data: bool = False) -> None:
    """Process a single experiment configuration."""
    config = load_config()
    global_config = config.get('global', {})
    experiments = config.get('experiments', {})

    if exp_name not in experiments:
        print(f"Error: experiment '{exp_name}' not found in config file")
        print(f"Available experiments: {list(experiments.keys())}")
        sys.exit(1)

    exp_config = experiments[exp_name]
    use_t2i = exp_config.get('use_t2i', False)

    print(f"\n{'='*60}")
    print(f"Processing experiment: {exp_name}")
    print(f"Use T2I: {use_t2i}")
    print(f"Re-convert data: {remake_data}")
    print(f"{'='*60}")

    # Set experiment directory
    exp_dir = EXPS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_dir = exp_dir / 'parquet'
    parquet_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare T2I data
    t2i_data_dir = None
    if use_t2i:
        t2i_jsonl_path = global_config.get('t2i_data_path', '')
        if t2i_jsonl_path:
            t2i_output_dir = exp_dir / "t2i_parquet"
            if not remake_data and check_t2i_data_prepared(t2i_output_dir):
                print(f"T2I data already exists, skipping preparation: {t2i_output_dir}")
                t2i_data_dir = t2i_output_dir
            else:
                t2i_data_dir, _, _ = prepare_t2i_data(t2i_jsonl_path, t2i_output_dir)
    
    # Prepare MultiRef data
    multiref_data_root = Path(global_config.get('multiref_data_root', ''))
    if not multiref_data_root.is_absolute():
        multiref_data_root = MACRO_DIR / multiref_data_root
    
    data_config = exp_config.get('data_config', {})
    
    multi_ref_data_dirs = []
    multiref_statistics = []
    if data_config:
        print("\nPreparing MultiRef data...")
        multi_ref_data_dirs, multiref_statistics = prepare_multiref_data(
            multiref_data_root, data_config, parquet_dir,
            force_regenerate=remake_data
        )
    
    # Create dataset_config.yaml
    create_dataset_config_yaml(
        exp_name, exp_config, global_config,
        exp_dir, t2i_data_dir, multi_ref_data_dirs
    )
    
    # Create run scripts
    create_run_scripts(exp_name, exp_config, global_config, exp_dir)
    
    # Save experiment summary
    summary = {
        'exp_name': exp_name,
        'config': exp_config,
        'data_stats': {
            'multiref': multiref_statistics,
            't2i_enabled': use_t2i,
            't2i_dir': str(t2i_data_dir) if t2i_data_dir else None
        },
        'run_script': str(exp_dir / 'run.sh')
    }
    with open(exp_dir / 'experiment_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nExperiment '{exp_name}' processing complete!")
    print(f"  Data directory: {exp_dir}")
    print(f"  Training script: {exp_dir / 'run.sh'}")
    print(f"  Start training: bash {exp_dir / 'run.sh'}")


def list_experiments() -> None:
    """List all available experiments."""
    config = load_config()
    experiments = config.get('experiments', {})

    print("\n" + "="*60)
    print("Available experiments")
    print("="*60)

    if not experiments:
        print("  No experiments configured")
        return

    for name, exp in experiments.items():
        use_t2i = "T2I" if exp.get('use_t2i', False) else "No-T2I"
        tasks = list(exp.get('data_config', {}).keys())
        print(f"\n  {name}:")
        print(f"    T2I: {use_t2i}")
        if tasks:
            print(f"    Tasks: {', '.join(tasks)}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Bagel configuration processing script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--exp_name', type=str, help='Experiment name')
    parser.add_argument('--list', action='store_true', help='List all available experiments')
    parser.add_argument('--all', action='store_true', help='Process all experiments')
    parser.add_argument('--remake', action='store_true', help='Force re-convert data')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
    elif args.exp_name:
        process_experiment(args.exp_name, remake_data=args.remake)
    elif args.all:
        config = load_config()
        experiments = config.get('experiments', {})
        for exp_name in experiments:
            process_experiment(exp_name, remake_data=args.remake)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
