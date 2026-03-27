#!/usr/bin/env python3
"""
Qwen-Image-Edit configuration processing script - reads config from config.yaml and generates files required for training

Based on DiffSynth-Studio framework, preserving the original training configuration format:
- metadata/: data metadata directory
- run.sh / run_local.sh: training launch scripts

Removed hope-related logic, keeping only local execution (supports multi-node multi-GPU)

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
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Directory configuration (using relative paths)
SCRIPT_DIR = Path(__file__).parent.resolve()
MACRO_DIR = SCRIPT_DIR.parent
CONFIG_FILE = SCRIPT_DIR / "config.yaml"
SOURCE_DIR = SCRIPT_DIR / "source"
EXPS_DIR = SCRIPT_DIR / "exps"
DATA_DIR = SCRIPT_DIR / "data"
CKPTS_DIR = MACRO_DIR / "ckpts"


def load_config() -> dict:
    """Load configuration file"""
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def prepare_t2i_data(t2i_jsonl_path: str, output_dir: Path,
                     force_regenerate: bool = False) -> Optional[Path]:
    """Prepare T2I data: convert from JSONL to DiffSynth format"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "t2i.jsonl"

    # Check if already exists
    if not force_regenerate and output_file.exists():
        info_file = output_dir / "t2i_info.json"
        if info_file.exists():
            print(f"T2I data already exists, skipping preparation: {output_file}")
            return output_file

    t2i_path = Path(t2i_jsonl_path)
    if not t2i_path.is_absolute():
        t2i_path = MACRO_DIR / t2i_path

    if not t2i_path.exists():
        print(f"Warning: T2I JSONL file does not exist: {t2i_jsonl_path}")
        return None

    print(f"Preparing T2I data: {t2i_path} -> {output_file}")

    converted_count = 0
    skipped_count = 0

    with open(t2i_path, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

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

                if not instruction or not output_image_path:
                    skipped_count += 1
                    continue

                # Convert to absolute path
                if not os.path.isabs(output_image_path):
                    output_image_path = str(MACRO_DIR / output_image_path)

                # DiffSynth format
                converted_data = {
                    'prompt': instruction,
                    'image': output_image_path,
                    'edit_image': None
                }

                f_out.write(json.dumps(converted_data, ensure_ascii=False) + '\n')
                converted_count += 1

            except Exception:
                skipped_count += 1
                continue

    print(f"T2I data preparation complete: {converted_count} succeeded, {skipped_count} skipped")

    info_file = output_dir / "t2i_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump({
            'source': t2i_jsonl_path,
            'converted_file': str(output_file),
            'num_samples': converted_count,
            'skipped': skipped_count
        }, f, indent=2, ensure_ascii=False)

    return output_file


def load_json_data_from_dir(json_dir: Path) -> List[dict]:
    """Load all JSON data from directory"""
    if not json_dir.exists():
        return []

    data_list = []
    for json_file in sorted(json_dir.glob("*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f_in:
                data = json.load(f_in)

            input_images = data.get('input_images', [])
            output_image = data.get('output_image', '')

            # Convert to absolute paths
            abs_input_images = []
            for img in input_images:
                if not os.path.isabs(img):
                    abs_input_images.append(str(MACRO_DIR / img))
                else:
                    abs_input_images.append(img)

            if output_image and not os.path.isabs(output_image):
                output_image = str(MACRO_DIR / output_image)

            # DiffSynth format
            ds_data = {
                'prompt': data.get('prompt', ''),
                'image': output_image,
                'edit_image': abs_input_images
            }
            data_list.append(ds_data)
        except Exception:
            continue

    return data_list


def adjust_data_to_target_num(data_list: List[dict], target_num: int) -> List[dict]:
    """Adjust data according to target count"""
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


def prepare_ic_data(multiref_data_root: Path, data_config: dict,
                    metadata_dir: Path, max_edit_images: int,
                    force_regenerate: bool = False) -> Dict[str, Dict[str, tuple]]:
    """Prepare IC data"""
    stats = {}

    for task, categories in data_config.items():
        stats[task] = {}
        for category, cat_config in categories.items():
            target_num = cat_config.get('data_num', 0)

            src_dir = multiref_data_root / task / "train" / category
            output_file = metadata_dir / f"{task}_{category}.jsonl"

            # Check if already exists
            if not force_regenerate and output_file.exists():
                info_file = metadata_dir / f"{task}_{category}_info.json"
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    stats[task][category] = (info.get('original', 0), info.get('final', 0))
                    print(f"  {task}/{category}: data already exists ({info.get('final', 0)} samples)")
                    continue

            print(f"  Processing {task}/{category} (target: {target_num})...")

            data_list = []
            for json_file in sorted(src_dir.glob("*.json")):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    input_images = data.get('input_images', [])
                    if len(input_images) > max_edit_images:
                        continue

                    # Convert to absolute paths
                    abs_input_images = []
                    for img in input_images:
                        if img and not os.path.isabs(img):
                            abs_input_images.append(str(MACRO_DIR / img))
                        else:
                            abs_input_images.append(img)

                    output_image = data.get('output_image', '')
                    if output_image and not os.path.isabs(output_image):
                        output_image = str(MACRO_DIR / output_image)

                    ds_data = {
                        'prompt': data.get('prompt', ''),
                        'image': output_image,
                        'edit_image': abs_input_images
                    }
                    data_list.append(ds_data)
                except Exception:
                    continue

            original_count = len(data_list)

            if not data_list:
                print(f"    Warning: no data")
                stats[task][category] = (0, 0)
                continue

            adjusted_data = adjust_data_to_target_num(data_list, target_num)
            final_count = len(adjusted_data)

            with open(output_file, 'w', encoding='utf-8') as f_out:
                for item in adjusted_data:
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

            info_file = metadata_dir / f"{task}_{category}_info.json"
            with open(info_file, 'w') as f:
                json.dump({'original': original_count, 'final': final_count}, f)

            stats[task][category] = (original_count, final_count)
            print(f"    Done: {original_count} -> {final_count} samples")

    return stats


def create_combined_metadata(metadata_dir: Path, ic_stats: Dict,
                              t2i_file: Optional[Path]) -> Path:
    """Create combined data metadata"""
    all_files = []

    # Calculate relative paths
    def get_rel_path(target_path):
        try:
            return os.path.relpath(target_path, metadata_dir)
        except ValueError:
            return str(target_path)

    # Add IC data files
    for task, categories in ic_stats.items():
        for category, (orig, final) in categories.items():
            if final > 0:
                rel_path = get_rel_path(metadata_dir / f"{task}_{category}.jsonl")
                all_files.append(rel_path)

    # Add T2I data files
    if t2i_file and t2i_file.exists():
        rel_path = get_rel_path(t2i_file)
        all_files.append(rel_path)

    # Create metadata list file
    metadata_list_path = metadata_dir / "metadata_list.txt"
    with open(metadata_list_path, 'w') as f:
        for file_path in all_files:
            f.write(file_path + '\n')

    return metadata_list_path


def create_run_scripts(exp_name: str, exp_config: dict, global_config: dict,
                       exp_dir: Path, metadata_dir: Path) -> None:
    """Create training launch scripts"""

    # Get training parameters
    num_epochs = exp_config.get('num_epochs', global_config.get('default_num_epochs', 10))
    learning_rate = exp_config.get('learning_rate', global_config.get('default_learning_rate', 1e-5))
    gradient_accumulation_steps = exp_config.get('gradient_accumulation_steps', global_config.get('default_gradient_accumulation_steps', 1))

    max_pixels = exp_config.get('max_pixels', global_config.get('default_max_pixels', 589824))
    max_input_pixels = exp_config.get('max_input_pixels',
                                       global_config.get('default_max_input_pixels',
                                                         [1048576, 1048576, 589824, 589824, 589824,
                                                          262144, 262144, 262144, 262144, 262144]))
    max_edit_images = exp_config.get('max_edit_images',
                                      global_config.get('default_max_edit_images', 10))

    # Model path (using absolute path)
    model_base_path = str(CKPTS_DIR.resolve())
    model_id = global_config.get('model_id', 'Qwen-Image-Edit-2511')

    if isinstance(max_input_pixels, list):
        max_input_pixels_str = ",".join(str(p) for p in max_input_pixels)
    else:
        max_input_pixels_str = str(max_input_pixels)

    # Use absolute paths to avoid relative path failures after cd to DIFFSYNTH_DIR
    metadata_abs_path = str(metadata_dir.resolve())
    diffsynth_abs_path = str(SOURCE_DIR.resolve())

    # Create run.sh
    run_content = f'''#!/bin/bash
# Qwen-Image-Edit training script - experiment name: {exp_name}
# Auto-generated by process_config.py
# Training mode: full fine-tuning

set -e

export TORCH_CPP_LOG_LEVEL=ERROR
export NCCL_DEBUG=WARN
ulimit -n 80000

# Get directory of this script
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR"

# DiffSynth-Studio directory
DIFFSYNTH_DIR="{diffsynth_abs_path}"
cd "$DIFFSYNTH_DIR"

export PYTHONPATH="${{DIFFSYNTH_DIR}}:${{PYTHONPATH}}"

# Dataset configuration (using absolute paths to avoid failures after cd)
DATASET_BASE_PATH=""
DATASET_METADATA_PATH="{metadata_abs_path}"

# Model configuration
export DIFFSYNTH_MODEL_BASE_PATH="{model_base_path}"
export DIFFSYNTH_SKIP_DOWNLOAD=true

# model_id_with_origin_paths format
MODEL_ID_WITH_ORIGIN_PATHS="{model_id}:transformer/diffusion_pytorch_model*.safetensors,{model_id}:text_encoder/model*.safetensors,{model_id}:vae/diffusion_pytorch_model.safetensors"

# tokenizer and processor paths
TOKENIZER_PATH="${{DIFFSYNTH_MODEL_BASE_PATH}}/{model_id}/tokenizer"
PROCESSOR_PATH="${{DIFFSYNTH_MODEL_BASE_PATH}}/{model_id}/processor"

# Training parameters
MAX_PIXELS={max_pixels}
MAX_INPUT_PIXELS="{max_input_pixels_str}"
MAX_EDIT_IMAGES={max_edit_images}
LEARNING_RATE={learning_rate}
NUM_EPOCHS={num_epochs}
GRADIENT_ACCUMULATION_STEPS={gradient_accumulation_steps}

# Output directory
OUTPUT_PATH="${{SCRIPT_DIR}}/results"

# Parse cluster info and set environment variables (if present)
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

TOTAL_PROCESSES=$((nnodes * nproc_per_node))

echo "============================================"
echo "Starting training... (full fine-tuning)"
echo "============================================"
echo "Nodes: $nnodes, GPUs per node: $nproc_per_node"
echo "Data file: $DATASET_METADATA_PATH"
echo "Model base path: $DIFFSYNTH_MODEL_BASE_PATH"
echo "Model ID: $MODEL_ID_WITH_ORIGIN_PATHS"
echo "Training params: max_pixels=$MAX_PIXELS, lr=$LEARNING_RATE, epochs=$NUM_EPOCHS"
echo "Output directory: $OUTPUT_PATH"
echo "============================================"

mkdir -p "$OUTPUT_PATH"

# Set distributed environment variables
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export WORLD_SIZE=$TOTAL_PROCESSES

# Dynamically generate DeepSpeed config file
DEEPSPEED_CONFIG="${{SCRIPT_DIR}}/deepspeed_config.json"
cat > "$DEEPSPEED_CONFIG" << EOF
{{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
  "gradient_clipping": 1.0,
  "zero_optimization": {{
    "stage": 2,
    "offload_optimizer": {{ "device": "none" }},
    "offload_param": {{ "device": "none" }},
    "overlap_comm": true,
    "reduce_scatter": true,
    "contiguous_gradients": true,
    "allgather_partitions": true
  }},
  "fp16": {{ "enabled": false }},
  "bf16": {{ "enabled": true }}
}}
EOF

# Build max_input_pixels argument
MAX_INPUT_PIXELS_ARG=""
if [[ -n "$MAX_INPUT_PIXELS" ]]; then
  MAX_INPUT_PIXELS_ARG="--max_input_pixels $MAX_INPUT_PIXELS"
fi

# Launch training with accelerate launch
accelerate launch \\
  --machine_rank=$node_rank \\
  --main_process_ip=$master_addr \\
  --main_process_port=$master_port \\
  --num_machines=$nnodes \\
  --num_processes=$TOTAL_PROCESSES \\
  --mixed_precision=bf16 \\
  examples/qwen_image/model_training/train.py \\
  --use_deepspeed \\
  --deepspeed_config "$DEEPSPEED_CONFIG" \\
  --dataset_base_path "$DATASET_BASE_PATH" \\
  --dataset_metadata_path "$DATASET_METADATA_PATH" \\
  --data_file_keys "image,edit_image" \\
  --extra_inputs "edit_image" \\
  --max_pixels $MAX_PIXELS \\
  $MAX_INPUT_PIXELS_ARG \\
  --max_edit_images $MAX_EDIT_IMAGES \\
  --dataset_repeat 1 \\
  --model_id_with_origin_paths "$MODEL_ID_WITH_ORIGIN_PATHS" \\
  --tokenizer_path "$TOKENIZER_PATH" \\
  --processor_path "$PROCESSOR_PATH" \\
  --learning_rate $LEARNING_RATE \\
  --num_epochs $NUM_EPOCHS \\
  --remove_prefix_in_ckpt "pipe.dit." \\
  --output_path "$OUTPUT_PATH" \\
  --trainable_models "dit" \\
  --use_gradient_checkpointing \\
  --dataset_num_workers 8 \\
  --find_unused_parameters \\
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \\
  --save_steps 5000 \\
  --sliding_window_step 500 \\
  --sliding_window_size 3 \\
  --zero_cond_t

echo "============================================"
echo "Training complete!"
echo "Output directory: $OUTPUT_PATH"
echo "============================================"
'''

    run_path = exp_dir / "run.sh"
    with open(run_path, 'w', encoding='utf-8') as f:
        f.write(run_content)
    run_path.chmod(0o755)
    print(f"Created run.sh: {run_path}")


def process_experiment(exp_name: str, remake_data: bool = False) -> None:
    """Process a single experiment configuration"""
    config = load_config()
    global_config = config.get('global', {})
    experiments = config.get('experiments', {})

    if exp_name not in experiments:
        print(f"Error: experiment '{exp_name}' does not exist")
        print(f"Available experiments: {list(experiments.keys())}")
        sys.exit(1)

    exp_config = experiments[exp_name]
    use_t2i = exp_config.get('use_t2i', False)

    print(f"\n{'='*60}")
    print(f"Processing experiment: {exp_name}")
    print(f"Use T2I: {use_t2i}")
    print(f"{'='*60}")

    # Create experiment directory
    exp_dir = EXPS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata directory
    metadata_dir = exp_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Prepare T2I data (saved under experiment directory, isolated per experiment)
    t2i_file = None
    if use_t2i:
        t2i_jsonl_path = global_config.get('t2i_data_path', '')
        if t2i_jsonl_path:
            t2i_output_dir = exp_dir / "t2i"
            t2i_file = prepare_t2i_data(t2i_jsonl_path, t2i_output_dir, force_regenerate=remake_data)

    # Prepare IC data
    multiref_data_root = Path(global_config.get('multiref_data_root', ''))
    if not multiref_data_root.is_absolute():
        multiref_data_root = MACRO_DIR / multiref_data_root

    data_config = exp_config.get('data_config', {})
    max_edit_images = global_config.get('default_max_edit_images', 10)
    ic_stats = {}

    if data_config:
        print("\nPreparing IC data...")
        ic_stats = prepare_ic_data(multiref_data_root, data_config, metadata_dir,
                                    max_edit_images, force_regenerate=remake_data)

    # Create combined metadata
    create_combined_metadata(metadata_dir, ic_stats, t2i_file)

    # Create run scripts
    create_run_scripts(exp_name, exp_config, global_config, exp_dir, metadata_dir)

    # Save experiment summary
    summary = {
        'exp_name': exp_name,
        'config': exp_config,
        'ic_stats': ic_stats,
        't2i_file': str(t2i_file) if t2i_file else None,
        'metadata_dir': str(metadata_dir),
        'run_script': str(exp_dir / 'run.sh')
    }
    with open(exp_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nExperiment '{exp_name}' processing complete!")
    print(f"  Data directory: {exp_dir}")
    print(f"  Training script: {exp_dir / 'run.sh'}")
    print(f"  Launch training: bash {exp_dir / 'run.sh'}")


def list_experiments() -> None:
    """List all available experiments"""
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
        use_lora = "LoRA" if exp.get('use_lora', False) else "Full"
        tasks = list(exp.get('data_config', {}).keys())
        print(f"\n  {name}:")
        print(f"    T2I: {use_t2i}")
        print(f"    Mode: {use_lora}")
        if tasks:
            print(f"    Tasks: {', '.join(tasks)}")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit configuration processing script")
    parser.add_argument('--exp_name', type=str, help='experiment name')
    parser.add_argument('--list', action='store_true', help='list all available experiments')
    parser.add_argument('--all', action='store_true', help='process all experiments')
    parser.add_argument('--remake', action='store_true', help='force re-convert data')

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
