#!/usr/bin/env python3
"""
Qwen-Image-Edit 配置处理脚本 - 从config.yaml读取配置并生成训练所需文件

基于 DiffSynth-Studio 框架，保持原有的训练配置格式：
- metadata/: 数据元数据目录
- run.sh / run_local.sh: 训练启动脚本

去除hope相关逻辑，只保留本地运行（支持多机多卡）

使用方法:
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


# 目录配置（使用相对路径）
SCRIPT_DIR = Path(__file__).parent.resolve()
MACRO_DIR = SCRIPT_DIR.parent
CONFIG_FILE = SCRIPT_DIR / "config.yaml"
SOURCE_DIR = SCRIPT_DIR / "source"
EXPS_DIR = SCRIPT_DIR / "exps"
DATA_DIR = SCRIPT_DIR / "data"
CKPTS_DIR = MACRO_DIR / "ckpts"


def load_config() -> dict:
    """加载配置文件"""
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def prepare_t2i_data(t2i_jsonl_path: str, output_dir: Path, 
                     force_regenerate: bool = False) -> Optional[Path]:
    """准备T2I数据：从JSONL转换为DiffSynth格式"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "t2i.jsonl"
    
    # 检查是否已存在
    if not force_regenerate and output_file.exists():
        info_file = output_dir / "t2i_info.json"
        if info_file.exists():
            print(f"T2I数据已存在，跳过准备: {output_file}")
            return output_file
    
    t2i_path = Path(t2i_jsonl_path)
    if not t2i_path.is_absolute():
        t2i_path = MACRO_DIR / t2i_path
    
    if not t2i_path.exists():
        print(f"警告: T2I JSONL文件不存在: {t2i_jsonl_path}")
        return None
    
    print(f"准备T2I数据: {t2i_path} -> {output_file}")
    
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
                
                # 支持两种格式
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
                
                # 转换为绝对路径
                if not os.path.isabs(output_image_path):
                    output_image_path = str(MACRO_DIR / output_image_path)
                
                # DiffSynth格式
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
    
    print(f"T2I数据准备完成: 成功 {converted_count} 条, 跳过 {skipped_count} 条")
    
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
    """从目录中加载所有JSON数据"""
    if not json_dir.exists():
        return []
    
    data_list = []
    for json_file in sorted(json_dir.glob("*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f_in:
                data = json.load(f_in)
            
            input_images = data.get('input_images', [])
            output_image = data.get('output_image', '')
            
            # 转换为绝对路径
            abs_input_images = []
            for img in input_images:
                if not os.path.isabs(img):
                    abs_input_images.append(str(MACRO_DIR / img))
                else:
                    abs_input_images.append(img)
                    
            if output_image and not os.path.isabs(output_image):
                output_image = str(MACRO_DIR / output_image)
            
            # DiffSynth格式
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
    """根据目标数量调整数据"""
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
    """准备IC数据"""
    stats = {}
    
    for task, categories in data_config.items():
        stats[task] = {}
        for category, cat_config in categories.items():
            target_num = cat_config.get('data_num', 0)
            
            src_dir = multiref_data_root / task / "train" / category
            output_file = metadata_dir / f"{task}_{category}.jsonl"
            
            # 检查是否已存在
            if not force_regenerate and output_file.exists():
                info_file = metadata_dir / f"{task}_{category}_info.json"
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    stats[task][category] = (info.get('original', 0), info.get('final', 0))
                    print(f"  {task}/{category}: 数据已存在 ({info.get('final', 0)} 样本)")
                    continue
            
            print(f"  处理 {task}/{category} (目标: {target_num})...")
            
            data_list = []
            for json_file in sorted(src_dir.glob("*.json")):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    input_images = data.get('input_images', [])
                    if len(input_images) > max_edit_images:
                        continue
                    
                    # 转换为绝对路径
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
                print(f"    警告: 无数据")
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
            print(f"    完成: {original_count} -> {final_count} 样本")
    
    return stats


def create_combined_metadata(metadata_dir: Path, ic_stats: Dict, 
                              t2i_file: Optional[Path]) -> Path:
    """创建合并的数据元数据"""
    all_files = []
    
    # 计算相对路径
    def get_rel_path(target_path):
        try:
            return os.path.relpath(target_path, metadata_dir)
        except ValueError:
            return str(target_path)
    
    # 添加IC数据文件
    for task, categories in ic_stats.items():
        for category, (orig, final) in categories.items():
            if final > 0:
                rel_path = get_rel_path(metadata_dir / f"{task}_{category}.jsonl")
                all_files.append(rel_path)
    
    # 添加T2I数据文件
    if t2i_file and t2i_file.exists():
        rel_path = get_rel_path(t2i_file)
        all_files.append(rel_path)
    
    # 创建元数据列表文件
    metadata_list_path = metadata_dir / "metadata_list.txt"
    with open(metadata_list_path, 'w') as f:
        for file_path in all_files:
            f.write(file_path + '\n')
    
    return metadata_list_path


def create_run_scripts(exp_name: str, exp_config: dict, global_config: dict,
                       exp_dir: Path, metadata_dir: Path) -> None:
    """创建训练启动脚本"""

    # 获取训练参数
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

    # 模型路径（使用绝对路径）
    model_base_path = str(CKPTS_DIR.resolve())
    model_id = global_config.get('model_id', 'Qwen-Image-Edit-2511')

    if isinstance(max_input_pixels, list):
        max_input_pixels_str = ",".join(str(p) for p in max_input_pixels)
    else:
        max_input_pixels_str = str(max_input_pixels)

    # 使用绝对路径，避免 cd 到 DIFFSYNTH_DIR 后相对路径失效
    metadata_abs_path = str(metadata_dir.resolve())
    diffsynth_abs_path = str(SOURCE_DIR.resolve())

    # 创建 run.sh
    run_content = f'''#!/bin/bash
# Qwen-Image-Edit 训练脚本 - 实验名称: {exp_name}
# 由 process_config.py 自动生成
# 训练模式: 全量微调

set -e

export TORCH_CPP_LOG_LEVEL=ERROR
export NCCL_DEBUG=WARN
ulimit -n 80000

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR"

# DiffSynth-Studio 目录
DIFFSYNTH_DIR="{diffsynth_abs_path}"
cd "$DIFFSYNTH_DIR"

export PYTHONPATH="${{DIFFSYNTH_DIR}}:${{PYTHONPATH}}"

# 数据集配置（使用绝对路径，避免 cd 后路径失效）
DATASET_BASE_PATH=""
DATASET_METADATA_PATH="{metadata_abs_path}"

# 模型配置
export DIFFSYNTH_MODEL_BASE_PATH="{model_base_path}"
export DIFFSYNTH_SKIP_DOWNLOAD=true

# model_id_with_origin_paths 格式
MODEL_ID_WITH_ORIGIN_PATHS="{model_id}:transformer/diffusion_pytorch_model*.safetensors,{model_id}:text_encoder/model*.safetensors,{model_id}:vae/diffusion_pytorch_model.safetensors"

# tokenizer 和 processor 路径
TOKENIZER_PATH="${{DIFFSYNTH_MODEL_BASE_PATH}}/{model_id}/tokenizer"
PROCESSOR_PATH="${{DIFFSYNTH_MODEL_BASE_PATH}}/{model_id}/processor"

# 训练参数
MAX_PIXELS={max_pixels}
MAX_INPUT_PIXELS="{max_input_pixels_str}"
MAX_EDIT_IMAGES={max_edit_images}
LEARNING_RATE={learning_rate}
NUM_EPOCHS={num_epochs}
GRADIENT_ACCUMULATION_STEPS={gradient_accumulation_steps}

# 输出目录
OUTPUT_PATH="${{SCRIPT_DIR}}/results"

# 解析集群信息并设置环境变量 (如果存在)
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
    # 本地运行默认值
    node_rank=0
    nnodes=1
    nproc_per_node=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ' || echo "1")
    master_addr="127.0.0.1"
    master_port=29500
fi

TOTAL_PROCESSES=$((nnodes * nproc_per_node))

echo "============================================"
echo "开始训练... (全量微调)"
echo "============================================"
echo "节点数: $nnodes, 每节点GPU数: $nproc_per_node"
echo "数据文件: $DATASET_METADATA_PATH"
echo "模型基础路径: $DIFFSYNTH_MODEL_BASE_PATH"
echo "Model ID: $MODEL_ID_WITH_ORIGIN_PATHS"
echo "训练参数: max_pixels=$MAX_PIXELS, lr=$LEARNING_RATE, epochs=$NUM_EPOCHS"
echo "输出目录: $OUTPUT_PATH"
echo "============================================"

mkdir -p "$OUTPUT_PATH"

# 设置分布式环境变量
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export WORLD_SIZE=$TOTAL_PROCESSES

# 动态生成 DeepSpeed 配置文件
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

# 构建 max_input_pixels 参数
MAX_INPUT_PIXELS_ARG=""
if [[ -n "$MAX_INPUT_PIXELS" ]]; then
  MAX_INPUT_PIXELS_ARG="--max_input_pixels $MAX_INPUT_PIXELS"
fi

# 使用 accelerate launch 启动训练
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
echo "训练完成!"
echo "输出目录: $OUTPUT_PATH"
echo "============================================"
'''

    run_path = exp_dir / "run.sh"
    with open(run_path, 'w', encoding='utf-8') as f:
        f.write(run_content)
    run_path.chmod(0o755)
    print(f"创建 run.sh: {run_path}")


def process_experiment(exp_name: str, remake_data: bool = False) -> None:
    """处理单个实验配置"""
    config = load_config()
    global_config = config.get('global', {})
    experiments = config.get('experiments', {})
    
    if exp_name not in experiments:
        print(f"错误: 实验 '{exp_name}' 不存在")
        print(f"可用实验: {list(experiments.keys())}")
        sys.exit(1)
    
    exp_config = experiments[exp_name]
    use_t2i = exp_config.get('use_t2i', False)
    
    print(f"\n{'='*60}")
    print(f"处理实验: {exp_name}")
    print(f"使用T2I: {use_t2i}")
    print(f"{'='*60}")
    
    # 创建实验目录
    exp_dir = EXPS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建metadata目录
    metadata_dir = exp_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备T2I数据（保存在实验目录下，与实验隔离）
    t2i_file = None
    if use_t2i:
        t2i_jsonl_path = global_config.get('t2i_data_path', '')
        if t2i_jsonl_path:
            t2i_output_dir = exp_dir / "t2i"
            t2i_file = prepare_t2i_data(t2i_jsonl_path, t2i_output_dir, force_regenerate=remake_data)
    
    # 准备IC数据
    multiref_data_root = Path(global_config.get('multiref_data_root', ''))
    if not multiref_data_root.is_absolute():
        multiref_data_root = MACRO_DIR / multiref_data_root
    
    data_config = exp_config.get('data_config', {})
    max_edit_images = global_config.get('default_max_edit_images', 10)
    ic_stats = {}
    
    if data_config:
        print("\n准备IC数据...")
        ic_stats = prepare_ic_data(multiref_data_root, data_config, metadata_dir, 
                                    max_edit_images, force_regenerate=remake_data)
    
    # 创建合并的元数据
    create_combined_metadata(metadata_dir, ic_stats, t2i_file)
    
    # 创建运行脚本
    create_run_scripts(exp_name, exp_config, global_config, exp_dir, metadata_dir)
    
    # 保存实验摘要
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
    
    print(f"\n实验 '{exp_name}' 处理完成!")
    print(f"  数据目录: {exp_dir}")
    print(f"  训练脚本: {exp_dir / 'run.sh'}")
    print(f"  启动训练: bash {exp_dir / 'run.sh'}")


def list_experiments() -> None:
    """列出所有可用实验"""
    config = load_config()
    experiments = config.get('experiments', {})
    
    print("\n" + "="*60)
    print("可用实验列表")
    print("="*60)
    
    if not experiments:
        print("  没有配置任何实验")
        return
    
    for name, exp in experiments.items():
        use_t2i = "T2I" if exp.get('use_t2i', False) else "No-T2I"
        use_lora = "LoRA" if exp.get('use_lora', False) else "Full"
        tasks = list(exp.get('data_config', {}).keys())
        print(f"\n  {name}:")
        print(f"    T2I: {use_t2i}")
        print(f"    模式: {use_lora}")
        if tasks:
            print(f"    任务: {', '.join(tasks)}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit 配置处理脚本")
    parser.add_argument('--exp_name', type=str, help='实验名称')
    parser.add_argument('--list', action='store_true', help='列出所有可用实验')
    parser.add_argument('--all', action='store_true', help='处理所有实验')
    parser.add_argument('--remake', action='store_true', help='强制重新转换数据')
    
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
