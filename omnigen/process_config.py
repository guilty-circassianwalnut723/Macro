#!/usr/bin/env python3
"""
OmniGen 配置处理脚本 - 从config.yaml读取配置并生成训练所需文件

保持原有的训练配置格式：
- X2I2.yml: 训练配置文件
- ic.yml, t2i.yml, mix.yml: 数据配置文件
- run.sh: 训练启动脚本

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
    """准备T2I数据：从JSONL转换为OmniGen格式"""
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
                
                converted_data = {
                    'task_type': 't2i',
                    'instruction': instruction,
                    'output_image': output_image_path
                }
                
                f_out.write(json.dumps(converted_data, ensure_ascii=False) + '\n')
                converted_count += 1
                
            except Exception:
                skipped_count += 1
                continue
    
    print(f"T2I数据准备完成: 成功 {converted_count} 条, 跳过 {skipped_count} 条")
    
    # 生成info文件
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
            
            ic_data = {
                'task_type': 'ic',
                'instruction': data.get('prompt', ''),
                'input_images': abs_input_images,
                'output_image': output_image
            }
            data_list.append(ic_data)
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
                    exp_dir: Path, force_regenerate: bool = False) -> Dict[str, Dict[str, tuple]]:
    """准备IC数据"""
    stats = {}
    ic_dir = exp_dir / "multi-ic"
    ic_dir.mkdir(parents=True, exist_ok=True)
    
    for task, categories in data_config.items():
        stats[task] = {}
        for category, cat_config in categories.items():
            target_num = cat_config.get('data_num', 0)
            
            src_dir = multiref_data_root / task / "train" / category
            output_file = ic_dir / f"{task}_{category}.jsonl"
            
            # 检查是否已存在
            if not force_regenerate and output_file.exists():
                info_file = ic_dir / f"{task}_{category}_info.json"
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    stats[task][category] = (info.get('original', 0), info.get('final', 0))
                    print(f"  {task}/{category}: 数据已存在 ({info.get('final', 0)} 样本)")
                    continue
            
            print(f"  处理 {task}/{category} (目标: {target_num})...")
            
            data_list = load_json_data_from_dir(src_dir)
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
            
            # 保存info
            info_file = ic_dir / f"{task}_{category}_info.json"
            with open(info_file, 'w') as f:
                json.dump({'original': original_count, 'final': final_count}, f)
            
            stats[task][category] = (original_count, final_count)
            print(f"    完成: {original_count} -> {final_count} 样本")
    
    return stats


def create_data_yml_files(exp_dir: Path, ic_stats: Dict, t2i_file: Optional[Path], t2i_ratio: float = 0.2) -> Tuple[Path, Path, Path]:
    """创建 ic.yml, t2i.yml, mix.yml 数据配置文件"""
    
    ic_dir = exp_dir / "multi-ic"
    
    # 创建 ic.yml
    ic_yml_path = exp_dir / "ic.yml"
    with open(ic_yml_path, 'w', encoding='utf-8') as f:
        f.write('data:\n')
        for task, categories in ic_stats.items():
            for category, (orig, final) in categories.items():
                if final > 0:
                    abs_path = str((ic_dir / f'{task}_{category}.jsonl').resolve())
                    f.write('  - \n')
                    f.write(f"    path: '{abs_path}'\n")
                    f.write(f"    type: 'ic'\n")
                    f.write(f"    use_all: true\n")
    
    # 创建 t2i.yml
    t2i_yml_path = exp_dir / "t2i.yml"
    if t2i_file and t2i_file.exists():
        abs_path = str(t2i_file.resolve())
        with open(t2i_yml_path, 'w', encoding='utf-8') as f:
            f.write('data:\n')
            f.write('  - \n')
            f.write(f"    path: '{abs_path}'\n")
            f.write("    type: 't2i'\n")
            # 子配置只描述数据源本身，采样比例在 mix.yml 中由 ratio_to_all 控制
            f.write("    ratio: !!float 1.0\n")
    
    # 创建 mix.yml
    mix_yml_path = exp_dir / "mix.yml"
    with open(mix_yml_path, 'w', encoding='utf-8') as f:
        f.write('data:\n')

        # 添加T2I数据：使用 ratio_to_all 表示其占总数据的比例
        if t2i_file and t2i_file.exists():
            f.write('  - \n')
            f.write(f"    path: '{str(t2i_yml_path.resolve())}'\n")
            f.write("    type: 't2i'\n")
            f.write(f"    ratio_to_all: !!float {t2i_ratio}\n")

        # 添加IC数据：全量使用
        f.write('  - \n')
        f.write(f"    path: '{str(ic_yml_path.resolve())}'\n")
        f.write("    type: 'ic'\n")
        f.write("    use_all: true\n")
    
    return ic_yml_path, t2i_yml_path, mix_yml_path


def create_x2i2_yml(exp_name: str, exp_config: dict, global_config: dict,
                    exp_dir: Path, mix_yml_path: Path) -> Path:
    """创建 X2I2.yml 训练配置文件"""
    
    # 获取训练参数
    num_epochs = exp_config.get('num_epochs', global_config.get('default_num_epochs', 10))
    learning_rate = exp_config.get('learning_rate', global_config.get('default_learning_rate', 1e-6))
    batch_size = exp_config.get('batch_size', global_config.get('default_batch_size', 1))
    global_batch_size = exp_config.get('global_batch_size', global_config.get('default_global_batch_size', 64))
    gradient_accumulation_steps = exp_config.get('gradient_accumulation_steps', global_config.get('default_gradient_accumulation_steps', 2))
    
    max_input_pixels = exp_config.get('max_input_pixels', 
                                       global_config.get('default_max_input_pixels', 
                                                         [1048576, 1048576, 589824, 262144, 262144, 262144, 262144]))
    max_output_pixels = exp_config.get('max_output_pixels', 
                                        global_config.get('default_max_output_pixels', 1048576))
    
    # 模型路径（使用相对路径）
    vae_path = str(CKPTS_DIR / "FLUX.1-dev")
    text_encoder_path = str(CKPTS_DIR / "Qwen2.5-VL-3B-Instruct")
    model_path = str(CKPTS_DIR / "OmniGen2")
    
    if isinstance(max_input_pixels, list):
        max_input_pixels_str = f"[{', '.join(str(p) for p in max_input_pixels)}]"
    else:
        max_input_pixels_str = str(max_input_pixels)
    
    config_content = f"""name: {exp_name}
output_dir: {exp_dir / 'exp'}

seed: 2233
device_specific_seed: true
workder_specific_seed: true

data:
  data_path: {mix_yml_path}
  use_chat_template: true
  maximum_text_tokens: 888
  prompt_dropout_prob: !!float 0.0001
  ref_img_dropout_prob: !!float 0.5
  max_output_pixels: {max_output_pixels}
  max_input_pixels: {max_input_pixels_str}
  max_side_length: 2048
  
model:
  pretrained_vae_model_name_or_path: {vae_path}
  pretrained_text_encoder_model_name_or_path: {text_encoder_path}
  pretrained_model_path: {model_path}
  arch_opt:
    patch_size: 2
    in_channels: 16
    hidden_size: 2520
    num_layers: 32
    num_refiner_layers: 2
    num_attention_heads: 21
    num_kv_heads: 7
    multiple_of: 256
    norm_eps: !!float 1e-05
    axes_dim_rope: [40, 40, 40]
    axes_lens: [10000, 10000, 10000]
    text_feat_dim: 2048
    timestep_scale: !!float 1000
    max_ref_images: 10

transport:
  snr_type: lognorm
  do_shift: true
  dynamic_time_shift: true

train:
  global_batch_size: {global_batch_size}
  batch_size: {batch_size}
  gradient_accumulation_steps: {gradient_accumulation_steps}
  num_train_epochs: {num_epochs}
  dataloader_num_workers: 6
  learning_rate: !!float {learning_rate}
  scale_lr: false
  lr_scheduler: timm_constant_with_warmup
  warmup_t: 500
  warmup_lr_init: 1e-18
  warmup_prefix: true
  t_in_epochs: false
  resume_from_checkpoint: latest
  use_8bit_adam: false
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: !!float 0.01
  adam_epsilon: !!float 1e-08
  max_grad_norm: 1
  gradient_checkpointing: true
  set_grads_to_none: true
  allow_tf32: false
  mixed_precision: 'bf16'
  ema_decay: 0.0
  lora_ft: false
  lora_rank: 8
  lora_dropout: 0

val:
  validation_steps: 500
  train_visualization_steps: 1000

logger:
  log_with: [tensorboard,]
  checkpointing_steps: 5000
  checkpoints_total_limit: ~
  sliding_window_step: 1000
  sliding_window_size: 3

cache_dir:
resume_from_checkpoint: ~
"""
    
    x2i2_path = exp_dir / "X2I2.yml"
    with open(x2i2_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"创建 X2I2.yml: {x2i2_path}")
    return x2i2_path


def create_run_script(exp_name: str, exp_dir: Path, x2i2_path: Path) -> Path:
    """创建训练启动脚本"""
    
    # 计算相对路径
    def get_rel_path(target_path, base_dir):
        try:
            return os.path.relpath(target_path, base_dir)
        except ValueError:
            return str(target_path)
            
    rel_config_path = get_rel_path(x2i2_path, SOURCE_DIR)
    
    run_content = f'''#!/bin/bash
# OmniGen 训练脚本 - 实验名称: {exp_name}
# 由 process_config.py 自动生成

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd {SOURCE_DIR}

export PYTHONPATH={SOURCE_DIR}:$PYTHONPATH

# 配置文件
CONFIG_FILE="{rel_config_path}"
OUTPUT_DIR="{exp_dir / 'exp'}"

echo "========================================"
echo "OmniGen 训练 - {exp_name}"
echo "========================================"
echo "配置文件: $CONFIG_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

# 使用 DeepSpeed 启动训练
bash train_deepspeed_cluster.sh --config "$CONFIG_FILE" --experiment_name "{exp_name}"

echo "========================================"
echo "训练完成!"
echo "输出目录: $OUTPUT_DIR"
echo "========================================"
'''
    
    run_path = exp_dir / "run.sh"
    with open(run_path, 'w', encoding='utf-8') as f:
        f.write(run_content)
    run_path.chmod(0o755)
    print(f"创建 run.sh: {run_path}")
    
    return run_path


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
    t2i_ratio = exp_config.get('t2i_ratio', global_config.get('default_t2i_ratio', 0.2))
    
    print(f"\n{'='*60}")
    print(f"处理实验: {exp_name}")
    print(f"使用T2I: {use_t2i}")
    print(f"{'='*60}")
    
    # 创建实验目录
    exp_dir = EXPS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备T2I数据（每个实验单独保存）
    t2i_file = None
    if use_t2i:
        t2i_jsonl_path = global_config.get('t2i_data_path', '')
        if t2i_jsonl_path:
            # T2I 转换结果保存在实验自己的目录下
            t2i_output_dir = exp_dir / "t2i"
            t2i_file = prepare_t2i_data(t2i_jsonl_path, t2i_output_dir, force_regenerate=remake_data)

    # 准备IC数据（experiment 级配置优先）
    multiref_root_value = exp_config.get('multiref_data_root', global_config.get('multiref_data_root', ''))
    multiref_data_root = Path(multiref_root_value)
    if not multiref_data_root.is_absolute():
        multiref_data_root = MACRO_DIR / multiref_data_root

    
    data_config = exp_config.get('data_config', {})
    ic_stats = {}
    
    if data_config:
        print("\n准备IC数据...")
        ic_stats = prepare_ic_data(multiref_data_root, data_config, exp_dir, force_regenerate=remake_data)
    
    # 创建数据配置文件
    ic_yml, t2i_yml, mix_yml = create_data_yml_files(exp_dir, ic_stats, t2i_file, t2i_ratio)
    
    # 创建X2I2.yml
    x2i2_path = create_x2i2_yml(exp_name, exp_config, global_config, exp_dir, mix_yml)
    
    # 创建run.sh
    run_path = create_run_script(exp_name, exp_dir, x2i2_path)
    
    # 保存实验摘要
    summary = {
        'exp_name': exp_name,
        'config': exp_config,
        'ic_stats': ic_stats,
        't2i_file': str(t2i_file) if t2i_file else None,
        'x2i2_config': str(x2i2_path),
        'run_script': str(run_path)
    }
    with open(exp_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n实验 '{exp_name}' 处理完成!")
    print(f"  数据目录: {exp_dir}")
    print(f"  训练脚本: {run_path}")
    print(f"  启动训练: bash {run_path}")


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
        tasks = list(exp.get('data_config', {}).keys())
        print(f"\n  {name}:")
        print(f"    T2I: {use_t2i}")
        if tasks:
            print(f"    任务: {', '.join(tasks)}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="OmniGen 配置处理脚本")
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
