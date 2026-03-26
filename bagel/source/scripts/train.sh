# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# --------------------------pretraining--------------------------
# torchrun \
#   --nnodes=$num_nodes \
#   --node_rank=$node_rank \
#   --nproc_per_node=8 \
#   --master_addr=$master_addr \
#   --master_port=$master_port \
#   train/pretrain_unified_navit.py \
#   --dataset_config_file ./data/configs/example.yaml \
#   --layer_module Qwen2MoTDecoderLayer \
#   --vae_path $vae_path \
#   --vit_path $vit_path \
#   --llm_path $llm_path \
#   --use_flex True \
#   --resume_from $resume_from \
#   --results_dir $output_path \
#   --checkpoint_dir $ckpt_path \
#   --max_latent_size 64  \
#   --num_workers 1 # use small num_workers since the num_used_data (10) are not enough to split

# --------------------------sft--------------------------
export TORCH_CPP_LOG_LEVEL=ERROR
export NCCL_DEBUG=WARN
ulimit -n 80000

# Set default paths if not provided by environment variables
DATASET_CONFIG=${DATASET_CONFIG:-"../data/configs/example.yaml"}
MODEL_PATH=${MODEL_PATH:-"../../ckpts/BAGEL-7B-MoT"}
LLM_PATH=${LLM_PATH:-"../../ckpts/Qwen/Qwen2.5-7B-Instruct"}

# 解析集群信息并设置环境变量 (如果存在)
if [ -n "$AFO_ENV_CLUSTER_SPEC" ]; then
    cluster_spec=${AFO_ENV_CLUSTER_SPEC}
    role=$(jq -r .role <<< "$cluster_spec")
    [ "$role" = "worker" ] || { echo "Error: $role vs worker" >&2; exit 1; }

    node_rank=$(jq -r .index <<< "$cluster_spec")
    nnodes=$(jq -r ".worker | length" <<< "$cluster_spec")
    nproc_per_node=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')

    master=$(jq -r '.worker[0]' <<< "$cluster_spec")
    IFS=":" read -r master_addr ports <<< "$master"
    IFS="," read -ra master_ports <<< "$ports"
    master_port=${master_ports[0]}
else
    # 本地运行默认值
    node_rank=0
    nnodes=1
    nproc_per_node=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
    master_addr="127.0.0.1"
    master_port=29500
fi

echo "nproc_per_node=$nproc_per_node, nnodes=$nnodes, node_rank=$node_rank, master_addr=$master_addr, master_port=$master_port"

export PYTHONPATH=$(pwd)

torchrun \
  --nnodes=$nnodes \
  --node_rank=$node_rank \
  --nproc_per_node=$nproc_per_node \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file $DATASET_CONFIG \
  --model_path $MODEL_PATH \
  --llm_path $LLM_PATH \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from $MODEL_PATH \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --num_worker 1 \
  --vit_cond_dropout_prob 0. \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240