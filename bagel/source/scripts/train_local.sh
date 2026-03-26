# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# --------------------------sft--------------------------
# Set default paths if not provided by environment variables
DATASET_CONFIG=${DATASET_CONFIG:-"../data/configs/example.yaml"}
MODEL_PATH=${MODEL_PATH:-"../../ckpts/BAGEL-7B-MoT"}
LLM_PATH=${LLM_PATH:-"../../ckpts/Qwen/Qwen2.5-7B-Instruct"}

torchrun \
  --nproc_per_node=4 \
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
  --num_shard 4 \
  --vit_cond_dropout_prob 0. \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240

