# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import subprocess


def create_logger(logging_dir, rank, filename="log"):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0 and logging_dir is not None:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(), 
                logging.FileHandler(f"{logging_dir}/{filename}.txt")
            ]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def get_latest_ckpt(checkpoint_dir):
    """
    获取最新的完整checkpoint。
    通过检查 scheduler.pt 是否存在来判断checkpoint是否完整。
    对于不完整的checkpoint（没有scheduler.pt的），会被删除。
    """
    step_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if len(step_dirs) == 0:
        return None
    
    # 按step排序（从大到小）
    step_dirs = sorted(step_dirs, key=lambda x: int(x), reverse=True)
    
    # 找到第一个包含 scheduler.pt 的完整checkpoint
    latest_valid_ckpt = None
    invalid_ckpts = []
    
    for step_dir in step_dirs:
        ckpt_path = os.path.join(checkpoint_dir, step_dir)
        scheduler_path = os.path.join(ckpt_path, "scheduler.pt")
        
        if os.path.exists(scheduler_path):
            # 找到第一个完整的checkpoint
            if latest_valid_ckpt is None:
                latest_valid_ckpt = ckpt_path
        else:
            # 记录不完整的checkpoint
            invalid_ckpts.append(ckpt_path)
    
    # 删除不完整的checkpoint（后台异步删除，不阻塞主进程）
    if invalid_ckpts:
        logger = logging.getLogger(__name__)
        for invalid_ckpt in invalid_ckpts:
            try:
                logger.warning(f"Deleting incomplete checkpoint (missing scheduler.pt): {invalid_ckpt}")
                # Use subprocess to delete the checkpoint in a separate process
                # This ensures that the main process is not blocked by file I/O
                subprocess.Popen(["rm", "-rf", invalid_ckpt])
            except Exception as e:
                logger.error(f"Failed to delete incomplete checkpoint {invalid_ckpt}: {e}")
    
    return latest_valid_ckpt
