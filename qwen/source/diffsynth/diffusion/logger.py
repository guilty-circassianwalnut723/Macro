import os
import torch
import threading
import subprocess
from accelerate import Accelerator


def delete_file_async(file_path):
    """异步删除文件，不阻塞主进程"""
    if os.path.exists(file_path):
        try:
            subprocess.Popen(["rm", "-f", file_path])
        except Exception as e:
            print(f"[ModelLogger] 删除文件失败 {file_path}: {e}")


def get_latest_valid_ckpt(output_path):
    """
    获取最新的完整 step checkpoint 文件路径。
    只查找 step-*.safetensors 文件，不查找 epoch-*.safetensors。
    
    完整性判断：
    - .safetensors 文件存在且没有对应的 .safetensors.tmp 文件
    - 如果存在 .tmp 文件，说明保存被中断，该 checkpoint 不完整
    
    Args:
        output_path: checkpoint 根目录
    
    Returns:
        最新完整 checkpoint 的路径，如果没有则返回 None
    """
    if not os.path.exists(output_path):
        return None
    
    # 获取所有文件
    all_files = set(os.listdir(output_path))
    
    # 只获取 step-*.safetensors 文件（不查找 epoch-*）
    ckpt_items = []
    invalid_ckpts = []
    
    for item in all_files:
        if not item.endswith(".safetensors"):
            continue
        if not item.startswith("step-"):
            continue
            
        item_path = os.path.join(output_path, item)
        tmp_path = item_path + ".tmp"
        
        # 检查完整性：如果存在 .tmp 文件，说明保存被中断
        if os.path.exists(tmp_path):
            print(f"[get_latest_valid_ckpt] 发现不完整的 checkpoint: {item_path}")
            invalid_ckpts.append(item_path)
            invalid_ckpts.append(tmp_path)
            continue
        
        try:
            step_num = int(item.replace("step-", "").replace(".safetensors", ""))
            ckpt_items.append((step_num, item_path))
        except ValueError:
            continue
    
    # 清理孤立的 .tmp 文件（没有对应的 .safetensors 文件）
    for item in all_files:
        if item.endswith(".safetensors.tmp") and item.startswith("step-"):
            base_name = item[:-4]  # 移除 .tmp
            if base_name not in all_files:
                tmp_path = os.path.join(output_path, item)
                print(f"[get_latest_valid_ckpt] 发现孤立的临时文件: {tmp_path}")
                invalid_ckpts.append(tmp_path)
    
    # 异步删除不完整的 checkpoint
    for invalid_ckpt in invalid_ckpts:
        print(f"[get_latest_valid_ckpt] 删除不完整/临时文件: {invalid_ckpt}")
        threading.Thread(target=delete_file_async, args=(invalid_ckpt,)).start()
    
    if not ckpt_items:
        return None
    
    # 按 step 号排序（从大到小），返回最新的
    ckpt_items.sort(key=lambda x: x[0], reverse=True)
    return ckpt_items[0][1]


def get_step_from_ckpt_path(ckpt_path):
    """
    从 checkpoint 路径中提取 step 号
    
    Args:
        ckpt_path: checkpoint 文件路径
    
    Returns:
        step 号，如果无法解析则返回 0
    """
    if ckpt_path is None:
        return 0
    
    basename = os.path.basename(ckpt_path)
    try:
        if basename.startswith("step-"):
            return int(basename.replace("step-", "").replace(".safetensors", ""))
    except ValueError:
        pass
    return 0


class ModelLogger:
    def __init__(
        self, 
        output_path, 
        remove_prefix_in_ckpt=None, 
        state_dict_converter=lambda x: x,
        sliding_window_step=1000,
        sliding_window_size=5,
        save_steps=None,
    ):
        """
        ModelLogger - 处理模型 checkpoint 保存
        
        Args:
            output_path: checkpoint 保存目录
            remove_prefix_in_ckpt: 保存时移除的前缀
            state_dict_converter: state_dict 转换函数
            sliding_window_step: sliding checkpoint 保存间隔（步数）
            sliding_window_size: 保留的 sliding checkpoint 数量
            save_steps: 永久保存间隔（用于判断滑动 checkpoint）
        """
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        
        # Sliding checkpoint 配置
        self.sliding_window_step = sliding_window_step
        self.sliding_window_size = sliding_window_size
        self.save_steps = save_steps
        self.sliding_checkpoints = []  # 记录 sliding checkpoint 的 step 号
    
    def rebuild_sliding_checkpoints(self):
        """
        从目录重建 sliding_checkpoints 列表，用于训练恢复时。
        会自动删除超过窗口大小的旧滑动 checkpoint。
        
        滑动 checkpoint 的判断标准：
        - step % sliding_window_step == 0（是 sliding 保存点）
        - step % save_steps != 0（不是永久保存点）
        """
        if not os.path.exists(self.output_path):
            return
        
        if self.save_steps is None or self.sliding_window_step is None:
            print("[ModelLogger] save_steps 或 sliding_window_step 未设置，跳过重建")
            return
        
        # 获取所有 step-*.safetensors 文件
        all_files = os.listdir(self.output_path)
        all_ckpts = []
        for f in all_files:
            if f.startswith("step-") and f.endswith(".safetensors"):
                try:
                    step = int(f.replace("step-", "").replace(".safetensors", ""))
                    all_ckpts.append(step)
                except ValueError:
                    continue
        
        if not all_ckpts:
            return
        
        # 筛选出滑动 checkpoint（不是永久保存点的）
        sliding_ckpts = []
        for step in all_ckpts:
            # 是滑动保存点但不是永久保存点
            if step % self.sliding_window_step == 0 and step % self.save_steps != 0:
                sliding_ckpts.append(step)
        
        # 按 step 排序（从小到大）
        sliding_ckpts.sort()
        
        print(f"[ModelLogger] 发现 {len(sliding_ckpts)} 个滑动 checkpoint: {sliding_ckpts}")
        
        # 如果超过窗口大小，删除最旧的
        to_delete = []
        while len(sliding_ckpts) > self.sliding_window_size:
            old_step = sliding_ckpts.pop(0)
            to_delete.append(old_step)
        
        if to_delete:
            print(f"[ModelLogger] 需要删除 {len(to_delete)} 个旧的滑动 checkpoint: {to_delete}")
            for step in to_delete:
                old_ckpt_path = os.path.join(self.output_path, f"step-{step}.safetensors")
                threading.Thread(target=delete_file_async, args=(old_ckpt_path,)).start()
        
        self.sliding_checkpoints = sliding_ckpts
        print(f"[ModelLogger] 保留 {len(self.sliding_checkpoints)} 个滑动 checkpoint: {self.sliding_checkpoints}")

    def try_resume_from_checkpoint(self):
        """
        尝试从最新的 checkpoint 恢复训练状态
        
        Returns:
            tuple: (ckpt_path, num_steps) 如果找到有效的 checkpoint
                   (None, 0) 如果没有找到
        """
        ckpt_path = get_latest_valid_ckpt(self.output_path)
        if ckpt_path is not None:
            num_steps = get_step_from_ckpt_path(ckpt_path)
            self.num_steps = num_steps
            print(f"[ModelLogger] 找到 checkpoint: {ckpt_path}, 从 step {num_steps} 恢复")
            # 重建 sliding checkpoint 列表（用于训练恢复时清理多余的滑动 checkpoint）
            self.rebuild_sliding_checkpoints()
            return ckpt_path, num_steps
        return None, 0

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        """
        每个训练步结束时调用
        
        Args:
            accelerator: Accelerator 实例
            model: 训练模型
            save_steps: 永久保存间隔（如果设置）
        """
        self.num_steps += 1
        
        # 永久保存（按 save_steps 间隔）
        save_permanent = save_steps is not None and self.num_steps % save_steps == 0
        
        # Sliding 保存（按 sliding_window_step 间隔，但不与永久保存重叠）
        save_sliding = (
            self.sliding_window_step is not None 
            and self.num_steps % self.sliding_window_step == 0
            and not save_permanent  # 避免与永久保存重叠
        )
        
        if save_permanent:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors", is_permanent=True)
        
        if save_sliding:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors", is_permanent=False)
            self._manage_sliding_checkpoints(accelerator, self.num_steps)

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        """每个 epoch 结束时调用"""
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors", is_permanent=True)

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        """训练结束时调用"""
        # 如果最后一步没有保存过，保存最终状态
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors", is_permanent=True)

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name: str, is_permanent: bool = True):
        """
        保存模型 checkpoint
        
        使用临时文件+重命名的方式保证原子性：
        1. 先保存到 .tmp 临时文件
        2. 保存完成后重命名为正式文件
        如果保存过程中断，只会留下 .tmp 文件，不会破坏已有的 checkpoint
        
        Args:
            accelerator: Accelerator 实例
            model: 训练模型
            file_name: 保存的文件名（如 step-1000.safetensors）
            is_permanent: 是否为永久保存（影响日志输出）
        """
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
            final_path = os.path.join(self.output_path, file_name)
            tmp_path = final_path + ".tmp"
            
            save_type = "永久" if is_permanent else "滑动"
            print(f"[ModelLogger] 保存{save_type} checkpoint: {final_path}")
            
            # 获取并处理 state_dict
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                state_dict, remove_prefix=self.remove_prefix_in_ckpt
            )
            state_dict = self.state_dict_converter(state_dict)
            
            # 先保存到临时文件
            accelerator.save(state_dict, tmp_path, safe_serialization=True)
            
            # 保存成功后，重命名为正式文件（原子操作）
            os.rename(tmp_path, final_path)
        
        accelerator.wait_for_everyone()

    def _manage_sliding_checkpoints(self, accelerator: Accelerator, step: int):
        """
        管理 sliding checkpoint，保持最多 sliding_window_size 个
        
        Args:
            accelerator: Accelerator 实例
            step: 当前 step 号
        """
        if not accelerator.is_main_process:
            return
        
        self.sliding_checkpoints.append(step)
        
        # 如果超过窗口大小，删除最旧的
        while len(self.sliding_checkpoints) > self.sliding_window_size:
            old_step = self.sliding_checkpoints.pop(0)
            old_ckpt_path = os.path.join(self.output_path, f"step-{old_step}.safetensors")
            
            if os.path.exists(old_ckpt_path):
                print(f"[ModelLogger] 删除旧的滑动 checkpoint: {old_ckpt_path}")
                threading.Thread(target=delete_file_async, args=(old_ckpt_path,)).start()
