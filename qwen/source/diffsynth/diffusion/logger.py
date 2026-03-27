import os
import torch
import threading
import subprocess
from accelerate import Accelerator


def delete_file_async(file_path):
    """Asynchronously delete a file without blocking the main process"""
    if os.path.exists(file_path):
        try:
            subprocess.Popen(["rm", "-f", file_path])
        except Exception as e:
            print(f"[ModelLogger] Failed to delete file {file_path}: {e}")


def get_latest_valid_ckpt(output_path):
    """
    Get the path of the latest complete step checkpoint file.
    Only searches for step-*.safetensors files, not epoch-*.safetensors.

    Completeness check:
    - .safetensors file exists and has no corresponding .safetensors.tmp file
    - If a .tmp file exists, the save was interrupted and the checkpoint is incomplete

    Args:
        output_path: checkpoint root directory

    Returns:
        Path of the latest complete checkpoint, or None if not found
    """
    if not os.path.exists(output_path):
        return None

    # Get all files
    all_files = set(os.listdir(output_path))

    # Only get step-*.safetensors files (not epoch-*)
    ckpt_items = []
    invalid_ckpts = []

    for item in all_files:
        if not item.endswith(".safetensors"):
            continue
        if not item.startswith("step-"):
            continue

        item_path = os.path.join(output_path, item)
        tmp_path = item_path + ".tmp"

        # Check completeness: if .tmp file exists, save was interrupted
        if os.path.exists(tmp_path):
            print(f"[get_latest_valid_ckpt] Found incomplete checkpoint: {item_path}")
            invalid_ckpts.append(item_path)
            invalid_ckpts.append(tmp_path)
            continue

        try:
            step_num = int(item.replace("step-", "").replace(".safetensors", ""))
            ckpt_items.append((step_num, item_path))
        except ValueError:
            continue

    # Clean up orphaned .tmp files (no corresponding .safetensors file)
    for item in all_files:
        if item.endswith(".safetensors.tmp") and item.startswith("step-"):
            base_name = item[:-4]  # remove .tmp
            if base_name not in all_files:
                tmp_path = os.path.join(output_path, item)
                print(f"[get_latest_valid_ckpt] Found orphaned temp file: {tmp_path}")
                invalid_ckpts.append(tmp_path)

    # Asynchronously delete incomplete checkpoints
    for invalid_ckpt in invalid_ckpts:
        print(f"[get_latest_valid_ckpt] Deleting incomplete/temp file: {invalid_ckpt}")
        threading.Thread(target=delete_file_async, args=(invalid_ckpt,)).start()

    if not ckpt_items:
        return None

    # Sort by step number (descending), return the latest
    ckpt_items.sort(key=lambda x: x[0], reverse=True)
    return ckpt_items[0][1]


def get_step_from_ckpt_path(ckpt_path):
    """
    Extract the step number from a checkpoint path

    Args:
        ckpt_path: checkpoint file path

    Returns:
        step number, or 0 if unparseable
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
        ModelLogger - handles model checkpoint saving

        Args:
            output_path: checkpoint save directory
            remove_prefix_in_ckpt: prefix to remove when saving
            state_dict_converter: state_dict conversion function
            sliding_window_step: sliding checkpoint save interval (steps)
            sliding_window_size: number of sliding checkpoints to retain
            save_steps: permanent save interval (used to identify sliding checkpoints)
        """
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0

        # Sliding checkpoint configuration
        self.sliding_window_step = sliding_window_step
        self.sliding_window_size = sliding_window_size
        self.save_steps = save_steps
        self.sliding_checkpoints = []  # track sliding checkpoint step numbers

    def rebuild_sliding_checkpoints(self):
        """
        Rebuild sliding_checkpoints list from directory, used when resuming training.
        Automatically deletes sliding checkpoints exceeding the window size.

        Criteria for sliding checkpoints:
        - step % sliding_window_step == 0 (is a sliding save point)
        - step % save_steps != 0 (is not a permanent save point)
        """
        if not os.path.exists(self.output_path):
            return

        if self.save_steps is None or self.sliding_window_step is None:
            print("[ModelLogger] save_steps or sliding_window_step not set, skipping rebuild")
            return

        # Get all step-*.safetensors files
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

        # Filter out sliding checkpoints (not permanent save points)
        sliding_ckpts = []
        for step in all_ckpts:
            # Is a sliding save point but not a permanent save point
            if step % self.sliding_window_step == 0 and step % self.save_steps != 0:
                sliding_ckpts.append(step)

        # Sort by step (ascending)
        sliding_ckpts.sort()

        print(f"[ModelLogger] Found {len(sliding_ckpts)} sliding checkpoints: {sliding_ckpts}")

        # If exceeds window size, delete the oldest
        to_delete = []
        while len(sliding_ckpts) > self.sliding_window_size:
            old_step = sliding_ckpts.pop(0)
            to_delete.append(old_step)

        if to_delete:
            print(f"[ModelLogger] Need to delete {len(to_delete)} old sliding checkpoints: {to_delete}")
            for step in to_delete:
                old_ckpt_path = os.path.join(self.output_path, f"step-{step}.safetensors")
                threading.Thread(target=delete_file_async, args=(old_ckpt_path,)).start()

        self.sliding_checkpoints = sliding_ckpts
        print(f"[ModelLogger] Retaining {len(self.sliding_checkpoints)} sliding checkpoints: {self.sliding_checkpoints}")

    def try_resume_from_checkpoint(self):
        """
        Try to resume training state from the latest checkpoint

        Returns:
            tuple: (ckpt_path, num_steps) if a valid checkpoint is found
                   (None, 0) if not found
        """
        ckpt_path = get_latest_valid_ckpt(self.output_path)
        if ckpt_path is not None:
            num_steps = get_step_from_ckpt_path(ckpt_path)
            self.num_steps = num_steps
            print(f"[ModelLogger] Found checkpoint: {ckpt_path}, resuming from step {num_steps}")
            # Rebuild sliding checkpoint list (to clean up excess sliding checkpoints when resuming)
            self.rebuild_sliding_checkpoints()
            return ckpt_path, num_steps
        return None, 0

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        """
        Called at the end of each training step

        Args:
            accelerator: Accelerator instance
            model: training model
            save_steps: permanent save interval (if set)
        """
        self.num_steps += 1

        # Permanent save (at save_steps interval)
        save_permanent = save_steps is not None and self.num_steps % save_steps == 0

        # Sliding save (at sliding_window_step interval, but not overlapping with permanent saves)
        save_sliding = (
            self.sliding_window_step is not None
            and self.num_steps % self.sliding_window_step == 0
            and not save_permanent  # avoid overlap with permanent saves
        )

        if save_permanent:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors", is_permanent=True)

        if save_sliding:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors", is_permanent=False)
            self._manage_sliding_checkpoints(accelerator, self.num_steps)

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        """Called at the end of each epoch"""
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors", is_permanent=True)

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        """Called at the end of training"""
        # If the last step was not saved, save the final state
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors", is_permanent=True)

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name: str, is_permanent: bool = True):
        """
        Save model checkpoint

        Uses temp file + rename to ensure atomicity:
        1. First save to .tmp temp file
        2. After save completes, rename to the final file
        If the save is interrupted, only a .tmp file remains and existing checkpoints are not corrupted

        Args:
            accelerator: Accelerator instance
            model: training model
            file_name: file name to save (e.g. step-1000.safetensors)
            is_permanent: whether this is a permanent save (affects log output)
        """
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
            final_path = os.path.join(self.output_path, file_name)
            tmp_path = final_path + ".tmp"

            save_type = "permanent" if is_permanent else "sliding"
            print(f"[ModelLogger] Saving {save_type} checkpoint: {final_path}")

            # Get and process state_dict
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                state_dict, remove_prefix=self.remove_prefix_in_ckpt
            )
            state_dict = self.state_dict_converter(state_dict)

            # Save to temp file first
            accelerator.save(state_dict, tmp_path, safe_serialization=True)

            # After successful save, rename to final file (atomic operation)
            os.rename(tmp_path, final_path)

        accelerator.wait_for_everyone()

    def _manage_sliding_checkpoints(self, accelerator: Accelerator, step: int):
        """
        Manage sliding checkpoints, keeping at most sliding_window_size

        Args:
            accelerator: Accelerator instance
            step: current step number
        """
        if not accelerator.is_main_process:
            return

        self.sliding_checkpoints.append(step)

        # If exceeds window size, delete the oldest
        while len(self.sliding_checkpoints) > self.sliding_window_size:
            old_step = self.sliding_checkpoints.pop(0)
            old_ckpt_path = os.path.join(self.output_path, f"step-{old_step}.safetensors")

            if os.path.exists(old_ckpt_path):
                print(f"[ModelLogger] Deleting old sliding checkpoint: {old_ckpt_path}")
                threading.Thread(target=delete_file_async, args=(old_ckpt_path,)).start()
