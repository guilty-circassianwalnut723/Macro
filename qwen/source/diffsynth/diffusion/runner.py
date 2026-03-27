import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from ..core import load_state_dict


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    # Auto-resume: if auto_resume is enabled, resume from the latest checkpoint
    skip_steps = 0
    if args is not None and getattr(args, 'auto_resume', False):
        ckpt_path, resumed_steps = model_logger.try_resume_from_checkpoint()
        if ckpt_path is not None:
            # Load checkpoint into model
            state_dict = load_state_dict(ckpt_path)
            # Add prefix back (prefix was removed during save, needs to be added back when loading)
            if args.remove_prefix_in_ckpt:
                prefix = args.remove_prefix_in_ckpt
                state_dict = {prefix + k: v for k, v in state_dict.items()}

            # Determine if this is a LoRA checkpoint (check for lora-related keys)
            is_lora_ckpt = any("lora_A" in k or "lora_B" in k for k in state_dict.keys())
            if is_lora_ckpt:
                # Handle LoRA format (convert to peft format)
                state_dict = accelerator.unwrap_model(model).mapping_lora_state_dict(state_dict)

            load_result = accelerator.unwrap_model(model).load_state_dict(state_dict, strict=False)
            if accelerator.is_main_process:
                print(f"[auto_resume] Loading checkpoint: {ckpt_path}, total {len(state_dict)} parameters, is_lora={is_lora_ckpt}")
                if len(load_result.missing_keys) > 0:
                    print(f"[auto_resume] Missing parameters: {len(load_result.missing_keys)}")

                    # === Diagnostics: check if missing is caused by only saving trainable parameters ===
                    unwrapped_model = accelerator.unwrap_model(model)
                    ckpt_keys = set(state_dict.keys())
                    model_all_keys = set(unwrapped_model.state_dict().keys())
                    missing_keys_set = set(load_result.missing_keys)

                    # Get trainable parameter keys using the same method as during save (via requires_grad)
                    trainable_keys = unwrapped_model.trainable_param_names()

                    # If remove_prefix is set, consider that checkpoint keys may not have the prefix
                    # Compute keys that would result from saving (after removing prefix)
                    if args.remove_prefix_in_ckpt:
                        prefix = args.remove_prefix_in_ckpt
                        expected_ckpt_keys = set()
                        for k in trainable_keys:
                            if k.startswith(prefix):
                                expected_ckpt_keys.add(k[len(prefix):])
                            else:
                                expected_ckpt_keys.add(k)
                    else:
                        expected_ckpt_keys = trainable_keys

                    # Compute non-trainable parameters
                    non_trainable_keys = model_all_keys - trainable_keys

                    # Check if missing parameters are all non-trainable
                    missing_in_non_trainable = missing_keys_set & non_trainable_keys
                    missing_in_trainable = missing_keys_set & trainable_keys

                    print(f"[auto_resume] === Diagnostic info ===")
                    print(f"[auto_resume] Checkpoint parameter count: {len(ckpt_keys)}")
                    print(f"[auto_resume] Model total parameter count: {len(model_all_keys)}")
                    print(f"[auto_resume] Model trainable parameter count (requires_grad=True): {len(trainable_keys)}")
                    print(f"[auto_resume] Model non-trainable parameter count: {len(non_trainable_keys)}")
                    print(f"[auto_resume] Expected checkpoint key count at save time: {len(expected_ckpt_keys)}")
                    print(f"[auto_resume] Missing params that are non-trainable: {len(missing_in_non_trainable)}")
                    print(f"[auto_resume] Missing params that are trainable: {len(missing_in_trainable)}")

                    # Determine if missing is caused by "only saved trainable parameters"
                    if len(missing_in_non_trainable) == len(missing_keys_set) and len(missing_in_trainable) == 0:
                        print(f"[auto_resume] Diagnosis: all missing parameters are non-trainable; checkpoint only saved trainable parameters - this is expected!")
                    elif len(missing_in_non_trainable) > 0 and len(missing_in_trainable) == 0:
                        print(f"[auto_resume] Diagnosis: all missing parameters are non-trainable - this is expected!")
                    elif len(missing_in_non_trainable) > 0:
                        print(f"[auto_resume] Warning: most missing parameters are non-trainable, but {len(missing_in_trainable)} trainable parameters are also missing")
                        if len(missing_in_trainable) <= 10:
                            print(f"[auto_resume] Missing trainable parameters: {missing_in_trainable}")
                    else:
                        print(f"[auto_resume] Error: all missing parameters are trainable - there may be an issue!")
                        if len(missing_in_trainable) <= 10:
                            print(f"[auto_resume] Missing trainable parameters: {missing_in_trainable}")

                    # Additional check: whether checkpoint keys match expectations
                    # Note: prefix needs to be restored before comparison since state_dict already has prefix added
                    ckpt_in_trainable = ckpt_keys & trainable_keys
                    ckpt_not_in_trainable = ckpt_keys - trainable_keys
                    print(f"[auto_resume] Checkpoint parameters that are trainable: {len(ckpt_in_trainable)}")
                    print(f"[auto_resume] Checkpoint parameters that are not trainable: {len(ckpt_not_in_trainable)}")

                    # Check if checkpoint keys exactly match expectations
                    if len(ckpt_in_trainable) == len(ckpt_keys) and len(ckpt_keys) == len(trainable_keys):
                        print(f"[auto_resume] Checkpoint exactly matches trainable parameters")
                    elif len(ckpt_in_trainable) == len(ckpt_keys):
                        print(f"[auto_resume] Warning: checkpoint is a subset of trainable parameters (may have saved partial trainable)")
                    else:
                        print(f"[auto_resume] Warning: checkpoint contains some non-trainable parameters, or key naming is inconsistent")
                        if len(ckpt_not_in_trainable) <= 5:
                            print(f"[auto_resume] Checkpoint parameters not in trainable: {ckpt_not_in_trainable}")
                        else:
                            print(f"[auto_resume] Checkpoint parameters not in trainable (first 5): {list(ckpt_not_in_trainable)[:5]}")

                    print(f"[auto_resume] === Diagnostics end ===")

                if len(load_result.unexpected_keys) > 0:
                    print(f"[auto_resume] Unexpected parameters: {load_result.unexpected_keys}")
            skip_steps = resumed_steps

    # Compute epoch and steps-within-epoch to skip
    steps_per_epoch = len(dataloader)
    start_epoch = skip_steps // steps_per_epoch if skip_steps > 0 else 0
    skip_steps_in_epoch = skip_steps % steps_per_epoch if skip_steps > 0 else 0

    for epoch_id in range(start_epoch, num_epochs):
        # Use accelerate's skip_first_batches to efficiently skip already-trained steps
        if epoch_id == start_epoch and skip_steps_in_epoch > 0:
            if accelerator.is_main_process:
                print(f"[auto_resume] Resuming training from epoch {epoch_id} step {skip_steps_in_epoch}...")
            active_dataloader = accelerator.skip_first_batches(dataloader, skip_steps_in_epoch)
            step_offset = skip_steps_in_epoch
        else:
            active_dataloader = dataloader
            step_offset = 0

        for step_in_epoch, data in enumerate(tqdm(active_dataloader)):
            # Compute global step (accounting for skipped steps)
            global_step = epoch_id * steps_per_epoch + step_offset + step_in_epoch + 1

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)

    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
