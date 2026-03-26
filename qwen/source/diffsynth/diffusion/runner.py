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
    
    # 自动恢复：如果启用了 auto_resume，从最新的 checkpoint 恢复
    skip_steps = 0
    if args is not None and getattr(args, 'auto_resume', False):
        ckpt_path, resumed_steps = model_logger.try_resume_from_checkpoint()
        if ckpt_path is not None:
            # 加载 checkpoint 到模型
            state_dict = load_state_dict(ckpt_path)
            # 添加前缀恢复（保存时移除了前缀，加载时需要加回来）
            if args.remove_prefix_in_ckpt:
                prefix = args.remove_prefix_in_ckpt
                state_dict = {prefix + k: v for k, v in state_dict.items()}
            
            # 判断是否为 LoRA checkpoint（检查是否包含 lora 相关的 key）
            is_lora_ckpt = any("lora_A" in k or "lora_B" in k for k in state_dict.keys())
            if is_lora_ckpt:
                # 处理 LoRA 格式（转换为 peft 格式）
                state_dict = accelerator.unwrap_model(model).mapping_lora_state_dict(state_dict)
            
            load_result = accelerator.unwrap_model(model).load_state_dict(state_dict, strict=False)
            if accelerator.is_main_process:
                print(f"[auto_resume] 加载 checkpoint: {ckpt_path}, 共 {len(state_dict)} 个参数, is_lora={is_lora_ckpt}")
                if len(load_result.missing_keys) > 0:
                    print(f"[auto_resume] 缺失的参数: {len(load_result.missing_keys)} 个")
                    
                    # === 诊断：检查是否因为只保存了trainable参数导致缺失 ===
                    unwrapped_model = accelerator.unwrap_model(model)
                    ckpt_keys = set(state_dict.keys())
                    model_all_keys = set(unwrapped_model.state_dict().keys())
                    missing_keys_set = set(load_result.missing_keys)
                    
                    # 使用与保存相同的方法获取 trainable 参数的 keys（通过 requires_grad）
                    trainable_keys = unwrapped_model.trainable_param_names()
                    
                    # 如果有 remove_prefix，需要考虑 checkpoint 中的 keys 可能没有前缀
                    # 计算保存时会得到的 keys（移除前缀后）
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
                    
                    # 计算 non-trainable 参数
                    non_trainable_keys = model_all_keys - trainable_keys
                    
                    # 检查缺失的参数是否都是 non-trainable 的
                    missing_in_non_trainable = missing_keys_set & non_trainable_keys
                    missing_in_trainable = missing_keys_set & trainable_keys
                    
                    print(f"[auto_resume] === 诊断信息 ===")
                    print(f"[auto_resume] checkpoint 参数数量: {len(ckpt_keys)}")
                    print(f"[auto_resume] 模型全部参数数量: {len(model_all_keys)}")
                    print(f"[auto_resume] 模型 trainable 参数数量 (requires_grad=True): {len(trainable_keys)}")
                    print(f"[auto_resume] 模型 non-trainable 参数数量: {len(non_trainable_keys)}")
                    print(f"[auto_resume] 保存时预期 checkpoint keys 数量: {len(expected_ckpt_keys)}")
                    print(f"[auto_resume] 缺失参数中属于 non-trainable 的数量: {len(missing_in_non_trainable)}")
                    print(f"[auto_resume] 缺失参数中属于 trainable 的数量: {len(missing_in_trainable)}")
                    
                    # 判断是否是"只保存trainable"导致的问题
                    if len(missing_in_non_trainable) == len(missing_keys_set) and len(missing_in_trainable) == 0:
                        print(f"[auto_resume] ✓ 诊断结论: 缺失的参数全部是 non-trainable 参数，checkpoint 只保存了 trainable 参数，这是预期行为！")
                    elif len(missing_in_non_trainable) > 0 and len(missing_in_trainable) == 0:
                        print(f"[auto_resume] ✓ 诊断结论: 缺失的参数全部是 non-trainable 参数，这是预期行为！")
                    elif len(missing_in_non_trainable) > 0:
                        print(f"[auto_resume] ⚠ 诊断结论: 大部分缺失参数是 non-trainable，但仍有 {len(missing_in_trainable)} 个 trainable 参数缺失")
                        if len(missing_in_trainable) <= 10:
                            print(f"[auto_resume] 缺失的 trainable 参数: {missing_in_trainable}")
                    else:
                        print(f"[auto_resume] ✗ 诊断结论: 缺失的参数都是 trainable 参数，可能存在问题！")
                        if len(missing_in_trainable) <= 10:
                            print(f"[auto_resume] 缺失的 trainable 参数: {missing_in_trainable}")
                    
                    # 额外检查：checkpoint 中的 keys 是否与预期一致
                    # 注意：需要先还原前缀再比较，因为 state_dict 已经加上了前缀
                    ckpt_in_trainable = ckpt_keys & trainable_keys
                    ckpt_not_in_trainable = ckpt_keys - trainable_keys
                    print(f"[auto_resume] checkpoint 中属于 trainable 的参数: {len(ckpt_in_trainable)}")
                    print(f"[auto_resume] checkpoint 中不属于 trainable 的参数: {len(ckpt_not_in_trainable)}")
                    
                    # 检查是否 checkpoint keys 与预期完全匹配
                    if len(ckpt_in_trainable) == len(ckpt_keys) and len(ckpt_keys) == len(trainable_keys):
                        print(f"[auto_resume] ✓ checkpoint 与 trainable 参数完全匹配")
                    elif len(ckpt_in_trainable) == len(ckpt_keys):
                        print(f"[auto_resume] ⚠ checkpoint 是 trainable 参数的子集（可能保存了部分 trainable）")
                    else:
                        print(f"[auto_resume] ⚠ checkpoint 包含一些 non-trainable 参数，或者 key 命名不一致")
                        if len(ckpt_not_in_trainable) <= 5:
                            print(f"[auto_resume] 不属于 trainable 的 checkpoint 参数: {ckpt_not_in_trainable}")
                        else:
                            print(f"[auto_resume] 不属于 trainable 的 checkpoint 参数 (前5个): {list(ckpt_not_in_trainable)[:5]}")
                    
                    print(f"[auto_resume] === 诊断结束 ===")
                    
                if len(load_result.unexpected_keys) > 0:
                    print(f"[auto_resume] 意外的参数: {load_result.unexpected_keys}")
            skip_steps = resumed_steps
    
    # 计算需要跳过的 epoch 和 epoch 内的步数
    steps_per_epoch = len(dataloader)
    start_epoch = skip_steps // steps_per_epoch if skip_steps > 0 else 0
    skip_steps_in_epoch = skip_steps % steps_per_epoch if skip_steps > 0 else 0
    
    for epoch_id in range(start_epoch, num_epochs):
        # 使用 accelerate 的 skip_first_batches 高效跳过已训练的步数
        if epoch_id == start_epoch and skip_steps_in_epoch > 0:
            if accelerator.is_main_process:
                print(f"[auto_resume] 从 epoch {epoch_id} step {skip_steps_in_epoch} 恢复训练...")
            active_dataloader = accelerator.skip_first_batches(dataloader, skip_steps_in_epoch)
            step_offset = skip_steps_in_epoch
        else:
            active_dataloader = dataloader
            step_offset = 0
        
        for step_in_epoch, data in enumerate(tqdm(active_dataloader)):
            # 计算全局步数（考虑跳过的步数）
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
