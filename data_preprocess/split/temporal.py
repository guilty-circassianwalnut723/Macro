#!/usr/bin/env python3
"""
Temporal数据拆分脚本：从processed/temporal中采样数据，并拆分为train/eval数据

功能：
1. 读取processed/temporal目录下的所有jsonl文件
2. 根据dino_threshold切分帧序列
3. 根据帧数量分类为1-3/4-5/6-7/>=8（注意：实际类别是片段帧数量-1）
4. 用户只需指定eval数量，其他自动归类为train
5. 保存到data_hl02/split/temporal目录，格式为json（使用list）
6. 记录image num category统计信息
7. 如果数据量大，按文件拆分为多个子文件保存
"""

import json
import random
import time
from pathlib import Path

MACRO_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = MACRO_DIR / "data"
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

# ====== 配置参数 ======
DATA_DIR = DATA_DIR / "processed" / "temporal"
OUTPUT_DIR = DATA_DIR / "split" / "temporal"

# DINO阈值
DINO_THRESHOLD = 0.7

# Eval数量配置：{image_num_category: eval_count}
# 注意：image_num_category是实际图像数量（片段帧数量-1）
# 用户只需指定eval数量，train数量自动计算为（总数据量 - eval数量）
EVAL_COUNTS = {
    "1-3": 500,
    "4-5": 500,
    "6-7": 500,
    ">=8": 500,
}

# 每个json文件的最大样本数（用于拆分大文件）
MAX_SAMPLES_PER_FILE = 10000

# 随机种子
RANDOM_SEED = 42
# ======================


def load_jsonl(jsonl_path: Path) -> List[dict]:
    """加载jsonl文件"""
    debug = False  # 可以改为True开启详细debug
    samples = []
    if not jsonl_path.exists():
        if debug:
            print(f"[DEBUG] 文件不存在: {jsonl_path}")
        return samples
    
    start_time = time.time()
    if debug:
        print(f"[DEBUG] 开始读取文件: {jsonl_path}")
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                line_count += 1
                if line_count % 10000 == 0 and debug:
                    elapsed = time.time() - start_time
                    print(f"[DEBUG] 已读取 {line_count} 行，耗时 {elapsed:.2f}s: {jsonl_path.name}")
                
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'error' in data:
                        continue
                    samples.append(data)
                except json.JSONDecodeError as e:
                    if debug:
                        print(f"[DEBUG] 警告: 无法解析JSON行 {line_count}: {e}")
                    continue
    except Exception as e:
        print(f"[ERROR] 读取文件失败 {jsonl_path}: {e}")
        return samples
    
    elapsed = time.time() - start_time
    if debug or elapsed > 5.0:  # 如果读取超过5秒，自动输出debug信息
        print(f"[DEBUG] 读取完成: {jsonl_path.name}, {len(samples)} 个样本, 耗时 {elapsed:.2f}s")
    
    return samples


def split_segments_by_threshold(
    timestamps: List[int],
    dino_scores: List[Optional[float]],
    threshold: float,
) -> List[List[int]]:
    """根据相邻DINO分数和阈值，将timestamp序列切分成多个片段"""
    if not timestamps or len(timestamps) < 2:
        return []
    
    if len(dino_scores) != len(timestamps):
        if len(dino_scores) == len(timestamps) - 1:
            dino_scores = [None] + dino_scores
        else:
            print(f"警告: timestamps长度({len(timestamps)})与dino_scores长度({len(dino_scores)})不匹配")
            return []
    
    parts: List[List[int]] = []
    current: List[int] = [timestamps[0]]
    
    for i in range(1, len(timestamps)):
        prev_idx = i - 1
        cur_idx = i
        
        score = dino_scores[cur_idx]
        
        if score is not None and score < threshold:
            parts.append(current)
            current = [timestamps[cur_idx]]
        else:
            current.append(timestamps[cur_idx])
    
    if current:
        parts.append(current)
    
    return parts


def get_image_num_category(image_num: int) -> str:
    """根据图像数量返回类别名称"""
    if image_num <= 3:
        return "1-3"
    elif image_num <= 5:
        return "4-5"
    elif image_num <= 7:
        return "6-7"
    else:
        return ">=8"


def map_viewfs_path(viewfs_path: str, hdfs_mount: str = "/mnt/hdfs") -> str:
    """
    将 viewfs:// 路径映射为本地挂载路径。
    
    If the source data contains viewfs:// paths, this function strips the
    scheme and cluster prefix so that files can be accessed via a local HDFS
    mount point.  Set the ``hdfs_mount`` argument (or the ``HDFS_MOUNT``
    environment variable) to point to your local HDFS mount.

    Args:
        viewfs_path: The original path, which may start with viewfs://.
        hdfs_mount: Local HDFS mount point (default: /mnt/hdfs).

    Returns:
        Local path string.
    """
    import os, re
    hdfs_mount = os.environ.get("HDFS_MOUNT", hdfs_mount).rstrip("/")
    if viewfs_path.startswith("viewfs://"):
        # Strip "viewfs://<cluster-name>" prefix
        local_path = re.sub(r"^viewfs://[^/]+", hdfs_mount, viewfs_path)
        return local_path
    return viewfs_path


def load_source_file(yt_file: str) -> Optional[Union[List, List[str]]]:
    """
    一次性加载source_file（yt_file），支持JSON数组或按行读取
    
    Args:
        yt_file: YouTube文件路径
    
    Returns:
        JSON数组（List）或行列表（List[str]），失败返回None
    """
    debug = False  # 可以改为True开启详细debug
    max_file_size_mb = 10  # 如果文件超过10MB，输出警告
    
    if not yt_file:
        return None
    
    # 映射viewfs路径
    local_yt_file = map_viewfs_path(yt_file)
    yt_path = Path(local_yt_file)
    
    if not yt_path.exists():
        if debug:
            print(f"[DEBUG] yt_file不存在: {local_yt_file}")
        return None
    
    # 检查文件大小
    file_size_mb = 0
    try:
        file_size_mb = yt_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_file_size_mb:
            print(f"[WARNING] 文件较大 ({file_size_mb:.1f}MB)，可能读取较慢: {yt_path.name}")
    except Exception:
        pass
    
    # 尝试作为JSON文件读取
    start_time = time.time()
    try:
        with open(yt_path, 'r', encoding='utf-8') as f:
            # 尝试解析为JSON
            try:
                if debug:
                    print(f"[DEBUG] 尝试解析JSON: {yt_path.name}")
                
                data = json.load(f)
                elapsed = time.time() - start_time
                if elapsed > 2.0:  # 如果解析超过2秒，输出警告
                    size_info = f"大小: {file_size_mb:.1f}MB" if file_size_mb > 0 else "大小: 未知"
                    print(f"[WARNING] JSON解析耗时 {elapsed:.2f}s: {yt_path.name} ({size_info})")
                
                if isinstance(data, list):
                    print(f"[DEBUG] 成功加载JSON数组: {yt_path.name}, 长度: {len(data)}")
                    return data
                else:
                    # 如果不是数组，回退到按行读取
                    raise ValueError("Not a JSON array")
            except (json.JSONDecodeError, ValueError) as e:
                # 不是JSON格式，按行读取
                if debug:
                    print(f"[DEBUG] 不是JSON格式，按行读取: {yt_path.name}, 错误: {e}")
                
                f.seek(0)  # 重置文件指针
                read_start = time.time()
                lines = f.readlines()
                read_elapsed = time.time() - read_start
                
                if read_elapsed > 2.0:
                    print(f"[WARNING] 按行读取耗时 {read_elapsed:.2f}s: {yt_path.name} (行数: {len(lines)})")
                
                print(f"[DEBUG] 成功按行加载: {yt_path.name}, 行数: {len(lines)}")
                return lines
    except Exception as e:
        print(f"[ERROR] 读取yt_file失败 {yt_path}: {e}")
        import traceback
        if debug:
            traceback.print_exc()
    
    return None


def get_video_path_from_loaded_data(loaded_data: Union[List, List[str]], yt_line: int) -> Optional[str]:
    """
    从已加载的数据中获取视频路径（避免重复读取文件）
    
    Args:
        loaded_data: 已加载的数据（JSON数组或行列表）
        yt_line: 行号或JSON数组索引
    
    Returns:
        视频路径字符串或None
    """
    debug = False  # 可以改为True开启详细debug
    
    if loaded_data is None or yt_line < 0:
        return None
    
    try:
        # 判断是JSON数组还是行列表
        # readlines()返回的行列表：所有元素都是字符串，且通常包含换行符（或结尾是换行符）
        # JSON数组：元素可能是dict、str、int等，字符串值通常不包含换行符
        
        is_line_list = False
        if len(loaded_data) > 0:
            # 如果第一个元素是字典，肯定是JSON数组
            if isinstance(loaded_data[0], dict):
                is_line_list = False
            # 如果所有元素都是字符串，检查是否有换行符（readlines的特性）
            elif all(isinstance(item, str) for item in loaded_data):
                # 检查是否有行以换行符结尾（readlines的特性，除了最后一行）
                # 或者检查是否有行包含换行符
                has_newline = any('\n' in item or item.endswith('\n') or item.endswith('\r\n') 
                                 for item in loaded_data[:min(10, len(loaded_data))])
                is_line_list = has_newline
            else:
                # 混合类型，应该是JSON数组
                is_line_list = False
        
        if is_line_list:
            # 是行列表（readlines的结果，包含换行符）
            if yt_line >= len(loaded_data):
                if debug:
                    print(f"[DEBUG] yt_line ({yt_line}) 超出文件行数 ({len(loaded_data)})")
                return None
            
            video_path_str = loaded_data[yt_line].strip()
            
            if not video_path_str:
                return None
            
            video_path = Path(map_viewfs_path(video_path_str))
            
            if not video_path.exists():
                if debug:
                    print(f"[DEBUG] 视频路径不存在: {video_path}")
                return None
            
            return str(video_path)
        else:
            # 是JSON数组
            if yt_line >= len(loaded_data):
                if debug:
                    print(f"[DEBUG] yt_line ({yt_line}) 超出数组长度 ({len(loaded_data)})")
                return None
            
            item = loaded_data[yt_line]
            if isinstance(item, dict):
                # 尝试从多个可能的字段获取视频路径
                video_path_str = item.get('videoPath') or item.get('video_path') or item.get('path')
                if not video_path_str:
                    return None
            elif isinstance(item, str):
                video_path_str = item
            else:
                return None
            
            if not video_path_str:
                return None
            
            video_path = Path(map_viewfs_path(video_path_str))
            
            if not video_path.exists():
                if debug:
                    print(f"[DEBUG] 视频路径不存在: {video_path}")
                return None
            
            return str(video_path)
    except Exception as e:
        if debug:
            print(f"[ERROR] 从已加载数据获取视频路径失败: {e}")
        return None


def get_video_path(yt_file: str, yt_line: int) -> Optional[str]:
    """
    从yt_file和yt_line获取视频路径（兼容旧接口，会重复读取文件）
    
    注意：为了性能，应该使用 load_source_file + get_video_path_from_loaded_data 的组合
    
    Args:
        yt_file: YouTube文件路径
        yt_line: 行号或JSON数组索引
    
    Returns:
        视频路径字符串或None
    """
    loaded_data = load_source_file(yt_file)
    if loaded_data is None:
        return None
    return get_video_path_from_loaded_data(loaded_data, yt_line)


def collect_all_segments(
    data_dir: Path,
    dino_threshold: float,
) -> Dict[str, List[dict]]:
    """收集所有切分后的片段，按类别分组"""
    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 开始收集片段，数据目录: {data_dir}")
    
    segments_by_category = defaultdict(list)
    
    print(f"[DEBUG] 正在扫描目录: {data_dir}")
    try:
        video_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        print(f"[DEBUG] 找到 {len(video_dirs)} 个视频目录")
    except Exception as e:
        print(f"[ERROR] 扫描目录失败: {e}")
        import traceback
        traceback.print_exc()
        return segments_by_category
    
    # 按目录分组收集所有jsonl文件（每个目录对应一个相同的source_file）
    dir_jsonl_files = defaultdict(list)  # {video_dir: [jsonl_file1, jsonl_file2, ...]}
    print(f"[DEBUG] 正在收集所有jsonl文件并按目录分组...")
    for idx, video_dir in enumerate(video_dirs):
        if idx % 100 == 0:
            print(f"[DEBUG] 已扫描 {idx}/{len(video_dirs)} 个目录...")
        try:
            jsonl_files = sorted(video_dir.glob("*.jsonl"))
            if jsonl_files:
                dir_jsonl_files[video_dir] = jsonl_files
        except Exception as e:
            print(f"[ERROR] 扫描目录失败 {video_dir}: {e}")
            continue
    
    total_jsonl_files = sum(len(files) for files in dir_jsonl_files.values())
    total_segments = 0
    total_samples = 0
    
    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 扫描完成: {len(video_dirs)} 个视频目录，{len(dir_jsonl_files)} 个有效目录，{total_jsonl_files} 个jsonl文件")
    print(f"正在处理 {len(dir_jsonl_files)} 个视频目录，{total_jsonl_files} 个jsonl文件...")
    
    # 统一的进度条：按目录处理
    file_start_time = time.time()
    last_log_time = time.time()
    processed_files = 0
    
    with tqdm(dir_jsonl_files.items(), desc="收集片段", unit="dir") as pbar:
        for dir_idx, (video_dir, jsonl_files) in enumerate(pbar):
            current_time = time.time()
            
            # 每10秒输出一次详细进度
            if current_time - last_log_time >= 10.0:
                elapsed = current_time - file_start_time
                avg_time = elapsed / (dir_idx + 1) if dir_idx > 0 else 0
                remaining = avg_time * (len(dir_jsonl_files) - dir_idx - 1)
                print(f"\n[DEBUG] {datetime.now().strftime('%H:%M:%S')} 进度: {dir_idx + 1}/{len(dir_jsonl_files)} "
                      f"({dir_idx * 100 // len(dir_jsonl_files)}%), "
                      f"已耗时: {elapsed:.1f}s, 预计剩余: {remaining:.1f}s, "
                      f"当前目录: {video_dir.name[:30]}")
                last_log_time = current_time
            
            pbar.set_postfix({
                "目录": video_dir.name[:20] + "..." if len(video_dir.name) > 20 else video_dir.name,
                "文件数": len(jsonl_files),
                "片段数": total_segments
            })
            
            # 步骤1: 确定该目录对应的source_file（yt_file）
            # 通过读取第一个jsonl文件的第一个样本来确定
            source_file_yt_file = None
            source_file_loaded_data = None
            
            try:
                first_jsonl_file = jsonl_files[0]
                first_samples = load_jsonl(first_jsonl_file)
                if first_samples and len(first_samples) > 0:
                    source_file_yt_file = first_samples[0].get('yt_file', '')
                    if source_file_yt_file:
                        # 一次性加载source_file
                        source_file_load_start = time.time()
                        source_file_loaded_data = load_source_file(source_file_yt_file)
                        source_file_load_elapsed = time.time() - source_file_load_start
                        
                        if source_file_loaded_data is None:
                            print(f"\n[WARNING] 无法加载source_file: {source_file_yt_file} (目录: {video_dir.name})")
                        elif source_file_load_elapsed > 5.0:
                            data_size = len(source_file_loaded_data) if isinstance(source_file_loaded_data, list) else 0
                            print(f"\n[WARNING] source_file加载耗时 {source_file_load_elapsed:.2f}s: {Path(source_file_yt_file).name} "
                                  f"(目录: {video_dir.name}, 数据大小: {data_size})")
                        else:
                            data_size = len(source_file_loaded_data) if isinstance(source_file_loaded_data, list) else 0
                            print(f"\n[DEBUG] 已加载source_file: {Path(source_file_yt_file).name} (目录: {video_dir.name}, 数据大小: {data_size})")
            except Exception as e:
                print(f"\n[ERROR] 确定source_file失败 (目录: {video_dir.name}): {e}")
                import traceback
                traceback.print_exc()
                # 如果无法确定source_file，回退到原来的方式
                source_file_yt_file = None
                source_file_loaded_data = None
            
            # 步骤2: 处理该目录下的所有jsonl文件
            with tqdm(jsonl_files, desc=f"文件", unit="文件", leave=False,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as file_pbar:
                for jsonl_file in file_pbar:
                    processed_files += 1
                    file_pbar.set_description(f"文件 {jsonl_file.name[:30]}..." if len(jsonl_file.name) > 30 else f"文件 {jsonl_file.name}")
                    
                    try:
                        file_load_start = time.time()
                        samples = load_jsonl(jsonl_file)
                        file_load_elapsed = time.time() - file_load_start
                        
                        if file_load_elapsed > 5.0:  # 如果加载文件超过5秒，输出警告
                            print(f"\n[WARNING] 文件加载耗时 {file_load_elapsed:.2f}s: {jsonl_file.name} (样本数: {len(samples)})")
                        
                        total_samples += len(samples)
                        
                        # 更新文件进度条的后缀信息
                        file_pbar.set_postfix({
                            "样本数": len(samples),
                            "总片段": total_segments,
                            "已处理": f"{processed_files}/{total_jsonl_files}"
                        })
                        
                        # 处理每个样本 - 添加进度条
                        process_start = time.time()
                        video_path_count = 0
                        video_path_total_time = 0
                        
                        # 为每个 jsonl_file 的样本处理添加进度条
                        file_name_display = jsonl_file.name[:50] + "..." if len(jsonl_file.name) > 50 else jsonl_file.name
                        with tqdm(samples, desc=f"  样本", unit="样本", leave=False, 
                                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as sample_pbar:
                            for sample_idx, sample in enumerate(sample_pbar):
                                timestamps = sample.get('timestamp', [])
                                dino_scores = sample.get('dino_score', [])
                                yt_file = sample.get('yt_file', '')
                                yt_line = sample.get('yt_line', -1)
                                
                                # 验证yt_file是否与source_file一致
                                if source_file_yt_file and yt_file != source_file_yt_file:
                                    print(f"\n[WARNING] yt_file不一致: 期望={source_file_yt_file}, 实际={yt_file} "
                                          f"(文件: {jsonl_file.name}, 样本: {sample_idx})")
                                
                                if not timestamps or len(timestamps) < 2:
                                    continue
                                
                                parts = split_segments_by_threshold(timestamps, dino_scores, dino_threshold)
                                
                                for part_idx, part_timestamps in enumerate(parts):
                                    image_num = len(part_timestamps) - 1
                                    
                                    if image_num <= 0:
                                        continue
                                    
                                    category = get_image_num_category(image_num)
                                    
                                    # 获取视频路径 - 使用已加载的数据避免重复读取
                                    video_path_start = time.time()
                                    if source_file_loaded_data is not None:
                                        # 使用已加载的数据（推荐方式，避免重复读取）
                                        video_path = get_video_path_from_loaded_data(source_file_loaded_data, yt_line)
                                    else:
                                        # 回退到原来的方式（如果无法确定source_file）
                                        video_path = get_video_path(yt_file, yt_line)
                                    
                                    video_path_elapsed = time.time() - video_path_start
                                    video_path_total_time += video_path_elapsed
                                    video_path_count += 1
                                    
                                    # 如果单个get_video_path调用超过3秒，输出警告
                                    if video_path_elapsed > 3.0:
                                        print(f"\n[WARNING] get_video_path耗时 {video_path_elapsed:.2f}s: "
                                              f"yt_file={yt_file}, yt_line={yt_line}")
                                    
                                    segment_info = {
                                        'yt_file': yt_file,
                                        'yt_line': yt_line,
                                        'jsonl_file': str(jsonl_file),
                                        'part_idx': part_idx,
                                        'timestamps': part_timestamps,
                                        'image_num': image_num,
                                        'image_num_category': category,
                                        'video_path': video_path,  # 保存视频路径
                                    }
                                    
                                    segments_by_category[category].append(segment_info)
                                    total_segments += 1
                                
                                # 更新样本进度条的后缀信息，显示当前文件的片段数
                                sample_pbar.set_postfix({
                                    "片段数": total_segments,
                                    "文件": f"{processed_files}/{total_jsonl_files}"
                                })
                        
                        process_elapsed = time.time() - process_start
                        if process_elapsed > 10.0:  # 如果处理单个文件超过10秒，输出详细信息
                            avg_video_path_time = video_path_total_time / video_path_count if video_path_count > 0 else 0
                            print(f"\n[DEBUG] 处理文件耗时 {process_elapsed:.2f}s: {jsonl_file.name}, "
                                  f"样本数: {len(samples)}, 片段数: {total_segments}, "
                                  f"get_video_path调用: {video_path_count}次, 平均耗时: {avg_video_path_time:.3f}s")
                    
                    except Exception as e:
                        print(f"\n[ERROR] 处理文件失败 {jsonl_file}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            # 更新进度条的后缀信息，显示各类别的片段数
            category_info = ", ".join([f"{k}={len(v)}" for k, v in sorted(segments_by_category.items())[:3]])
            if len(segments_by_category) > 3:
                category_info += "..."
            pbar.set_postfix({
                "片段总数": total_segments,
                "类别": category_info
            })
    
    total_elapsed = time.time() - file_start_time
    print(f"\n[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 收集片段完成")
    print(f"处理完成: 总样本数={total_samples}, 总片段数={total_segments}, 总耗时={total_elapsed:.2f}s")
    
    return segments_by_category


def sample_segments(
    segments: List[dict],
    eval_count: int,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """从片段列表中采样eval样本，其余为train样本，尽可能从不同视频中采样"""
    print(f"[DEBUG] {datetime.now().strftime('%H:%M:%S')} 开始采样，总片段数: {len(segments)}, eval数量: {eval_count}")
    start_time = time.time()
    
    if eval_count is None or eval_count <= 0:
        print(f"[DEBUG] eval_count无效，返回所有片段作为train")
        return segments, []
    
    if len(segments) <= eval_count:
        print(f"警告: 可用片段数({len(segments)})小于等于eval数量({eval_count})，所有样本用于eval")
        return [], segments
    
    random.seed(seed)
    
    # 按视频分组
    print(f"[DEBUG] 正在按视频分组...")
    group_start = time.time()
    segments_by_video = defaultdict(list)
    for seg in segments:
        video_key = f"{seg['yt_file']}_{seg['yt_line']}"
        segments_by_video[video_key].append(seg)
    
    video_keys = list(segments_by_video.keys())
    group_elapsed = time.time() - group_start
    print(f"[DEBUG] 分组完成: {len(video_keys)} 个视频，耗时 {group_elapsed:.2f}s")
    
    if group_elapsed > 5.0:
        print(f"[WARNING] 分组耗时较长: {group_elapsed:.2f}s")
    
    print(f"[DEBUG] 正在打乱视频顺序...")
    shuffle_start = time.time()
    random.shuffle(video_keys)
    
    for video_key in video_keys:
        random.shuffle(segments_by_video[video_key])
    shuffle_elapsed = time.time() - shuffle_start
    print(f"[DEBUG] 打乱完成，耗时 {shuffle_elapsed:.2f}s")
    
    selected_eval_segments = []
    selected_seg_keys = set()
    
    print(f"[DEBUG] 开始采样eval样本...")
    sample_start = time.time()
    
    # 策略1: 如果视频数量够，每个视频只采一个
    if len(video_keys) >= eval_count:
        print(f"[DEBUG] 策略1: 视频数量({len(video_keys)}) >= eval数量({eval_count})")
        for idx, video_key in enumerate(video_keys[:eval_count]):
            if idx % 100 == 0:
                print(f"[DEBUG] 采样进度: {idx}/{eval_count}")
            video_segments = segments_by_video[video_key]
            for seg in video_segments:
                seg_key = f"{seg['yt_file']}_{seg['yt_line']}_{seg['part_idx']}"
                if seg_key not in selected_seg_keys:
                    selected_eval_segments.append(seg)
                    selected_seg_keys.add(seg_key)
                    break
    else:
        # 策略2: 视频数量不够，先从每个视频采一个
        print(f"[DEBUG] 策略2: 视频数量({len(video_keys)}) < eval数量({eval_count})")
        for video_key in video_keys:
            if len(selected_eval_segments) >= eval_count:
                break
            video_segments = segments_by_video[video_key]
            for seg in video_segments:
                seg_key = f"{seg['yt_file']}_{seg['yt_line']}_{seg['part_idx']}"
                if seg_key not in selected_seg_keys:
                    selected_eval_segments.append(seg)
                    selected_seg_keys.add(seg_key)
                    break
        
        # 如果还不够，继续从视频中采样
        iteration = 0
        while len(selected_eval_segments) < eval_count:
            iteration += 1
            if iteration % 10 == 0:
                print(f"[DEBUG] 采样迭代 {iteration}，已采样: {len(selected_eval_segments)}/{eval_count}")
            
            found_new = False
            for video_key in video_keys:
                if len(selected_eval_segments) >= eval_count:
                    break
                video_segments = segments_by_video[video_key]
                for seg in video_segments:
                    seg_key = f"{seg['yt_file']}_{seg['yt_line']}_{seg['part_idx']}"
                    if seg_key not in selected_seg_keys:
                        selected_eval_segments.append(seg)
                        selected_seg_keys.add(seg_key)
                        found_new = True
                        break
            if not found_new:
                print(f"[WARNING] 无法找到更多新样本，停止采样")
                break
    
    sample_elapsed = time.time() - sample_start
    print(f"[DEBUG] eval采样完成: {len(selected_eval_segments)} 个样本，耗时 {sample_elapsed:.2f}s")
    
    # 剩余的所有样本作为train
    print(f"[DEBUG] 正在构建train样本列表...")
    train_start = time.time()
    train_segments = []
    for idx, seg in enumerate(segments):
        if idx % 100000 == 0 and idx > 0:
            print(f"[DEBUG] 处理train样本进度: {idx}/{len(segments)}")
        seg_key = f"{seg['yt_file']}_{seg['yt_line']}_{seg['part_idx']}"
        if seg_key not in selected_seg_keys:
            train_segments.append(seg)
    
    train_elapsed = time.time() - train_start
    total_elapsed = time.time() - start_time
    print(f"[DEBUG] 采样完成: train={len(train_segments)}, eval={len(selected_eval_segments)}, "
          f"总耗时={total_elapsed:.2f}s (train构建: {train_elapsed:.2f}s)")
    
    return train_segments, selected_eval_segments


def save_json(samples: List[dict], output_path: Path):
    """保存样本到json文件（使用list格式）"""
    start_time = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        elapsed = time.time() - start_time
        file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
        
        if elapsed > 5.0 or file_size_mb > 50:
            print(f"[DEBUG] 保存文件耗时 {elapsed:.2f}s: {output_path.name}, "
                  f"样本数: {len(samples)}, 文件大小: {file_size_mb:.1f}MB")
    except Exception as e:
        print(f"[ERROR] 保存文件失败 {output_path}: {e}")
        import traceback
        traceback.print_exc()
        raise


def save_segments_in_chunks(
    segments: List[dict],
    output_dir: Path,
    split_type: str,
    category: str,
    max_samples_per_file: int = 10000,
):
    """保存片段到json文件（使用list格式），如果数据量大则拆分为多个文件"""
    print(f"[DEBUG] {datetime.now().strftime('%H:%M:%S')} 开始保存{split_type}数据，样本数: {len(segments)}")
    start_time = time.time()
    
    if not segments:
        print(f"[DEBUG] 没有{split_type}数据需要保存")
        return
    
    if len(segments) <= max_samples_per_file:
        # 单个文件，直接保存，不使用进度条
        output_file = output_dir / f"{split_type}.json"
        print(f"正在保存 {len(segments)} 个{split_type}样本到: {output_file.name}...", end="", flush=True)
        save_json(segments, output_file)
        elapsed = time.time() - start_time
        print(f" 完成 (耗时 {elapsed:.2f}s)")
    else:
        # 多个文件，使用进度条
        num_files = (len(segments) + max_samples_per_file - 1) // max_samples_per_file
        print(f"[DEBUG] 需要拆分为 {num_files} 个文件保存")
        with tqdm(total=num_files, desc=f"保存{split_type}数据", unit="file") as pbar:
            for file_idx in range(num_files):
                file_start = time.time()
                start_idx = file_idx * max_samples_per_file
                end_idx = min(start_idx + max_samples_per_file, len(segments))
                chunk = segments[start_idx:end_idx]
                
                output_file = output_dir / f"{split_type}_{file_idx:04d}.json"
                save_json(chunk, output_file)
                file_elapsed = time.time() - file_start
                
                if file_elapsed > 10.0:
                    print(f"\n[WARNING] 保存文件耗时 {file_elapsed:.2f}s: {output_file.name}")
                
                pbar.update(1)
                pbar.set_postfix({
                    "文件": f"{file_idx + 1}/{num_files}",
                    "样本数": len(chunk),
                    "累计": end_idx,
                    "总计": len(segments)
                })
        
        total_elapsed = time.time() - start_time
        print(f"[DEBUG] {datetime.now().strftime('%H:%M:%S')} 保存{split_type}数据完成，总耗时: {total_elapsed:.2f}s")


def main():
    """主函数"""
    main_start_time = time.time()
    print("=" * 80)
    print("Temporal数据拆分脚本")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"DINO阈值: {DINO_THRESHOLD}")
    print(f"Eval数量配置: {EVAL_COUNTS}")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"每个文件最大样本数: {MAX_SAMPLES_PER_FILE}")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    
    print(f"[DEBUG] 检查数据目录是否存在: {DATA_DIR}")
    if not DATA_DIR.exists():
        print(f"[ERROR] 数据目录不存在: {DATA_DIR}")
        return
    print(f"[DEBUG] 数据目录存在，继续...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] 输出目录已创建/确认: {OUTPUT_DIR}")
    
    # 1. 收集所有片段
    print("=" * 80)
    print("步骤1: 收集所有片段")
    print("=" * 80)
    step1_start = time.time()
    segments_by_category = collect_all_segments(DATA_DIR, DINO_THRESHOLD)
    step1_elapsed = time.time() - step1_start
    print(f"\n[DEBUG] 步骤1完成，耗时: {step1_elapsed:.2f}s ({step1_elapsed/60:.1f}分钟)")
    
    # 统计信息
    print(f"\n[DEBUG] {datetime.now().strftime('%H:%M:%S')} 开始统计类别信息...")
    statistics = {}
    for category, segments in segments_by_category.items():
        statistics[category] = len(segments)
        print(f"类别 {category}: {len(segments)} 个片段")
    print(f"[DEBUG] 统计完成，共 {len(statistics)} 个类别")
    
    # 2. 采样train/eval样本
    print("\n" + "=" * 80)
    print("步骤2: 采样train/eval样本")
    print("=" * 80)
    step2_start = time.time()
    
    all_train_segments = []
    all_eval_segments = []
    category_stats = {}  # {category: {train: count, eval: count}}
    
    categories = list(segments_by_category.items())
    with tqdm(categories, desc="采样类别", unit="category") as pbar:
        for category, segments in pbar:
            eval_count = EVAL_COUNTS.get(category, 0)
            
            pbar.set_postfix({
                "类别": category,
                "总数": len(segments),
                "目标eval": eval_count
            })
            
            print(f"\n处理类别: {category}")
            print(f"  可用片段数: {len(segments)}")
            print(f"  需要eval: {eval_count}")
            
            train_segments, eval_segments = sample_segments(
                segments, eval_count, RANDOM_SEED
            )
            
            print(f"  实际采样train: {len(train_segments)}, eval: {len(eval_segments)}")
            
            # 添加image_num_category字段到每个样本
            for seg in train_segments:
                seg['image_num_category'] = category
            for seg in eval_segments:
                seg['image_num_category'] = category
            
            all_train_segments.extend(train_segments)
            all_eval_segments.extend(eval_segments)
            
            category_stats[category] = {
                'train': len(train_segments),
                'eval': len(eval_segments),
                'total': len(segments)
            }
            
            # 更新进度条显示累计统计
            pbar.set_postfix({
                "类别": category,
                "总train": len(all_train_segments),
                "总eval": len(all_eval_segments)
            })
    
    step2_elapsed = time.time() - step2_start
    print(f"\n[DEBUG] 步骤2完成，耗时: {step2_elapsed:.2f}s ({step2_elapsed/60:.1f}分钟)")
    
    # 3. 保存数据
    print("\n" + "=" * 80)
    print("步骤3: 保存数据")
    print("=" * 80)
    step3_start = time.time()
    
    # 保存train数据（按文件拆分）
    if all_train_segments:
        save_segments_in_chunks(
            all_train_segments,
            OUTPUT_DIR,
            "train",
            "all",
            MAX_SAMPLES_PER_FILE
        )
    
    # 保存eval数据（按文件拆分）
    if all_eval_segments:
        save_segments_in_chunks(
            all_eval_segments,
            OUTPUT_DIR,
            "eval",
            "all",
            MAX_SAMPLES_PER_FILE
        )
    
    # 4. 保存统计信息（包含image_num_category统计）
    stats_data = {
        'total_statistics': statistics,
        'category_statistics': category_stats,
        'summary': {
            'total_train': len(all_train_segments),
            'total_eval': len(all_eval_segments),
            'total_all': len(all_train_segments) + len(all_eval_segments),
        },
        'image_num_category': category_stats,  # 记录image num category统计
    }
    
    step3_elapsed = time.time() - step3_start
    print(f"\n[DEBUG] 步骤3完成，耗时: {step3_elapsed:.2f}s ({step3_elapsed/60:.1f}分钟)")
    
    stats_file = OUTPUT_DIR / "statistics.json"
    print(f"[DEBUG] 正在保存统计信息到: {stats_file}")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)
    print(f"\n已保存统计信息到: {stats_file}")
    
    total_elapsed = time.time() - main_start_time
    print("\n" + "=" * 80)
    print("处理完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_elapsed:.2f}s ({total_elapsed/60:.1f}分钟, {total_elapsed/3600:.2f}小时)")
    print("=" * 80)
    print(f"输出目录: {OUTPUT_DIR}")
    print("\n统计信息:")
    print(f"  总Train样本数: {len(all_train_segments)}")
    print(f"  总Eval样本数: {len(all_eval_segments)}")
    print(f"  总样本数: {len(all_train_segments) + len(all_eval_segments)}")
    print("\n按image_num_category统计:")
    for category, stats in sorted(category_stats.items()):
        print(f"  {category}: 总计={stats['total']}, train={stats['train']}, eval={stats['eval']}")
    print("\n各步骤耗时:")
    print(f"  步骤1 (收集片段): {step1_elapsed:.2f}s ({step1_elapsed/60:.1f}分钟)")
    print(f"  步骤2 (采样): {step2_elapsed:.2f}s ({step2_elapsed/60:.1f}分钟)")
    print(f"  步骤3 (保存): {step3_elapsed:.2f}s ({step3_elapsed/60:.1f}分钟)")
    print("=" * 80)


if __name__ == "__main__":
    main()
